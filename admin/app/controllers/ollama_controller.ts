import { ChatService } from '#services/chat_service'
import { DockerService } from '#services/docker_service'
import { OllamaService } from '#services/ollama_service'
import { RagService } from '#services/rag_service'
import Service from '#models/service'
import KVStore from '#models/kv_store'
import { modelNameSchema } from '#validators/download'
import { chatSchema, getAvailableModelsSchema } from '#validators/ollama'
import { inject } from '@adonisjs/core'
import type { HttpContext } from '@adonisjs/core/http'
import { DEFAULT_QUERY_REWRITE_MODEL, RAG_CONTEXT_LIMITS, SYSTEM_PROMPTS } from '../../constants/ollama.js'
import { SERVICE_NAMES } from '../../constants/service_names.js'
import logger from '@adonisjs/core/services/logger'
import path from 'node:path'
import { ChatSource } from '../../types/chat.js'
type Message = { role: 'system' | 'user' | 'assistant'; content: string }

@inject()
export default class OllamaController {
  constructor(
    private chatService: ChatService,
    private dockerService: DockerService,
    private ollamaService: OllamaService,
    private ragService: RagService
  ) { }

  async availableModels({ request }: HttpContext) {
    const reqData = await request.validateUsing(getAvailableModelsSchema)
    return await this.ollamaService.getAvailableModels({
      sort: reqData.sort,
      recommendedOnly: reqData.recommendedOnly,
      query: reqData.query || null,
      limit: reqData.limit || 15,
      force: reqData.force,
    })
  }

  async chat({ request, response }: HttpContext) {
    const reqData = await request.validateUsing(chatSchema)

    // Flush SSE headers immediately so the client connection is open while
    // pre-processing (query rewriting, RAG lookup) runs in the background.
    if (reqData.stream) {
      response.response.setHeader('Content-Type', 'text/event-stream')
      response.response.setHeader('Cache-Control', 'no-cache')
      response.response.setHeader('Connection', 'keep-alive')
      response.response.flushHeaders()
    }

    let chatSources: ChatSource[] = []
    try {
      // If there are no system messages in the chat inject system prompts
      const hasSystemMessage = reqData.messages.some((msg) => msg.role === 'system')
      if (!hasSystemMessage) {
        const systemPrompt = {
          role: 'system' as const,
          content: SYSTEM_PROMPTS.default,
        }
        logger.debug('[OllamaController] Injecting system prompt')
        reqData.messages.unshift(systemPrompt)
      }

      // Query rewriting for better RAG retrieval with manageable context
      // Will return user's latest message if no rewriting is needed
      const rewrittenQuery = await this.rewriteQueryWithContext(reqData.messages)

      logger.info(`[OllamaController] Rewritten query for RAG: "${rewrittenQuery}"`)
      if (rewrittenQuery) {
        const relevantDocs = await this.ragService.searchSimilarDocuments(
          rewrittenQuery,
          5, // Top 5 most relevant chunks
          0.3 // Minimum similarity score of 0.3
        )

        logger.info(`[RAG] Retrieved ${relevantDocs.length} relevant documents for query: "${rewrittenQuery}"`)

        // If relevant context is found, inject as a system message with adaptive limits
        if (relevantDocs.length > 0) {
          // Determine context budget based on model size
          const { maxResults, maxTokens } = this.getContextLimitsForModel(reqData.model)
          let trimmedDocs = relevantDocs.slice(0, maxResults)

          // Apply token cap if set (estimate ~3.5 chars per token)
          // Always include the first (most relevant) result — the cap only gates subsequent results
          if (maxTokens > 0) {
            const charCap = maxTokens * 3.5
            let totalChars = 0
            trimmedDocs = trimmedDocs.filter((doc, idx) => {
              totalChars += doc.text.length
              return idx === 0 || totalChars <= charCap
            })
          }

          logger.info(
            `[RAG] Injecting ${trimmedDocs.length}/${relevantDocs.length} results (model: ${reqData.model}, maxResults: ${maxResults}, maxTokens: ${maxTokens || 'unlimited'})`
          )

          // Build the user-facing sources list (deduped per article) from the same
          // chunks we're injecting into the prompt, so citations and LLM context stay aligned.
          chatSources = await this._buildChatSources(trimmedDocs)

          const contextText = trimmedDocs
            .map((doc, idx) => {
              const meta = doc.metadata ?? {}
              const label =
                meta.article_title || meta.full_title || (meta.source ? path.basename(meta.source) : `Context ${idx + 1}`)
              return `[Source ${idx + 1}: ${label}] (Relevance: ${(doc.score * 100).toFixed(1)}%)\n${doc.text}`
            })
            .join('\n\n')

          const systemMessage = {
            role: 'system' as const,
            content: SYSTEM_PROMPTS.rag_context(contextText),
          }

          // Insert system message at the beginning (after any existing system messages)
          const firstNonSystemIndex = reqData.messages.findIndex((msg) => msg.role !== 'system')
          const insertIndex = firstNonSystemIndex === -1 ? 0 : firstNonSystemIndex
          reqData.messages.splice(insertIndex, 0, systemMessage)
        }
      }

      // If system messages are large (e.g. due to RAG context), request a context window big
      // enough to fit them. Ollama respects num_ctx per-request; LM Studio ignores it gracefully.
      const systemChars = reqData.messages
        .filter((m) => m.role === 'system')
        .reduce((sum, m) => sum + m.content.length, 0)
      const estimatedSystemTokens = Math.ceil(systemChars / 3.5)
      let numCtx: number | undefined
      if (estimatedSystemTokens > 3000) {
        const needed = estimatedSystemTokens + 2048 // leave room for conversation + response
        numCtx = [8192, 16384, 32768, 65536].find((n) => n >= needed) ?? 65536
        logger.debug(`[OllamaController] Large system prompt (~${estimatedSystemTokens} tokens), requesting num_ctx: ${numCtx}`)
      }

      // Determine thinking mode: use client preference if provided, otherwise auto-detect
      let think: boolean | 'medium' = false
      if (reqData.enableThinking === true) {
        think = reqData.model.startsWith('gpt-oss') ? 'medium' : true
      } else if (reqData.enableThinking === undefined) {
        // Auto-detect for backwards compatibility
        const thinkingCapability = await this.ollamaService.checkModelHasThinking(reqData.model)
        think = thinkingCapability ? (reqData.model.startsWith('gpt-oss') ? 'medium' : true) : false
      }
      // enableThinking === false → think stays false

      // Separate sessionId and enableThinking from the Ollama request payload — Ollama rejects unknown fields
      const { sessionId, enableThinking, ...ollamaRequest } = reqData

      // Save user message to DB before streaming if sessionId provided
      let userContent: string | null = null
      if (sessionId) {
        const lastUserMsg = [...reqData.messages].reverse().find((m) => m.role === 'user')
        if (lastUserMsg) {
          userContent = lastUserMsg.content
          await this.chatService.addMessage(sessionId, 'user', userContent)
        }
      }

      if (reqData.stream) {
        logger.debug(`[OllamaController] Initiating streaming response for model: "${reqData.model}" with think: ${think}`)
        // Headers already flushed above
        const stream = await this.ollamaService.chatStream({ ...ollamaRequest, think, numCtx })
        let fullContent = ''
        for await (const chunk of stream) {
          if (chunk.message?.content) {
            fullContent += chunk.message.content
          }
          response.response.write(`data: ${JSON.stringify(chunk)}\n\n`)
        }
        if (chatSources.length > 0) {
          response.response.write(`data: ${JSON.stringify({ sources: chatSources })}\n\n`)
        }
        response.response.end()

        // Save assistant message and optionally generate title
        if (sessionId && fullContent) {
          await this.chatService.addMessage(sessionId, 'assistant', fullContent)
          const messageCount = await this.chatService.getMessageCount(sessionId)
          if (messageCount <= 2 && userContent) {
            this.chatService.generateTitle(sessionId, userContent, fullContent).catch((err) => {
              logger.error(`[OllamaController] Title generation failed: ${err instanceof Error ? err.message : err}`)
            })
          }
        }
        return
      }

      // Non-streaming (legacy) path
      const result: any = await this.ollamaService.chat({ ...ollamaRequest, think, numCtx })
      if (chatSources.length > 0) {
        result.sources = chatSources
      }

      if (sessionId && result?.message?.content) {
        await this.chatService.addMessage(sessionId, 'assistant', result.message.content)
        const messageCount = await this.chatService.getMessageCount(sessionId)
        if (messageCount <= 2 && userContent) {
          this.chatService.generateTitle(sessionId, userContent, result.message.content).catch((err) => {
            logger.error(`[OllamaController] Title generation failed: ${err instanceof Error ? err.message : err}`)
          })
        }
      }

      return result
    } catch (error: any) {
      // Auto-refresh token on 401/403 and retry once
      const status = error?.status || error?.response?.status
      if (status === 401 || status === 403) {
        const newToken = await this.ollamaService.refreshToken()
        if (newToken) {
          logger.info('[OllamaController] Token refreshed after 401, but requires restart to take effect')
        }
      }
      if (reqData.stream) {
        response.response.write(`data: ${JSON.stringify({ error: true, message: status === 401 ? 'Authentication expired. Please refresh the page.' : undefined })}\n\n`)
        response.response.end()
        return
      }
      throw error
    }
  }

  async remoteStatus() {
    const remoteUrl = await KVStore.getValue('ai.remoteOllamaUrl')
    if (!remoteUrl) {
      return { configured: false, connected: false }
    }
    try {
      let apiKey = await KVStore.getValue('ai.remoteOllamaApiKey')
      const headers: Record<string, string> = {}
      if (apiKey && typeof apiKey === 'string' && apiKey.trim()) {
        headers['Authorization'] = `Bearer ${apiKey.trim()}`
      }
      let testResponse = await fetch(`${remoteUrl.replace(/\/$/, '')}/v1/models`, {
        signal: AbortSignal.timeout(3000),
        headers,
      })

      // If unauthorized, try refreshing the token automatically
      if (testResponse.status === 401 || testResponse.status === 403) {
        const newToken = await this.ollamaService.refreshToken()
        if (newToken) {
          headers['Authorization'] = `Bearer ${newToken}`
          testResponse = await fetch(`${remoteUrl.replace(/\/$/, '')}/v1/models`, {
            signal: AbortSignal.timeout(3000),
            headers,
          })
        }
      }

      return { configured: true, connected: testResponse.ok }
    } catch {
      return { configured: true, connected: false }
    }
  }

  async configureRemote({ request, response }: HttpContext) {
    const remoteUrl: string | null = request.input('remoteUrl', null)
    const apiKey: string | null = request.input('apiKey', null)
    const refreshToken: string | null = request.input('refreshToken', null)

    const ollamaService = await Service.query().where('service_name', SERVICE_NAMES.OLLAMA).first()
    if (!ollamaService) {
      return response.status(404).send({ success: false, message: 'Ollama service record not found.' })
    }

    // Clear path: null or empty URL removes remote config and marks service as not installed
    if (!remoteUrl || remoteUrl.trim() === '') {
      await KVStore.clearValue('ai.remoteOllamaUrl')
      await KVStore.clearValue('ai.remoteOllamaApiKey')
      await KVStore.clearValue('ai.remoteOllamaRefreshToken')
      ollamaService.installed = false
      ollamaService.installation_status = 'idle'
      await ollamaService.save()
      return { success: true, message: 'Remote Ollama configuration cleared.' }
    }

    // Validate URL format
    if (!remoteUrl.startsWith('http')) {
      return response.status(400).send({
        success: false,
        message: 'Invalid URL. Must start with http:// or https://',
      })
    }

    // Test connectivity via OpenAI-compatible /v1/models endpoint (works with Ollama, LM Studio, llama.cpp, Unsloth Studio, etc.)
    try {
      const headers: Record<string, string> = {}
      if (apiKey?.trim()) {
        headers['Authorization'] = `Bearer ${apiKey.trim()}`
      }
      const testResponse = await fetch(`${remoteUrl.replace(/\/$/, '')}/v1/models`, {
        signal: AbortSignal.timeout(5000),
        headers,
      })
      if (!testResponse.ok) {
        return response.status(400).send({
          success: false,
          message: `Could not connect to ${remoteUrl} (HTTP ${testResponse.status}). Make sure the server is running and accessible. ${testResponse.status === 401 || testResponse.status === 403 ? 'Check your API key.' : 'For Ollama, start it with OLLAMA_HOST=0.0.0.0.'}`,
        })
      }
    } catch (error) {
      return response.status(400).send({
        success: false,
        message: `Could not connect to ${remoteUrl}. Make sure the server is running and reachable. For Ollama, start it with OLLAMA_HOST=0.0.0.0.`,
      })
    }

    // Save remote URL, API key, refresh token, and mark service as installed
    await KVStore.setValue('ai.remoteOllamaUrl', remoteUrl.trim())
    if (apiKey?.trim()) {
      await KVStore.setValue('ai.remoteOllamaApiKey', apiKey.trim())
    } else {
      await KVStore.clearValue('ai.remoteOllamaApiKey')
    }
    if (refreshToken?.trim()) {
      await KVStore.setValue('ai.remoteOllamaRefreshToken', refreshToken.trim())
    } else {
      await KVStore.clearValue('ai.remoteOllamaRefreshToken')
    }
    ollamaService.installed = true
    ollamaService.installation_status = 'idle'
    await ollamaService.save()

    // Install Qdrant if not already installed (fire-and-forget)
    const qdrantService = await Service.query().where('service_name', SERVICE_NAMES.QDRANT).first()
    if (qdrantService && !qdrantService.installed) {
      this.dockerService.createContainerPreflight(SERVICE_NAMES.QDRANT).catch((error) => {
        logger.error('[OllamaController] Failed to start Qdrant preflight:', error)
      })
    }

    // Mirror post-install side effects: disable suggestions, trigger docs discovery
    await KVStore.setValue('chat.suggestionsEnabled', false)
    this.ragService.discoverNomadDocs().catch((error) => {
      logger.error('[OllamaController] Failed to discover Nomad docs:', error)
    })

    return { success: true, message: 'Remote Ollama configured.' }
  }

  async deleteModel({ request }: HttpContext) {
    const reqData = await request.validateUsing(modelNameSchema)
    await this.ollamaService.deleteModel(reqData.model)
    return {
      success: true,
      message: `Model deleted: ${reqData.model}`,
    }
  }

  async dispatchModelDownload({ request }: HttpContext) {
    const reqData = await request.validateUsing(modelNameSchema)
    await this.ollamaService.dispatchModelDownload(reqData.model)
    return {
      success: true,
      message: `Download job dispatched for model: ${reqData.model}`,
    }
  }

  async installedModels({ }: HttpContext) {
    return await this.ollamaService.getModels()
  }

  /**
   * Build a deduped, user-facing source list from the RAG chunks injected into the prompt.
   * ZIM-sourced chunks get a Kiwix viewer URL reconstructed from the book name and article path.
   * Uploaded-file chunks fall back to the bare filename (no URL).
   */
  private async _buildChatSources(
    docs: Array<{ text: string; score: number; metadata?: Record<string, any> }>
  ): Promise<ChatSource[]> {
    // Dedupe by document_id (multiple chunks from the same article → one source row).
    const byKey = new Map<string, { title: string; archive?: string; url?: string; score: number }>()

    let kiwixBase: string | null = null
    const hasZim = docs.some((d) => d.metadata?.content_type === 'zim_article')
    if (hasZim) {
      kiwixBase = await this.dockerService.getServiceURL(SERVICE_NAMES.KIWIX)
    }

    for (const doc of docs) {
      const meta = doc.metadata ?? {}
      const key = meta.document_id || meta.article_path || meta.source || doc.text.slice(0, 64)
      const existing = byKey.get(key)
      if (existing && existing.score >= doc.score) continue

      if (meta.content_type === 'zim_article' && kiwixBase && meta.source && meta.article_path) {
        const bookName = path.basename(meta.source, '.zim')
        const url = `${kiwixBase.replace(/\/$/, '')}/viewer#${bookName}/${meta.article_path}`
        byKey.set(key, {
          title: meta.article_title || meta.full_title || bookName,
          archive: meta.archive_title,
          url,
          score: doc.score,
        })
      } else {
        byKey.set(key, {
          title: meta.article_title || meta.full_title || (meta.source ? path.basename(meta.source) : 'Untitled source'),
          archive: meta.archive_title,
          score: doc.score,
        })
      }
    }

    return Array.from(byKey.values()).sort((a, b) => b.score - a.score)
  }

  /**
   * Determines RAG context limits based on model size extracted from the model name.
   * Parses size indicators like "1b", "3b", "8b", "70b" from model names/tags.
   */
  private getContextLimitsForModel(modelName: string): { maxResults: number; maxTokens: number } {
    // Extract parameter count from model name (e.g., "llama3.2:3b", "qwen2.5:1.5b", "gemma:7b")
    const sizeMatch = modelName.match(/(\d+\.?\d*)[bB]/)
    const paramBillions = sizeMatch ? parseFloat(sizeMatch[1]) : 8 // default to 8B if unknown

    for (const tier of RAG_CONTEXT_LIMITS) {
      if (paramBillions <= tier.maxParams) {
        return { maxResults: tier.maxResults, maxTokens: tier.maxTokens }
      }
    }

    // Fallback: no limits
    return { maxResults: 5, maxTokens: 0 }
  }

  private async rewriteQueryWithContext(
    messages: Message[]
  ): Promise<string | null> {
    try {
      // Get recent conversation history (last 6 messages for 3 turns)
      const recentMessages = messages.slice(-6)

      // Skip rewriting for short conversations. Rewriting adds latency with
      // little RAG benefit until there is enough context to matter.
      const userMessages = recentMessages.filter(msg => msg.role === 'user')
      if (userMessages.length <= 2) {
        return userMessages[userMessages.length - 1]?.content || null
      }

      const conversationContext = recentMessages
        .map(msg => {
          const role = msg.role === 'user' ? 'User' : 'Assistant'
          // Truncate assistant messages to first 200 chars to keep context manageable
          const content = msg.role === 'assistant'
            ? msg.content.slice(0, 200) + (msg.content.length > 200 ? '...' : '')
            : msg.content
          return `${role}: "${content}"`
        })
        .join('\n')

      const installedModels = await this.ollamaService.getModels(true)
      const rewriteModelAvailable = installedModels?.some(model => model.name === DEFAULT_QUERY_REWRITE_MODEL)
      if (!rewriteModelAvailable) {
        logger.warn(`[RAG] Query rewrite model "${DEFAULT_QUERY_REWRITE_MODEL}" not available. Skipping query rewriting.`)
        const lastUserMessage = [...messages].reverse().find(msg => msg.role === 'user')
        return lastUserMessage?.content || null
      }

      // FUTURE ENHANCEMENT: allow the user to specify which model to use for rewriting
      const response = await this.ollamaService.chat({
        model: DEFAULT_QUERY_REWRITE_MODEL,
        messages: [
          {
            role: 'system',
            content: SYSTEM_PROMPTS.query_rewrite,
          },
          {
            role: 'user',
            content: `Conversation:\n${conversationContext}\n\nRewritten Query:`,
          },
        ],
      })

      const rewrittenQuery = response.message.content.trim()
      logger.info(`[RAG] Query rewritten: "${rewrittenQuery}"`)
      return rewrittenQuery
    } catch (error) {
      logger.error(
        `[RAG] Query rewriting failed: ${error instanceof Error ? error.message : error}`
      )
      // Fallback to last user message if rewriting fails
      const lastUserMessage = [...messages].reverse().find(msg => msg.role === 'user')
      return lastUserMessage?.content || null
    }
  }
}
