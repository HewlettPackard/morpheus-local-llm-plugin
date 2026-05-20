/*
 * Copyright 2026 Morpheus Data, LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.morpheusdata.localllm

import com.morpheusdata.core.MorpheusContext
import com.morpheusdata.core.Plugin
import com.morpheusdata.core.data.DataQuery
import com.morpheusdata.localllm.sync.LlmModelsSync
import com.morpheusdata.core.providers.LlmProvider
import com.morpheusdata.model.AccountIntegration
import com.morpheusdata.model.Icon
import com.morpheusdata.model.OptionType
import com.morpheusdata.model.llm.*
import com.morpheusdata.response.LlmStreamingResponseHandler
import com.morpheusdata.response.ServiceResponse
import groovy.util.logging.Slf4j

/**
 * LlmProvider implementation for Ollama.
 * Ollama is a local LLM runtime that serves models via a REST API.
 * This provider supports:
 * <ul>
 *   <li>Model discovery via both Ollama-native (/api/tags) and OpenAI-compatible (/v1/models) endpoints</li>
 *   <li>Chat completion via OpenAI-compatible /v1/chat/completions endpoint</li>
 *   <li>SSE streaming via OpenAI-compatible /v1/chat/completions with stream=true</li>
 *   <li>Optional authentication (credentials not required for local Ollama instances)</li>
 * </ul>
 *
 * @author Morpheus
 */
@Slf4j
class OllamaProvider implements LlmProvider {

	Plugin plugin
	MorpheusContext morpheusContext
	LocalLlmApiService apiService

	static final String PROVIDER_CODE = 'ollama'
	static final String PROVIDER_NAME = 'Ollama'
	static final String DEFAULT_API_URL = 'http://localhost:11434'

	OllamaProvider(Plugin plugin, MorpheusContext morpheusContext) {
		this.plugin = plugin
		this.morpheusContext = morpheusContext
		this.apiService = new LocalLlmApiService()
	}

	@Override
	String getCode() {
		return PROVIDER_CODE
	}

	@Override
	String getName() {
		return PROVIDER_NAME
	}

	@Override
	MorpheusContext getMorpheus() {
		return this.morpheusContext
	}

	@Override
	Plugin getPlugin() {
		return this.plugin
	}

	@Override
	Icon getIcon() {
		return new Icon(path: 'ollama.svg', darkPath: 'ollama-white.svg')
	}

	@Override
	Boolean getChatSupported() {
		return true
	}

	@Override
	Boolean getStreamingChatSupported() {
		return true
	}

	@Override
	Boolean getEmbeddingSupported() {
		return false
	}

	@Override
	List<OptionType> getOptionTypes() {
		List<OptionType> optionTypes = []

		OptionType serviceUrl = new OptionType(
			code: "${PROVIDER_CODE}.serviceUrl",
			name: "Service URL",
			fieldName: "serviceUrl",
			fieldLabel: "API Endpoint",
			fieldContext: "domain",
			inputType: OptionType.InputType.TEXT,
			displayOrder: 0,
			required: true,
			defaultValue: DEFAULT_API_URL
		)
		optionTypes.add(serviceUrl)

		// Credentials are optional for Ollama (most local instances don't require auth)
		OptionType credentials = new OptionType(
			code: "${PROVIDER_CODE}.credential",
			name: "Credentials",
			inputType: OptionType.InputType.CREDENTIAL,
			fieldName: "type",
			fieldLabel: "Credentials",
			fieldContext: "credential",
			required: false,
			displayOrder: 1,
			defaultValue: "local",
			optionSource: "credentials",
			config: '{"credentialTypes":["api-key"]}'
		)
		optionTypes.add(credentials)

		return optionTypes
	}

	@Override
	ServiceResponse validate(LlmIntegration llmIntegration, Map opts) {
		try {
			AccountIntegration accountIntegration = llmIntegration?.accountIntegration
			if (!accountIntegration) {
				return ServiceResponse.error('Account integration is required')
			}
			String token = resolveToken(accountIntegration)
			String baseUrl = accountIntegration.serviceUrl ?: DEFAULT_API_URL

			// Try Ollama-native endpoint first, fall back to OpenAI-compatible
			def result = apiService.listModelsOllama(baseUrl, token)
			if (!result.success) {
				result = apiService.listModelsOpenAI(baseUrl, token)
			}
			if (result.success) {
				return ServiceResponse.success(llmIntegration)
			} else {
				return ServiceResponse.error(result.msg ?: 'Failed to connect to Ollama instance')
			}
		} catch (Exception e) {
			log.error("Error verifying Ollama integration: ${e.message}", e)
			return ServiceResponse.error("Verification failed: ${e.message}")
		}
	}

	@Override
	void refresh(LlmIntegration llmIntegration) {
		AccountIntegration accountIntegration = llmIntegration?.accountIntegration
		if (!accountIntegration) {
			log.warn("Cannot refresh Ollama integration - account integration is missing")
			return
		}
		log.debug("Refreshing Ollama integration: ${accountIntegration.name}")
		try {
			String token = resolveToken(accountIntegration)
			String baseUrl = accountIntegration.serviceUrl ?: DEFAULT_API_URL

			// Try Ollama-native endpoint first for richer model metadata
			Map apiResult = apiService.listModelsOllama(baseUrl, token)
			boolean ollamaNative = apiResult.success
			if (!ollamaNative) {
				apiResult = apiService.listModelsOpenAI(baseUrl, token)
			}

			new LlmModelsSync(llmIntegration).execute(morpheusContext, PROVIDER_CODE, apiResult) { Map apiResponse ->
				ollamaNative ? buildModelsFromOllamaResponse(llmIntegration, apiResponse) : buildModelsFromOpenAIResponse(llmIntegration, apiResponse)
			}
		} catch (Exception e) {
			log.error("Error refreshing Ollama integration: ${e.message}", e)
		}
	}

	@Override
	ServiceResponse<LlmChatResponse> generateResponse(LlmIntegration llmIntegration, LlmChatRequest request, Map opts) {
		AccountIntegration accountIntegration = llmIntegration?.accountIntegration
		if (!accountIntegration) {
			return ServiceResponse.error('Account integration is required')
		}
		String token = resolveToken(accountIntegration)
		String baseUrl = accountIntegration.serviceUrl ?: DEFAULT_API_URL
		String model = request.model
		if (!model) {
			// Try to pick the first available model
			model = loadFirstAvailableModel(llmIntegration)
		}
		if (!model) {
			return ServiceResponse.error('No model specified and no models available')
		}
		Map requestBody = buildChatRequestBody(request, model, false)

		if (log.isDebugEnabled()) {
			def toolCount = (requestBody.tools as List)?.size() ?: 0
			def toolNames = (requestBody.tools as List)?.collect { it?.function?.name ?: 'unknown' } ?: []
			def msgCount = (requestBody.messages as List)?.size() ?: 0
			log.debug("Ollama request: model=${model}, tools=${toolCount} ${toolNames}, messages=${msgCount}")
			log.debug("Ollama request body (messages last 2): ${(requestBody.messages as List)?.takeRight(2)}")
		}

		// Retry up to 2 times on connection errors (stale keep-alive sockets)
		int maxAttempts = 3
		Exception lastException = null
		for (int attempt = 1; attempt <= maxAttempts; attempt++) {
			if (attempt > 1) {
				log.warn("Ollama API retry ${attempt - 1}/${maxAttempts - 1} after connection failure")
				Thread.sleep(1500L * (attempt - 1))
			}
		Map requestOpts = opts ?: [:]
			try {
				def result = apiService.interact(baseUrl, token, LocalLlmApiService.CHAT_COMPLETIONS_PATH, requestBody, requestOpts)
				if (result.success && result.data) {
					if (log.isDebugEnabled()) {
						def rawChoice = (result.data.choices as List)?.getAt(0)
						log.debug("Ollama raw response: finish_reason=${rawChoice?.finish_reason}, has_tool_calls=${rawChoice?.message?.tool_calls != null}, content_length=${rawChoice?.message?.content?.toString()?.length() ?: 0}")
						if (rawChoice?.message?.tool_calls) {
							log.debug("Ollama raw tool_calls: ${rawChoice.message.tool_calls}")
						}
						if (rawChoice?.message?.content) {
							def contentPreview = rawChoice.message.content.toString()
							if (contentPreview.length() > 500) contentPreview = contentPreview[0..499] + '...'
							log.debug("Ollama raw content: ${contentPreview}")
						}
					}
					LlmChatResponse response = parseChatResponse(result.data)
					if (log.isDebugEnabled()) {
						log.debug("Ollama parsed response: finishReason=${response.finishReason}, has_tool_calls=${response.metadata?.tool_calls != null}, tool_call_count=${(response.metadata?.tool_calls as List)?.size() ?: 0}, content_empty=${!response.message?.content}")
					}
					return ServiceResponse.success(response)
				} else {
					log.warn("Ollama API error: ${result.msg}")
					return ServiceResponse.error(result.msg ?: 'Chat completion failed')
				}
			} catch (Exception e) {
				lastException = e
				def msg = e.message?.toLowerCase() ?: ''
				boolean isConnErr = msg.contains('failed to respond') || msg.contains('connection') ||
					msg.contains('reset') || msg.contains('broken pipe') || msg.contains('socket') ||
					msg.contains('nohttpresponse') || msg.contains('stream closed')
				if (isConnErr && attempt < maxAttempts) {
					log.warn("Ollama connection error (attempt ${attempt}): ${e.message}")
					continue
				}
				log.error("Error during Ollama chat completion: ${e.message}", e)
				return ServiceResponse.error("Chat completion failed: ${e.message}")
			}
		}
		return ServiceResponse.error("Chat completion failed after ${maxAttempts} attempts: ${lastException?.message}")
	}

	@Override
	void streamResponse(LlmIntegration llmIntegration, LlmChatRequest request, LlmStreamingResponseHandler handler, Map opts) {
		try {
			AccountIntegration accountIntegration = llmIntegration?.accountIntegration
			if (!accountIntegration) {
				handler?.onError(new IllegalArgumentException('Account integration is required'))
				return
			}
			String token = resolveToken(accountIntegration)
			String baseUrl = accountIntegration.serviceUrl ?: DEFAULT_API_URL
			String model = request.model
			if (!model) {
				model = loadFirstAvailableModel(llmIntegration)
			}
			if (!model) {
				handler?.onError(new IllegalArgumentException('No model specified and no models available'))
				return
			}
			Map requestBody = buildChatRequestBody(request, model, true)
			apiService.streamInteract(baseUrl, token, LocalLlmApiService.CHAT_COMPLETIONS_PATH, requestBody, handler, opts ?: [:])
		} catch (Exception e) {
			log.error("Error during Ollama streaming chat: ${e.message}", e)
			handler?.onError(e)
		}
	}

	// --- Helper methods ---

	/**
	 * Resolve API token from integration credentials.
	 * Returns null if no credentials are configured (Ollama typically needs no auth).
	 */
	protected String resolveToken(AccountIntegration accountIntegration) {
		return accountIntegration.credentialData?.password ?: accountIntegration.serviceToken ?: accountIntegration.servicePassword
	}

	/**
	 * Build LlmModel list from Ollama's native /api/tags response.
	 * The Ollama response format is: { "models": [ { "name": "llama3:latest", "details": { ... } }, ... ] }
	 */
	protected List<LlmModel> buildModelsFromOllamaResponse(LlmIntegration llmIntegration, Map apiResponse) {
		List<LlmModel> models = []
		def modelData = apiResponse?.models
		if (modelData instanceof List) {
			modelData.each { entry ->
				String modelName = entry.name?.toString()
				if (modelName) {
					LlmModel model = new LlmModel()
					model.code = modelName
					model.externalId = modelName
					model.name = formatOllamaModelName(modelName)
					model.providerCode = PROVIDER_CODE
					model.modelType = 'chat'
					model.contextWindow = estimateOllamaContextWindow(entry)
					Map<String, Object> metadata = buildOllamaMetadata(entry)
					if (metadata) {
						model.metadata = metadata
						model.configMap = metadata
					}
					model.llmIntegration = llmIntegration
					model.enabled = true
					models.add(model)
				}
			}
		}
		models.sort { a, b -> (a.name ?: '').compareTo(b.name ?: '') }
		return models
	}

	/**
	 * Build LlmModel list from OpenAI-compatible /v1/models response.
	 * Fallback when Ollama's native endpoint is unavailable.
	 */
	protected List<LlmModel> buildModelsFromOpenAIResponse(LlmIntegration llmIntegration, Map apiResponse) {
		List<LlmModel> models = []
		def modelData = apiResponse?.data
		if (modelData instanceof List) {
			modelData.each { entry ->
				String modelId = entry.id?.toString()
				if (modelId) {
					LlmModel model = new LlmModel()
					model.code = modelId
					model.externalId = modelId
					model.name = formatOllamaModelName(modelId)
					model.providerCode = PROVIDER_CODE
					model.modelType = 'chat'
					model.contextWindow = 4096L  // Default; Ollama's OpenAI-compat endpoint doesn't expose this
					Map<String, Object> metadata = [
						interactionEndpoint: LocalLlmApiService.CHAT_COMPLETIONS_PATH,
						streamingInteractionEndpoint: LocalLlmApiService.CHAT_COMPLETIONS_PATH
					]
					model.metadata = metadata
					model.configMap = metadata
					model.llmIntegration = llmIntegration
					model.enabled = true
					models.add(model)
				}
			}
		}
		models.sort { a, b -> (a.name ?: '').compareTo(b.name ?: '') }
		return models
	}

	/**
	 * Format an Ollama model name into a human-readable display name.
	 * "llama3:latest" -> "Llama3", "codellama:7b-instruct" -> "Codellama 7b Instruct"
	 */
	protected String formatOllamaModelName(String modelName) {
		if (!modelName) return modelName
		// Strip :latest tag if present
		String name = modelName.contains(':') ? modelName : modelName
		String baseName = name.split(':')[0]
		String tag = name.contains(':') ? name.split(':')[1] : ''
		String formatted = baseName.split(/[-_]/).collect { it.capitalize() }.join(' ')
		if (tag && tag != 'latest') {
			formatted += " (${tag})"
		}
		return formatted
	}

	/**
	 * Estimate context window from Ollama model metadata.
	 * Ollama doesn't consistently report this, so we use heuristics based on model family.
	 */
	protected Long estimateOllamaContextWindow(Map entry) {
		String modelName = entry.name?.toString()?.toLowerCase() ?: ''
		Map details = entry.details instanceof Map ? entry.details as Map : [:]
		String family = details.family?.toString()?.toLowerCase() ?: ''

		// Check for known large-context models
		if (modelName.contains('llama3') || family.contains('llama3')) {
			return 8192L
		}
		if (modelName.contains('llama2') || family.contains('llama')) {
			return 4096L
		}
		if (modelName.contains('mistral') || family.contains('mistral')) {
			return 32768L
		}
		if (modelName.contains('mixtral') || family.contains('mixtral')) {
			return 32768L
		}
		if (modelName.contains('codellama') || modelName.contains('code-llama')) {
			return 16384L
		}
		if (modelName.contains('phi') || family.contains('phi')) {
			return 4096L
		}
		if (modelName.contains('gemma') || family.contains('gemma')) {
			return 8192L
		}
		if (modelName.contains('qwen') || family.contains('qwen')) {
			return 32768L
		}
		if (modelName.contains('deepseek') || family.contains('deepseek')) {
			return 16384L
		}
		if (modelName.contains('command-r') || family.contains('command')) {
			return 131072L
		}
		return 4096L
	}

	/**
	 * Build metadata map from Ollama model details.
	 */
	protected Map<String, Object> buildOllamaMetadata(Map entry) {
		Map<String, Object> metadata = [
			interactionEndpoint: LocalLlmApiService.CHAT_COMPLETIONS_PATH,
			streamingInteractionEndpoint: LocalLlmApiService.CHAT_COMPLETIONS_PATH
		]
		Map details = entry.details instanceof Map ? entry.details as Map : [:]
		if (details.family) {
			metadata.family = details.family.toString()
		}
		if (details.parameter_size) {
			metadata.parameterSize = details.parameter_size.toString()
		}
		if (details.quantization_level) {
			metadata.quantizationLevel = details.quantization_level.toString()
		}
		if (entry.size) {
			metadata.modelSize = entry.size
		}
		return metadata
	}

	/**
	 * Load the first available model for this integration from the database.
	 */
	protected String loadFirstAvailableModel(LlmIntegration llmIntegration) {
		if (!llmIntegration?.id) {
			return null
		}
		try {
			def modelService = morpheusContext.llm.model
			DataQuery query = new DataQuery()
				.withFilter('providerCode', PROVIDER_CODE)
				.withFilter('llmIntegration.id', llmIntegration.id)
				.withFilter('enabled', true)
			def models = modelService.list(query).toList().blockingGet()
			return models ? models[0].code : null
		} catch (Exception ex) {
			log.debug("Could not load models for Ollama integration: ${ex.message}")
			return null
		}
	}

	/**
	 * Build the OpenAI-compatible request body for chat completions.
	 */
	protected Map buildChatRequestBody(LlmChatRequest request, String model, Boolean stream) {
		boolean hasTools = request.options?.tools
		def apiMessages = request.messages?.collect { msg ->
			def m = [role: msg.role ?: 'user', content: msg.content]
			if (msg.metadata?.tool_calls) {
				m.tool_calls = msg.metadata.tool_calls.collect { tc ->
					def copy = new LinkedHashMap(tc)
					if (copy.function?.arguments != null && !(copy.function.arguments instanceof String)) {
						copy.function = new LinkedHashMap(copy.function)
						copy.function.arguments = new groovy.json.JsonBuilder(copy.function.arguments).toString()
					}
					return copy
				}
			}
			if (msg.metadata?.tool_call_id) {
				m.tool_call_id = msg.metadata.tool_call_id
			}
			return m
		} ?: []

		// For Qwen 3 models with tools, disable thinking mode to ensure clean tool call output.
		// Qwen 3's thinking mode wraps responses in <think> tags which can interfere with
		// structured tool call generation via Ollama's OpenAI-compatible endpoint.
		String modelLower = model?.toLowerCase() ?: ''
		if (hasTools && (modelLower.contains('qwen') || modelLower.contains('qwen3'))) {
			// Inject /no_think directive into the last user message if not already present
			def lastUserMsg = apiMessages.reverse().find { it.role == 'user' }
			if (lastUserMsg && !lastUserMsg.content?.contains('/no_think')) {
				lastUserMsg.content = '/no_think\n' + (lastUserMsg.content ?: '')
			}
		}

		Map requestBody = [
			model   : model,
			messages: apiMessages
		]
		if (request.temperature != null) {
			requestBody.temperature = request.temperature
		}
		if (request.topP != null) {
			requestBody.top_p = request.topP
		}
		if (request.maxOutputTokens != null) {
			requestBody.max_tokens = request.maxOutputTokens
		}
		if (request.stopSequences) {
			requestBody.stop = request.stopSequences
		}
		if (request.options?.tools) {
			requestBody.tools = request.options.tools
		}
		if (request.options?.tool_choice) {
			requestBody.tool_choice = request.options.tool_choice
		}
		if (stream != null) {
			requestBody.stream = stream
		}
		return requestBody
	}

	/**
	 * Parse OpenAI-compatible chat completion response into LlmChatResponse.
	 * Includes fallback detection for models that emit tool calls as JSON text
	 * in the content field rather than using the structured tool_calls array.
	 */
	protected LlmChatResponse parseChatResponse(Map data) {
		LlmChatResponse response = new LlmChatResponse()
		response.id = data.id?.toString()
		response.model = data.model?.toString()

		def choices = data.choices
		if (choices instanceof List && choices.size() > 0) {
			def firstChoice = choices[0]
			response.finishReason = firstChoice.finish_reason?.toString()

			LlmChatMessage msg = new LlmChatMessage()
			msg.role = firstChoice.message?.role?.toString() ?: 'assistant'
			msg.content = firstChoice.message?.content?.toString()
			// Strip <think>...</think> blocks from content (Qwen 3 thinking mode)
			if (msg.content) {
				msg.content = msg.content.replaceAll(/(?s)<think>.*?<\/think>/, '').trim()
			}
			response.message = msg

			// Collect tool_calls from all choices
			List allToolCalls = []
			choices.each { choice ->
				def tc = choice?.message?.tool_calls
				if (tc instanceof List) allToolCalls.addAll(tc)
			}

			if (allToolCalls) {
				List<Map> normalizedToolCalls = allToolCalls.collect { tc ->
					Map toolCall = [
						id  : tc.id?.toString() ?: UUID.randomUUID().toString(),
						type: tc.type?.toString() ?: 'function'
					]
					if (tc.function instanceof Map) {
						toolCall.function = [
							name     : tc.function.name?.toString(),
							arguments: tc.function.arguments?.toString() ?: '{}'
						]
					}
					return toolCall
				}
				response.metadata.put('tool_calls', normalizedToolCalls)
			} else if (msg.content) {
				// Fallback: some local models (e.g. Qwen, DeepSeek) emit tool calls as
				// JSON text in the content field instead of using the structured tool_calls
				// array. Detect this pattern and convert it into proper tool call metadata
				// so the langchain4j tool-calling loop can execute them.
				List<Map> parsedToolCalls = extractToolCallsFromContent(msg.content)
				if (parsedToolCalls) {
					log.debug("Extracted ${parsedToolCalls.size()} tool call(s) from content fallback: ${parsedToolCalls.collect { it?.function?.name }}")
					response.metadata.put('tool_calls', parsedToolCalls)
					response.finishReason = 'tool_calls'
					// Clear content so langchain4j doesn't treat this as a text response
					msg.content = ''
				}
			}
		}

		def usage = data.usage
		if (usage) {
			LlmTokenUsage tokenUsage = new LlmTokenUsage()
			tokenUsage.inputTokens = usage.prompt_tokens as Integer
			tokenUsage.outputTokens = usage.completion_tokens as Integer
			tokenUsage.totalTokens = usage.total_tokens as Integer
			response.tokenUsage = tokenUsage
		}

		return response
	}

	/**
	 * Attempt to extract tool call structures from raw text content.
	 * Handles several formats that local models commonly emit:
	 * <ul>
	 *   <li>Single object: {@code {"name":"tool","arguments":{...}}}</li>
	 *   <li>Array of objects: {@code [{"name":"tool","arguments":{...}}]}</li>
	 *   <li>Wrapped: {@code {"tool_calls":[...]}}</li>
	 *   <li>With type/function nesting: {@code {"type":"function","function":{"name":"tool","arguments":{...}}}}</li>
	 * </ul>
	 *
	 * @param content the raw assistant message content
	 * @return normalized tool call list, or empty list if content is not a tool call
	 */
	protected List<Map> extractToolCallsFromContent(String content) {
		if (!content) return []
		String trimmed = content.trim()

		// Strip <think>...</think> blocks (Qwen 3 "thinking" mode)
		trimmed = trimmed.replaceAll(/(?s)<think>.*?<\/think>/, '').trim()

		// Extract content from <tool_call>...</tool_call> tags (Qwen 3 format)
		def toolCallTagPattern = /(?s)<tool_call>\s*(.*?)\s*<\/tool_call>/
		def tagMatcher = (trimmed =~ toolCallTagPattern)
		if (tagMatcher) {
			List<Map> toolCalls = []
			tagMatcher.each { match ->
				String jsonStr = (match as List)[1]?.toString()?.trim()
				if (jsonStr) {
					try {
						def parsed = new groovy.json.JsonSlurper().parseText(jsonStr)
						if (parsed instanceof Map) {
							Map normalized = normalizeToolCallCandidate(parsed as Map)
							if (normalized) toolCalls << normalized
						}
					} catch (Exception ignored) {}
				}
			}
			if (toolCalls) return toolCalls
		}

		// Quick guard: remaining content must look like JSON
		if (!(trimmed.startsWith('{') || trimmed.startsWith('['))) return []
		try {
			def parsed = new groovy.json.JsonSlurper().parseText(trimmed)
			List<Map> candidates = []
			if (parsed instanceof List) {
				candidates = parsed.findAll { it instanceof Map } as List<Map>
			} else if (parsed instanceof Map) {
				Map m = parsed as Map
				// Unwrap {"tool_calls": [...]} wrapper
				if (m.tool_calls instanceof List) {
					candidates = (m.tool_calls as List).findAll { it instanceof Map } as List<Map>
				} else {
					candidates = [m]
				}
			}
			List<Map> toolCalls = []
			candidates.each { Map candidate ->
				Map normalized = normalizeToolCallCandidate(candidate)
				if (normalized) {
					toolCalls << normalized
				}
			}
			return toolCalls
		} catch (Exception ignored) {
			// Content is not valid JSON — not a tool call
			return []
		}
	}

	/**
	 * Normalize a single parsed JSON object into the standard tool call structure
	 * {@code [id, type, function: [name, arguments]]} if it looks like a tool call.
	 * Returns null if it does not match any known tool call pattern.
	 */
	protected Map normalizeToolCallCandidate(Map candidate) {
		String name = null
		def arguments = null

		// Pattern 1: {"type":"function","function":{"name":"...","arguments":{...}}}
		if (candidate.function instanceof Map) {
			Map fn = candidate.function as Map
			name = fn.name?.toString()
			arguments = fn.arguments
		}
		// Pattern 2: {"name":"...","arguments":{...}} or {"name":"...","parameters":{...}}
		else if (candidate.name) {
			name = candidate.name.toString()
			arguments = candidate.arguments ?: candidate.parameters
		}

		if (!name) return null

		String argsStr
		if (arguments instanceof Map || arguments instanceof List) {
			argsStr = new groovy.json.JsonBuilder(arguments).toString()
		} else if (arguments != null) {
			argsStr = arguments.toString()
		} else {
			argsStr = '{}'
		}

		return [
			id      : candidate.id?.toString() ?: UUID.randomUUID().toString(),
			type    : 'function',
			function: [
				name     : name,
				arguments: argsStr
			]
		]
	}
}
