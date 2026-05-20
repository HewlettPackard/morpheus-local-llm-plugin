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

import com.morpheusdata.core.MorpheusAccountIntegrationService
import com.morpheusdata.core.MorpheusContext
import com.morpheusdata.core.Plugin
import com.morpheusdata.core.data.DataQuery
import com.morpheusdata.core.integration.MorpheusLlmIntegrationService
import com.morpheusdata.core.integration.MorpheusLlmModelService
import com.morpheusdata.core.integration.MorpheusLlmService
import com.morpheusdata.localllm.sync.LlmModelsSync
import com.morpheusdata.model.AccountIntegration
import com.morpheusdata.model.llm.*
import com.morpheusdata.response.LlmStreamingResponseHandler
import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.core.Single
import spock.lang.Specification

class OpenAICompatibleProviderSpec extends Specification {

	OpenAICompatibleProvider provider
	Plugin plugin
	MorpheusContext morpheusContext
	MorpheusLlmService llmService
	MorpheusLlmModelService modelService
	MorpheusLlmIntegrationService integrationService
	MorpheusAccountIntegrationService accountIntegrationService
	LocalLlmApiService apiService

	def setup() {
		plugin = Mock(Plugin)
		morpheusContext = Mock(MorpheusContext)
		llmService = Mock(MorpheusLlmService)
		modelService = Mock(MorpheusLlmModelService)
		integrationService = Mock(MorpheusLlmIntegrationService)
		accountIntegrationService = Mock(MorpheusAccountIntegrationService)

		morpheusContext.getLlm() >> llmService
		morpheusContext.getAccountIntegration() >> accountIntegrationService
		llmService.getModel() >> modelService
		llmService.getIntegration() >> integrationService
		modelService.list(_) >> Observable.empty()
		modelService.bulkCreate(_) >> Single.just(true)
		modelService.bulkSave(_) >> Single.just(true)
		accountIntegrationService.save(_) >> Single.just(new AccountIntegration())

		provider = new OpenAICompatibleProvider(plugin, morpheusContext)
		apiService = Mock(LocalLlmApiService)
		provider.apiService = apiService
	}

	// --- Provider metadata ---

	def "provider returns correct code"() {
		expect:
		provider.getCode() == 'openai-compatible'
	}

	def "provider returns correct name"() {
		expect:
		provider.getName() == 'OpenAI Compatible'
	}

	def "provider returns icon with correct paths"() {
		expect:
		provider.getIcon().path == 'openai-compatible.svg'
		provider.getIcon().darkPath == 'openai-compatible-white.svg'
	}

	def "provider capability fields are set correctly"() {
		expect:
		provider.chatSupported
		provider.streamingChatSupported
		!provider.embeddingSupported
	}

	def "provider returns option types without default URL"() {
		when:
		def opts = provider.getOptionTypes()

		then:
		opts != null
		opts.size() >= 2
		def urlOpt = opts.find { it.fieldName == 'serviceUrl' }
		urlOpt != null
		urlOpt.required
		urlOpt.defaultValue == null
		def credOpt = opts.find { it.fieldContext == 'credential' }
		credOpt.name == 'Credentials'
		!credOpt.required
	}

	// --- Validate ---

	def "validate succeeds when API returns models"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:8080',
			serviceToken: 'test-key'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		1 * apiService.listModelsOpenAI('http://localhost:8080', 'test-key') >> [success: true, data: [data: []]]
		result.success
	}

	def "validate succeeds without credentials (unauthenticated)"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:8080'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		1 * apiService.listModelsOpenAI('http://localhost:8080', null) >> [success: true, data: [data: []]]
		result.success
	}

	def "validate fails when URL is missing"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration()
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		!result.success
		result.errors?.error?.contains('API Endpoint URL is required')
	}

	def "validate fails when API call fails"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://bad-host:9999'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		1 * apiService.listModelsOpenAI('http://bad-host:9999', null) >> [success: false, msg: 'Connection refused']
		!result.success
	}

	// --- Refresh ---

	def "refresh syncs models from OpenAI-compatible endpoint"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			id: 20L,
			serviceUrl: 'http://localhost:8080',
			serviceToken: 'test-key'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		provider.refresh(integration)

		then:
		1 * apiService.listModelsOpenAI('http://localhost:8080', 'test-key') >> [
			success: true,
			data   : [data: [
				[id: 'llama-3-8b', owned_by: 'vllm'],
				[id: 'mistral-7b-instruct', owned_by: 'vllm']
			]]
		]
		1 * modelService.list(_) >> Observable.empty()
		1 * modelService.bulkCreate({ List models ->
			models.size() == 2 && models*.code.containsAll(['llama-3-8b', 'mistral-7b-instruct'])
		}) >> Single.just(true)
	}

	def "refresh does nothing when no URL is configured"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 20L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		provider.refresh(integration)

		then:
		0 * apiService.listModelsOpenAI(_, _)
		0 * modelService._
	}

	// --- Chat completion ---

	def "chatCompletion succeeds with valid response"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:8080',
			serviceToken: 'test-key'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'llama-3-8b'
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Hello'
		request.messages = [msg]

		when:
		def result = provider.generateResponse(integration, request, [:])

		then:
		1 * apiService.interact('http://localhost:8080', 'test-key', '/v1/chat/completions', _, _) >> [
			success: true,
			data   : [
				id     : 'chatcmpl-vllm-1',
				model  : 'llama-3-8b',
				choices: [
					[
						message      : [role: 'assistant', content: 'Hello! How can I help?'],
						finish_reason: 'stop'
					]
				],
				usage  : [prompt_tokens: 8, completion_tokens: 12, total_tokens: 20]
			]
		]
		result.success
		result.data.message.content == 'Hello! How can I help?'
		result.data.tokenUsage.totalTokens == 20
	}

	def "chatCompletion fails when URL is missing"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration()
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'test-model'
		request.messages = []

		when:
		def result = provider.generateResponse(integration, request, [:])

		then:
		!result.success
		result.errors?.error?.contains('API Endpoint URL is required')
	}

	def "chatCompletion handles API failure"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:8080'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'test-model'
		request.messages = []

		when:
		def result = provider.generateResponse(integration, request, [:])

		then:
		1 * apiService.interact(_, _, '/v1/chat/completions', _, _) >> [success: false, msg: 'Internal server error']
		!result.success
		result.errors?.error?.contains('Internal server error')
	}

	def "chatCompletion passes correct request body fields"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:8080'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'gpt-4o'
		request.temperature = 0.3
		request.topP = 0.8
		request.maxOutputTokens = 4096
		request.stopSequences = ['END', 'STOP']
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Test'
		request.messages = [msg]

		when:
		provider.generateResponse(integration, request, [:])

		then:
		1 * apiService.interact(_, _, '/v1/chat/completions', { Map body ->
			body.model == 'gpt-4o' &&
			body.temperature == 0.3 &&
			body.top_p == 0.8 &&
			body.max_tokens == 4096 &&
			body.stop == ['END', 'STOP'] &&
			body.stream == false &&
			body.messages.size() == 1
		}, _) >> [success: true, data: [choices: [[message: [content: 'ok']]]]]
	}

	// --- Streaming ---

	def "streamChatCompletion calls apiService with stream true"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:8080',
			serviceToken: 'test-key'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'llama-3-8b'
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Stream test'
		request.messages = [msg]
		LlmStreamingResponseHandler handler = Mock()

		when:
		provider.streamResponse(integration, request, handler, [:])

		then:
		1 * apiService.streamInteract('http://localhost:8080', 'test-key', '/v1/chat/completions', { Map body ->
			body.model == 'llama-3-8b' && body.stream == true
		}, handler, _)
	}

	def "streamChatCompletion calls onError when account integration is missing"() {
		given:
		LlmIntegration integration = new LlmIntegration()
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'test-model'
		LlmStreamingResponseHandler handler = Mock()

		when:
		provider.streamResponse(integration, request, handler, [:])

		then:
		1 * handler.onError({ it instanceof IllegalArgumentException && it.message.contains('Account integration is required') })
		0 * apiService.streamInteract(_, _, _, _, _, _)
	}

	def "streamChatCompletion calls onError when URL is missing"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration()
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'test-model'
		LlmStreamingResponseHandler handler = Mock()

		when:
		provider.streamResponse(integration, request, handler, [:])

		then:
		1 * handler.onError({ it instanceof IllegalArgumentException && it.message.contains('API Endpoint URL is required') })
		0 * apiService.streamInteract(_, _, _, _, _, _)
	}

	// --- Model building ---

	def "buildModelsFromApiResponse parses models correctly"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 7L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		Map apiResponse = [
			data: [
				[id: 'llama-3-8b', owned_by: 'vllm'],
				[id: 'mistral-7b-instruct', owned_by: 'lm-studio'],
				[id: 'gpt-4o-mini']
			]
		]

		when:
		def models = provider.buildModelsFromApiResponse(integration, apiResponse)

		then:
		models.size() == 3
		models*.code.contains('llama-3-8b')
		models*.code.contains('mistral-7b-instruct')
		models*.code.contains('gpt-4o-mini')
		models.every { it.llmIntegration == integration }
		models.every { it.providerCode == 'openai-compatible' }
		models.every { it.modelType == 'chat' }
		models.every { it.enabled }
		models.every { it.metadata.interactionEndpoint == '/v1/chat/completions' }
		// Check owned_by metadata
		def vllmModel = models.find { it.code == 'llama-3-8b' }
		vllmModel.metadata.ownedBy == 'vllm'
	}

	def "buildModelsFromApiResponse handles empty response"() {
		given:
		LlmIntegration integration = languageModelIntegration(new AccountIntegration(id: 1L))

		when:
		def models = provider.buildModelsFromApiResponse(integration, [data: []])

		then:
		models.isEmpty()
	}

	def "buildModelsFromApiResponse handles null data"() {
		given:
		LlmIntegration integration = languageModelIntegration(new AccountIntegration(id: 1L))

		when:
		def models = provider.buildModelsFromApiResponse(integration, [:])

		then:
		models.isEmpty()
	}

	// --- Context window estimation ---

	def "estimateContextWindow uses server-provided context_length"() {
		expect:
		provider.estimateContextWindow('some-model', [context_length: 65536]) == 65536L
	}

	def "estimateContextWindow uses server-provided max_model_len"() {
		expect:
		provider.estimateContextWindow('some-model', [max_model_len: 32768]) == 32768L
	}

	def "estimateContextWindow falls back to name heuristics"() {
		expect:
		provider.estimateContextWindow('gpt-4o', [:]) == 128000L
		provider.estimateContextWindow('gpt-4-turbo', [:]) == 128000L
		provider.estimateContextWindow('gpt-4', [:]) == 8192L
		provider.estimateContextWindow('gpt-3.5-turbo', [:]) == 16385L
		provider.estimateContextWindow('claude-3-sonnet', [:]) == 200000L
		provider.estimateContextWindow('gemini-pro', [:]) == 128000L
		provider.estimateContextWindow('mistral-7b', [:]) == 32768L
		provider.estimateContextWindow('mixtral-8x7b', [:]) == 32768L
		provider.estimateContextWindow('llama-3-8b', [:]) == 8192L
		provider.estimateContextWindow('qwen2-7b', [:]) == 32768L
		provider.estimateContextWindow('deepseek-coder-6.7b', [:]) == 16384L
	}

	def "estimateContextWindow defaults to 4096 for unknown models"() {
		expect:
		provider.estimateContextWindow('my-custom-model', [:]) == 4096L
	}

	// --- Model name formatting ---

	def "formatModelName formats correctly"() {
		expect:
		provider.formatModelName('llama-3-8b') == 'Llama 3 8b'
		provider.formatModelName('gpt-4o-mini') == 'Gpt 4o Mini'
		provider.formatModelName('mistral_7b_instruct') == 'Mistral 7b Instruct'
	}

	def "formatModelName handles null and empty"() {
		expect:
		provider.formatModelName(null) == null
		provider.formatModelName('') == ''
	}

	// --- Response parsing ---

	def "parseChatResponse handles full response"() {
		given:
		Map data = [
			id     : 'chatcmpl-vllm',
			model  : 'llama-3-8b',
			choices: [
				[
					message      : [role: 'assistant', content: 'Response text'],
					finish_reason: 'stop'
				]
			],
			usage  : [prompt_tokens: 20, completion_tokens: 30, total_tokens: 50]
		]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.id == 'chatcmpl-vllm'
		response.model == 'llama-3-8b'
		response.message.content == 'Response text'
		response.message.role == 'assistant'
		response.finishReason == 'stop'
		response.tokenUsage.inputTokens == 20
		response.tokenUsage.outputTokens == 30
		response.tokenUsage.totalTokens == 50
	}

	def "parseChatResponse handles empty choices"() {
		given:
		Map data = [id: 'test', model: 'llama-3-8b', choices: []]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.id == 'test'
		response.message == null
	}

	def "parseChatResponse handles tool calls"() {
		given:
		Map data = [
			id     : 'chatcmpl-tc',
			model  : 'llama-3-8b',
			choices: [
				[
					message      : [
						role      : 'assistant',
						content   : null,
						tool_calls: [
							[
								id      : 'call_abc',
								type    : 'function',
								function: [name: 'search', arguments: '{"query":"test"}']
							]
						]
					],
					finish_reason: 'tool_calls'
				]
			]
		]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.metadata.tool_calls != null
		response.metadata.tool_calls.size() == 1
		response.metadata.tool_calls[0].function.name == 'search'
	}

	def "parseChatResponse extracts tool call from content - simple name/arguments format"() {
		given:
		Map data = [
			id     : 'chatcmpl-local',
			model  : 'llama-3-8b',
			choices: [
				[
					message      : [role: 'assistant', content: '{"name":"get_weather","arguments":{"city":"NYC"}}'],
					finish_reason: 'stop'
				]
			]
		]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.metadata.tool_calls != null
		response.metadata.tool_calls.size() == 1
		response.metadata.tool_calls[0].function.name == 'get_weather'
		response.metadata.tool_calls[0].function.arguments == '{"city":"NYC"}'
		response.finishReason == 'tool_calls'
		response.message.content == ''
	}

	def "parseChatResponse extracts tool call from content - function wrapper format"() {
		given:
		Map data = [
			id     : 'chatcmpl-local2',
			model  : 'llama-3-8b',
			choices: [
				[
					message      : [role: 'assistant', content: '{"type":"function","function":{"name":"search","arguments":{"query":"test"}}}'],
					finish_reason: 'stop'
				]
			]
		]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.metadata.tool_calls != null
		response.metadata.tool_calls.size() == 1
		response.metadata.tool_calls[0].function.name == 'search'
		response.finishReason == 'tool_calls'
	}

	def "parseChatResponse extracts tool calls from content - array format"() {
		given:
		Map data = [
			id     : 'chatcmpl-arr',
			model  : 'llama-3-8b',
			choices: [
				[
					message      : [role: 'assistant', content: '[{"name":"tool_a","arguments":{"x":1}},{"name":"tool_b","arguments":{"y":2}}]'],
					finish_reason: 'stop'
				]
			]
		]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.metadata.tool_calls != null
		response.metadata.tool_calls.size() == 2
		response.metadata.tool_calls[0].function.name == 'tool_a'
		response.metadata.tool_calls[1].function.name == 'tool_b'
		response.finishReason == 'tool_calls'
	}

	def "parseChatResponse does not extract tool call from normal text content"() {
		given:
		Map data = [
			id     : 'chatcmpl-text',
			model  : 'llama-3-8b',
			choices: [
				[
					message      : [role: 'assistant', content: 'Here is a regular response about JSON formatting.'],
					finish_reason: 'stop'
				]
			]
		]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.metadata.tool_calls == null || response.metadata.tool_calls.isEmpty()
		response.message.content == 'Here is a regular response about JSON formatting.'
		response.finishReason == 'stop'
	}

	def "parseChatResponse does not extract tool call from JSON without tool call structure"() {
		given:
		Map data = [
			id     : 'chatcmpl-json',
			model  : 'llama-3-8b',
			choices: [
				[
					message      : [role: 'assistant', content: '{"key":"value","count":42}'],
					finish_reason: 'stop'
				]
			]
		]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.metadata.tool_calls == null || response.metadata.tool_calls.isEmpty()
		response.message.content == '{"key":"value","count":42}'
	}

	// --- Token resolution ---

	def "resolveToken prefers credentialData.password"() {
		given:
		AccountIntegration integration = new AccountIntegration()
		integration.credentialData = [password: 'cred-token']
		integration.serviceToken = 'service-token'

		expect:
		provider.resolveToken(integration) == 'cred-token'
	}

	def "resolveToken falls back to serviceToken"() {
		given:
		AccountIntegration integration = new AccountIntegration()
		integration.serviceToken = 'service-token'

		expect:
		provider.resolveToken(integration) == 'service-token'
	}

	def "resolveToken falls back to servicePassword"() {
		given:
		AccountIntegration integration = new AccountIntegration()
		integration.servicePassword = 'pwd-token'

		expect:
		provider.resolveToken(integration) == 'pwd-token'
	}

	def "resolveToken returns null when no credentials are set"() {
		given:
		AccountIntegration integration = new AccountIntegration()

		expect:
		provider.resolveToken(integration) == null
	}

	// --- Model sync ---

	def "llmModelsSync creates new models"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 123L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		Map apiResult = [
			success: true,
			data   : [data: [[id: 'llama-3-8b', owned_by: 'vllm']]]
		]

		when:
		new LlmModelsSync(integration).execute(morpheusContext, 'openai-compatible', apiResult) { Map apiResponse ->
			provider.buildModelsFromApiResponse(integration, apiResponse)
		}

		then:
		1 * modelService.list(_) >> Observable.empty()
		1 * modelService.bulkCreate({ List models ->
			models.size() == 1 && models[0].code == 'llama-3-8b' && models[0].llmIntegration == integration
		}) >> Single.just(true)
	}

	def "llmModelsSync disables removed models"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 123L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmModel existing = new LlmModel()
		existing.code = 'llama-3-8b'
		existing.enabled = true
		existing.providerCode = 'openai-compatible'
		Map apiResult = [success: true, data: [data: []]]

		when:
		new LlmModelsSync(integration).execute(morpheusContext, 'openai-compatible', apiResult) { Map apiResponse ->
			provider.buildModelsFromApiResponse(integration, apiResponse)
		}

		then:
		1 * modelService.list(_) >> Observable.just(existing)
		1 * modelService.bulkSave({ List models -> models.size() == 1 && !models[0].enabled }) >> Single.just(true)
	}

	def "llmModelsSync re-enables reappearing models"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 123L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmModel existing = new LlmModel()
		existing.code = 'llama-3-8b'
		existing.enabled = false
		existing.name = 'Llama 3 8b'
		existing.modelType = 'chat'
		existing.contextWindow = 8192L
		existing.providerCode = 'openai-compatible'
		Map apiResult = [
			success: true,
			data   : [data: [[id: 'llama-3-8b', owned_by: 'vllm']]]
		]

		when:
		new LlmModelsSync(integration).execute(morpheusContext, 'openai-compatible', apiResult) { Map apiResponse ->
			provider.buildModelsFromApiResponse(integration, apiResponse)
		}

		then:
		1 * modelService.list(_) >> Observable.just(existing)
		1 * modelService.bulkSave({ List models -> models.size() == 1 && models[0].enabled }) >> Single.just(true)
	}

	// --- Request body building ---

	def "buildChatRequestBody includes all specified parameters"() {
		given:
		LlmChatRequest request = new LlmChatRequest()
		request.temperature = 0.5
		request.topP = 0.95
		request.maxOutputTokens = 1024
		request.stopSequences = ['STOP']
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Hello'
		request.messages = [msg]

		when:
		Map body = provider.buildChatRequestBody(request, 'llama-3-8b', false)

		then:
		body.model == 'llama-3-8b'
		body.temperature == 0.5
		body.top_p == 0.95
		body.max_tokens == 1024
		body.stop == ['STOP']
		body.stream == false
		body.messages.size() == 1
		body.messages[0].role == 'user'
		body.messages[0].content == 'Hello'
	}

	def "buildChatRequestBody omits null optional parameters"() {
		given:
		LlmChatRequest request = new LlmChatRequest()
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Test'
		request.messages = [msg]

		when:
		Map body = provider.buildChatRequestBody(request, 'llama-3-8b', true)

		then:
		body.model == 'llama-3-8b'
		body.stream == true
		!body.containsKey('temperature')
		!body.containsKey('top_p')
		!body.containsKey('max_tokens')
		!body.containsKey('stop')
	}

	def "buildChatRequestBody includes tools when provided"() {
		given:
		LlmChatRequest request = new LlmChatRequest()
		request.options = [
			tools      : [[type: 'function', function: [name: 'get_weather', parameters: [:]]]],
			tool_choice: 'auto'
		]
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'What is the weather?'
		request.messages = [msg]

		when:
		Map body = provider.buildChatRequestBody(request, 'llama-3-8b', false)

		then:
		body.tools.size() == 1
		body.tools[0].function.name == 'get_weather'
		body.tool_choice == 'auto'
	}

	private static LlmIntegration languageModelIntegration(AccountIntegration accountIntegration) {
		LlmIntegration integration = new LlmIntegration()
		integration.accountIntegration = accountIntegration
		if (accountIntegration?.id) {
			integration.id = accountIntegration.id
			integration.code = "language-model-${accountIntegration.id}"
		}
		return integration
	}
}
