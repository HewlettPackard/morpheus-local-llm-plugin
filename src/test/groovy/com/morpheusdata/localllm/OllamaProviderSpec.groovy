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

class OllamaProviderSpec extends Specification {

	OllamaProvider provider
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

		provider = new OllamaProvider(plugin, morpheusContext)
		apiService = Mock(LocalLlmApiService)
		provider.apiService = apiService
	}

	// --- Provider metadata ---

	def "provider returns correct code"() {
		expect:
		provider.getCode() == 'ollama'
	}

	def "provider returns correct name"() {
		expect:
		provider.getName() == 'Ollama'
	}

	def "provider returns icon with correct paths"() {
		expect:
		provider.getIcon().path == 'ollama.svg'
		provider.getIcon().darkPath == 'ollama-white.svg'
	}

	def "provider capability fields are set correctly"() {
		expect:
		provider.chatSupported
		provider.streamingChatSupported
		!provider.embeddingSupported
	}

	def "provider returns option types with Ollama defaults"() {
		when:
		def opts = provider.getOptionTypes()

		then:
		opts != null
		opts.size() >= 2
		def urlOpt = opts.find { it.fieldName == 'serviceUrl' }
		urlOpt.defaultValue == 'http://localhost:11434'
		def credOpt = opts.find { it.fieldContext == 'credential' }
		credOpt.name == 'Credentials'
		!credOpt.required
	}

	// --- Validate ---

	def "validate succeeds when Ollama native API returns models"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		1 * apiService.listModelsOllama('http://localhost:11434', null) >> [success: true, data: [models: []]]
		result.success
	}

	def "validate falls back to OpenAI-compatible endpoint"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		1 * apiService.listModelsOllama('http://localhost:11434', null) >> [success: false, msg: 'Not found']
		1 * apiService.listModelsOpenAI('http://localhost:11434', null) >> [success: true, data: [data: []]]
		result.success
	}

	def "validate fails when both endpoints fail"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		1 * apiService.listModelsOllama('http://localhost:11434', null) >> [success: false, msg: 'Connection refused']
		1 * apiService.listModelsOpenAI('http://localhost:11434', null) >> [success: false, msg: 'Connection refused']
		!result.success
	}

	def "validate sends token when credentials are configured"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://remote-ollama:11434',
			serviceToken: 'my-api-key'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		1 * apiService.listModelsOllama('http://remote-ollama:11434', 'my-api-key') >> [success: true, data: [models: []]]
		result.success
	}

	def "validate uses default URL when serviceUrl is blank"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration()
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		def result = provider.validate(integration, [:])

		then:
		1 * apiService.listModelsOllama('http://localhost:11434', null) >> [success: true, data: [models: []]]
		result.success
	}

	// --- Refresh ---

	def "refresh syncs models from Ollama native endpoint"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			id: 10L,
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		provider.refresh(integration)

		then:
		1 * apiService.listModelsOllama('http://localhost:11434', null) >> [
			success: true,
			data   : [models: [
				[name: 'llama3:latest', details: [family: 'llama3', parameter_size: '8B']],
				[name: 'mistral:7b', details: [family: 'mistral']]
			]]
		]
		1 * modelService.list(_) >> Observable.empty()
		1 * modelService.bulkCreate({ List models ->
			models.size() == 2 && models*.code.containsAll(['llama3:latest', 'mistral:7b'])
		}) >> Single.just(true)
	}

	def "refresh falls back to OpenAI-compatible when Ollama native fails"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			id: 10L,
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)

		when:
		provider.refresh(integration)

		then:
		1 * apiService.listModelsOllama('http://localhost:11434', null) >> [success: false, msg: 'Not found']
		1 * apiService.listModelsOpenAI('http://localhost:11434', null) >> [
			success: true,
			data   : [data: [
				[id: 'llama3:latest']
			]]
		]
		1 * modelService.list(_) >> Observable.empty()
		1 * modelService.bulkCreate({ List models ->
			models.size() == 1 && models[0].code == 'llama3:latest'
		}) >> Single.just(true)
	}

	// --- Chat completion ---

	def "chatCompletion succeeds with valid response"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'llama3:latest'
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Hello'
		request.messages = [msg]

		when:
		def result = provider.generateResponse(integration, request, [:])

		then:
		1 * apiService.interact('http://localhost:11434', null, '/v1/chat/completions', _, _) >> [
			success: true,
			data   : [
				id     : 'chatcmpl-abc',
				model  : 'llama3:latest',
				choices: [
					[
						message      : [role: 'assistant', content: 'Hi there!'],
						finish_reason: 'stop'
					]
				],
				usage  : [prompt_tokens: 5, completion_tokens: 10, total_tokens: 15]
			]
		]
		result.success
		result.data.message.content == 'Hi there!'
		result.data.tokenUsage.totalTokens == 15
	}

	def "chatCompletion handles API failure"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'llama3:latest'
		request.messages = []

		when:
		def result = provider.generateResponse(integration, request, [:])

		then:
		1 * apiService.interact(_, _, '/v1/chat/completions', _, _) >> [success: false, msg: 'Model not loaded']
		!result.success
		result.errors?.error?.contains('Model not loaded')
	}

	def "chatCompletion passes correct request body fields"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'llama3:latest'
		request.temperature = 0.7
		request.topP = 0.9
		request.maxOutputTokens = 2048
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Test'
		request.messages = [msg]

		when:
		provider.generateResponse(integration, request, [:])

		then:
		1 * apiService.interact(_, _, '/v1/chat/completions', { Map body ->
			body.model == 'llama3:latest' &&
			body.temperature == 0.7 &&
			body.top_p == 0.9 &&
			body.max_tokens == 2048 &&
			body.stream == false &&
			body.messages.size() == 1
		}, _) >> [success: true, data: [choices: [[message: [content: 'ok']]]]]
	}

	def "chatCompletion sends no auth when token is null"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'llama3:latest'
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Test'
		request.messages = [msg]

		when:
		provider.generateResponse(integration, request, [:])

		then:
		// Token is null (no auth configured), passed as null to the API
		1 * apiService.interact('http://localhost:11434', null, '/v1/chat/completions', _, _) >> [
			success: true, data: [choices: [[message: [content: 'ok']]]]
		]
	}

	// --- Streaming ---

	def "streamChatCompletion calls apiService with stream true"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(
			serviceUrl: 'http://localhost:11434'
		)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'llama3:latest'
		LlmChatMessage msg = new LlmChatMessage()
		msg.role = 'user'
		msg.content = 'Stream test'
		request.messages = [msg]
		LlmStreamingResponseHandler handler = Mock()

		when:
		provider.streamResponse(integration, request, handler, [:])

		then:
		1 * apiService.streamInteract('http://localhost:11434', null, '/v1/chat/completions', { Map body ->
			body.model == 'llama3:latest' && body.stream == true
		}, handler, _)
	}

	def "streamChatCompletion calls onError when account integration is missing"() {
		given:
		LlmIntegration integration = new LlmIntegration()
		LlmChatRequest request = new LlmChatRequest()
		request.model = 'llama3:latest'
		LlmStreamingResponseHandler handler = Mock()

		when:
		provider.streamResponse(integration, request, handler, [:])

		then:
		1 * handler.onError({ it instanceof IllegalArgumentException && it.message.contains('Account integration is required') })
		0 * apiService.streamInteract(_, _, _, _, _, _)
	}

	// --- Model building from Ollama native response ---

	def "buildModelsFromOllamaResponse parses models correctly"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 7L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		Map apiResponse = [
			models: [
				[
					name   : 'llama3:latest',
					size   : 4661224676,
					details: [
						family            : 'llama',
						parameter_size    : '8B',
						quantization_level: 'Q4_0'
					]
				],
				[
					name   : 'mistral:7b-instruct',
					size   : 4108916384,
					details: [family: 'mistral']
				],
				[
					name   : 'codellama:13b',
					size   : 7365960704,
					details: [family: 'llama']
				]
			]
		]

		when:
		def models = provider.buildModelsFromOllamaResponse(integration, apiResponse)

		then:
		models.size() == 3
		models*.code.contains('llama3:latest')
		models*.code.contains('mistral:7b-instruct')
		models*.code.contains('codellama:13b')
		models.every { it.llmIntegration == integration }
		models.every { it.providerCode == 'ollama' }
		models.every { it.modelType == 'chat' }
		models.every { it.enabled }
		// Check metadata
		def llama3 = models.find { it.code == 'llama3:latest' }
		llama3.metadata.parameterSize == '8B'
		llama3.metadata.quantizationLevel == 'Q4_0'
		llama3.metadata.modelSize == 4661224676
	}

	def "buildModelsFromOllamaResponse handles empty response"() {
		given:
		LlmIntegration integration = languageModelIntegration(new AccountIntegration(id: 1L))

		when:
		def models = provider.buildModelsFromOllamaResponse(integration, [models: []])

		then:
		models.isEmpty()
	}

	def "buildModelsFromOllamaResponse handles null models list"() {
		given:
		LlmIntegration integration = languageModelIntegration(new AccountIntegration(id: 1L))

		when:
		def models = provider.buildModelsFromOllamaResponse(integration, [:])

		then:
		models.isEmpty()
	}

	// --- Model building from OpenAI-compatible response ---

	def "buildModelsFromOpenAIResponse parses models correctly"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 7L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		Map apiResponse = [
			data: [
				[id: 'llama3:latest'],
				[id: 'mistral:7b']
			]
		]

		when:
		def models = provider.buildModelsFromOpenAIResponse(integration, apiResponse)

		then:
		models.size() == 2
		models*.code.contains('llama3:latest')
		models*.code.contains('mistral:7b')
		models.every { it.contextWindow == 4096L }  // Default for OpenAI fallback
		models.every { it.metadata.interactionEndpoint == '/v1/chat/completions' }
	}

	// --- Context window estimation ---

	def "estimateOllamaContextWindow returns expected values for known families"() {
		expect:
		provider.estimateOllamaContextWindow([name: 'llama3:latest', details: [family: 'llama3']]) == 8192L
		provider.estimateOllamaContextWindow([name: 'llama2:7b', details: [family: 'llama']]) == 4096L
		provider.estimateOllamaContextWindow([name: 'mistral:7b', details: [family: 'mistral']]) == 32768L
		provider.estimateOllamaContextWindow([name: 'mixtral:8x7b', details: [family: 'mixtral']]) == 32768L
		provider.estimateOllamaContextWindow([name: 'codellama:13b', details: [:]]) == 16384L
		provider.estimateOllamaContextWindow([name: 'phi3:latest', details: [family: 'phi']]) == 4096L
		provider.estimateOllamaContextWindow([name: 'gemma:7b', details: [family: 'gemma']]) == 8192L
		provider.estimateOllamaContextWindow([name: 'qwen2:7b', details: [family: 'qwen']]) == 32768L
		provider.estimateOllamaContextWindow([name: 'deepseek-coder:6.7b', details: [family: 'deepseek']]) == 16384L
		provider.estimateOllamaContextWindow([name: 'command-r:latest', details: [family: 'command']]) == 131072L
	}

	def "estimateOllamaContextWindow defaults to 4096 for unknown models"() {
		expect:
		provider.estimateOllamaContextWindow([name: 'some-unknown-model:latest', details: [:]]) == 4096L
	}

	// --- Model name formatting ---

	def "formatOllamaModelName formats correctly"() {
		expect:
		provider.formatOllamaModelName('llama3:latest') == 'Llama3'
		provider.formatOllamaModelName('codellama:7b-instruct') == 'Codellama (7b-instruct)'
		provider.formatOllamaModelName('mistral:7b') == 'Mistral (7b)'
		provider.formatOllamaModelName('phi3') == 'Phi3'
	}

	def "formatOllamaModelName handles null and empty"() {
		expect:
		provider.formatOllamaModelName(null) == null
		provider.formatOllamaModelName('') == ''
	}

	// --- Response parsing ---

	def "parseChatResponse handles full response"() {
		given:
		Map data = [
			id     : 'chatcmpl-xyz',
			model  : 'llama3:latest',
			choices: [
				[
					message      : [role: 'assistant', content: 'Hello!'],
					finish_reason: 'stop'
				]
			],
			usage  : [prompt_tokens: 10, completion_tokens: 5, total_tokens: 15]
		]

		when:
		def response = provider.parseChatResponse(data)

		then:
		response.id == 'chatcmpl-xyz'
		response.model == 'llama3:latest'
		response.message.content == 'Hello!'
		response.message.role == 'assistant'
		response.finishReason == 'stop'
		response.tokenUsage.inputTokens == 10
		response.tokenUsage.outputTokens == 5
		response.tokenUsage.totalTokens == 15
	}

	def "parseChatResponse handles empty choices"() {
		given:
		Map data = [id: 'test', model: 'llama3:latest', choices: []]

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
			model  : 'llama3:latest',
			choices: [
				[
					message      : [
						role      : 'assistant',
						content   : null,
						tool_calls: [
							[
								id      : 'call_123',
								type    : 'function',
								function: [name: 'get_weather', arguments: '{"city":"NYC"}']
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
		response.metadata.tool_calls[0].function.name == 'get_weather'
	}

	def "parseChatResponse extracts tool call from content - simple name/arguments format"() {
		given:
		Map data = [
			id     : 'chatcmpl-qwen',
			model  : 'qwen2:7b',
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
			id     : 'chatcmpl-qwen2',
			model  : 'qwen2:7b',
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
			model  : 'qwen2:7b',
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
			model  : 'qwen2:7b',
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
			model  : 'qwen2:7b',
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

	def "resolveToken returns null when no credentials are set"() {
		given:
		AccountIntegration integration = new AccountIntegration()

		expect:
		provider.resolveToken(integration) == null
	}

	// --- Model sync ---

	def "llmModelsSync creates new models from Ollama response"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 123L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		Map apiResult = [
			success: true,
			data   : [models: [[name: 'llama3:latest', details: [family: 'llama3']]]]
		]

		when:
		new LlmModelsSync(integration).execute(morpheusContext, 'ollama', apiResult) { Map apiResponse ->
			provider.buildModelsFromOllamaResponse(integration, apiResponse)
		}

		then:
		1 * modelService.list(_) >> Observable.empty()
		1 * modelService.bulkCreate({ List models ->
			models.size() == 1 && models[0].code == 'llama3:latest' && models[0].llmIntegration == integration
		}) >> Single.just(true)
	}

	def "llmModelsSync disables removed models"() {
		given:
		AccountIntegration accountIntegration = new AccountIntegration(id: 123L)
		LlmIntegration integration = languageModelIntegration(accountIntegration)
		LlmModel existing = new LlmModel()
		existing.code = 'llama3:latest'
		existing.enabled = true
		existing.providerCode = 'ollama'
		Map apiResult = [success: true, data: [models: []]]

		when:
		new LlmModelsSync(integration).execute(morpheusContext, 'ollama', apiResult) { Map apiResponse ->
			provider.buildModelsFromOllamaResponse(integration, apiResponse)
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
		existing.code = 'llama3:latest'
		existing.enabled = false
		existing.name = 'Llama3'
		existing.modelType = 'chat'
		existing.contextWindow = 8192L
		existing.providerCode = 'ollama'
		Map apiResult = [
			success: true,
			data   : [models: [[name: 'llama3:latest', details: [family: 'llama3']]]]
		]

		when:
		new LlmModelsSync(integration).execute(morpheusContext, 'ollama', apiResult) { Map apiResponse ->
			provider.buildModelsFromOllamaResponse(integration, apiResponse)
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
		Map body = provider.buildChatRequestBody(request, 'llama3:latest', false)

		then:
		body.model == 'llama3:latest'
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
		Map body = provider.buildChatRequestBody(request, 'llama3:latest', true)

		then:
		body.model == 'llama3:latest'
		body.stream == true
		!body.containsKey('temperature')
		!body.containsKey('top_p')
		!body.containsKey('max_tokens')
		!body.containsKey('stop')
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
