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

import com.morpheusdata.core.util.HttpApiClient
import com.morpheusdata.model.llm.LlmChatMessage
import com.morpheusdata.model.llm.LlmChatResponse
import com.morpheusdata.response.ServiceResponse
import com.morpheusdata.response.LlmStreamingResponseHandler
import groovy.json.JsonSlurper
import groovy.util.logging.Slf4j
import org.apache.http.client.methods.CloseableHttpResponse
import org.apache.http.util.EntityUtils

import java.util.concurrent.ConcurrentHashMap

/**
 * Shared HTTP client service for Ollama and OpenAI-compatible API interactions.
 * Supports both authenticated and unauthenticated requests.
 * When no token is provided, the Authorization header is omitted entirely.
 *
 * Uses session-scoped connection pooling to support multiple concurrent
 * conversations without connection interference. Each session (identified by
 * a clientScopeKey in opts) gets its own pooled HttpApiClient that is reused
 * across turns and automatically evicted after a TTL period of inactivity.
 *
 * @author Morpheus
 */
@Slf4j
class LocalLlmApiService {

	static final String CHAT_COMPLETIONS_PATH = '/v1/chat/completions'
	static final String MODELS_PATH = '/v1/models'
	static final String OLLAMA_TAGS_PATH = '/api/tags'
	static final String OLLAMA_CHAT_PATH = '/api/chat'
	static final String CLIENT_SCOPE_KEY = 'clientScopeKey'
	static final Integer DEFAULT_CONNECTION_TIMEOUT = 30000
	static final Integer DEFAULT_READ_TIMEOUT = 30000
	static final Integer DEFAULT_INTERACT_READ_TIMEOUT = 120000
	static final Long SESSION_CLIENT_TTL_MS = 60L * 60L * 1000L

	protected final ConcurrentHashMap<String, SessionClientHolder> sessionClients = new ConcurrentHashMap<>()

	protected static class SessionClientHolder {
		final HttpApiClient apiClient
		volatile long lastUsedAt

		SessionClientHolder(HttpApiClient apiClient, long lastUsedAt) {
			this.apiClient = apiClient
			this.lastUsedAt = lastUsedAt
		}
	}

	/**
	 * List available models using the OpenAI-compatible /v1/models endpoint.
	 */
	Map listModelsOpenAI(String baseUrl, String token, Map opts = [:]) {
		return executeGet(normalizeBaseUrl(baseUrl), MODELS_PATH, token, opts)
	}

	/**
	 * List available models using Ollama's native /api/tags endpoint.
	 */
	Map listModelsOllama(String baseUrl, String token, Map opts = [:]) {
		return executeGet(normalizeBaseUrl(baseUrl), OLLAMA_TAGS_PATH, token, opts)
	}

	/**
	 * Execute a non-streaming interaction request against a model endpoint.
	 */
	Map interact(String baseUrl, String token, String endpointPath, Map requestBody, Map opts = [:]) {
		String path = endpointPath ?: CHAT_COMPLETIONS_PATH
		return executePost(normalizeBaseUrl(baseUrl), path, token, requestBody, opts)
	}

	/**
	 * Execute a streaming interaction request via SSE.
	 * Uses the OpenAI-compatible SSE streaming protocol (data: prefix lines).
	 */
	void streamInteract(String baseUrl, String token, String endpointPath, Map requestBody, LlmStreamingResponseHandler handler, Map opts = [:]) {
		CloseableHttpResponse response = null
		try {
			String normalizedUrl = normalizeBaseUrl(baseUrl)
			requestBody.stream = true
			String path = endpointPath ?: CHAT_COMPLETIONS_PATH
			HttpApiClient.RequestOptions requestOptions = buildRequestOptions(token, [
				'Content-Type': 'application/json',
				'Accept'      : 'text/event-stream'
			], requestBody, DEFAULT_INTERACT_READ_TIMEOUT)

			withApiClient(opts) { HttpApiClient apiClient ->
				ServiceResponse<CloseableHttpResponse> apiResponse = apiClient.callStreamApi(normalizedUrl, path, null, null, requestOptions, 'POST')
				response = apiResponse?.data
				if (apiResponse?.success != true || response == null) {
					String statusCode = apiResponse?.errorCode ?: response?.statusLine?.statusCode?.toString() ?: 'unknown'
					String errorBody = readErrorBody(response)
					String errorMsg = errorBody ?: buildErrorMessage(apiResponse)
					handler?.onError(new RuntimeException("API returned ${statusCode}: ${errorMsg}"))
					return
				}

				StringBuilder fullContent = new StringBuilder()
				String responseModel = null
				String finishReason = null

				response.entity?.content?.withReader('UTF-8') { reader ->
					String line
					while ((line = reader.readLine()) != null) {
						if (line.startsWith('data: ')) {
							String data = line.substring(6).trim()
							if (data == '[DONE]') {
								break
							}
							try {
								Map chunk = new JsonSlurper().parseText(data) as Map
								responseModel = responseModel ?: chunk.model?.toString()
								def choices = chunk.choices
								if (choices instanceof List && choices.size() > 0) {
									def delta = choices[0].delta
									if (delta?.content) {
										String partialText = delta.content.toString()
										fullContent.append(partialText)
										handler?.onPartialResponse(partialText)
									}
									if (choices[0].finish_reason) {
										finishReason = choices[0].finish_reason.toString()
									}
								}
							} catch (Exception parseEx) {
								log.debug("Skipping unparseable SSE chunk: ${data}")
							}
						}
					}
				}

				LlmChatResponse streamResponse = new LlmChatResponse()
				streamResponse.model = responseModel
				streamResponse.finishReason = finishReason
				LlmChatMessage msg = new LlmChatMessage()
				msg.role = 'assistant'
				msg.content = fullContent.toString()
				streamResponse.message = msg
				handler?.onCompleteResponse(streamResponse)
			}

		} catch (Exception e) {
			log.error("Error during streaming chat completion: ${e.message}", e)
			handler?.onError(e)
		} finally {
			response?.close()
		}
	}

	/**
	 * Execute a GET request. Token is optional — when null/empty, no Authorization header is sent.
	 */
	protected Map executeGet(String baseUrl, String path, String token, Map opts = [:]) {
		try {
			HttpApiClient.RequestOptions requestOptions = buildRequestOptions(token, [
				'Accept': 'application/json'
			], null, DEFAULT_READ_TIMEOUT)
			return withApiClient(opts) { HttpApiClient apiClient ->
				ServiceResponse apiResponse = apiClient.callJsonApi(baseUrl, path, null, null, requestOptions, 'GET')
				if (apiResponse?.success) {
					Map responseData = apiResponse?.data instanceof Map ? apiResponse.data as Map : [:]
					if (!(apiResponse?.data instanceof Map) && apiResponse?.data != null) {
						responseData.data = apiResponse.data
					}
					return [success: true, data: responseData, headers: apiResponse?.headers]
				} else {
					return [success: false, msg: buildErrorMessage(apiResponse), headers: apiResponse?.headers]
				}
			}
		} catch (Exception e) {
			log.error("Error executing GET ${baseUrl}${path}: ${e.message}", e)
			return [success: false, msg: e.message]
		}
	}

	/**
	 * Execute a POST request. Token is optional — when null/empty, no Authorization header is sent.
	 */
	protected Map executePost(String baseUrl, String path, String token, Map requestBody, Map opts = [:]) {
		try {
			HttpApiClient.RequestOptions requestOptions = buildRequestOptions(token, [
				'Content-Type': 'application/json',
				'Accept'      : 'application/json'
			], requestBody, DEFAULT_INTERACT_READ_TIMEOUT)
			return withApiClient(opts) { HttpApiClient apiClient ->
				ServiceResponse apiResponse = apiClient.callJsonApi(baseUrl, path, null, null, requestOptions, 'POST')
				if (apiResponse?.success) {
					Map responseData = apiResponse?.data instanceof Map ? apiResponse.data as Map : [:]
					if (!(apiResponse?.data instanceof Map) && apiResponse?.data != null) {
						responseData.data = apiResponse.data
					}
					return [success: true, data: responseData, headers: apiResponse?.headers]
				} else {
					return [success: false, msg: buildErrorMessage(apiResponse), headers: apiResponse?.headers]
				}
			}
		} catch (Exception e) {
			log.error("Error executing POST ${baseUrl}${path}: ${e.message}", e)
			return [success: false, msg: e.message]
		}
	}

	/**
	 * Execute work with an appropriate HttpApiClient. When a clientScopeKey is provided
	 * in opts, a session-scoped pooled client is used (allowing multiple concurrent
	 * conversations to each maintain their own connection pool). When no scope key is
	 * provided, a transient client is created and destroyed after use.
	 */
	protected <T> T withApiClient(Map opts = [:], Closure<T> work) {
		evictExpiredSessionClients()
		String clientScopeKey = resolveClientScopeKey(opts)
		if (clientScopeKey) {
			long now = System.currentTimeMillis()
			SessionClientHolder sessionClient = sessionClients.compute(clientScopeKey) { String key, SessionClientHolder existing ->
				if (existing && !isExpired(existing, now)) {
					existing.lastUsedAt = now
					return existing
				}
				if (existing?.apiClient) {
					shutdownApiClient(existing.apiClient)
				}
				return new SessionClientHolder(createApiClient(true), now)
			}
			return work.call(sessionClient.apiClient)
		}

		HttpApiClient apiClient = createApiClient(false)
		try {
			return work.call(apiClient)
		} finally {
			shutdownApiClient(apiClient)
		}
	}

	protected String resolveClientScopeKey(Map opts = [:]) {
		String clientScopeKey = opts?.get(CLIENT_SCOPE_KEY)?.toString()?.trim()
		return clientScopeKey ?: null
	}

	protected HttpApiClient createApiClient(boolean sessionScoped) {
		return sessionScoped ? new HttpApiClient(true) : new HttpApiClient()
	}

	protected void shutdownApiClient(HttpApiClient apiClient) {
		apiClient?.shutdownClient()
	}

	protected void evictExpiredSessionClients() {
		long now = System.currentTimeMillis()
		sessionClients.each { String clientScopeKey, SessionClientHolder holder ->
			if (holder && isExpired(holder, now) && sessionClients.remove(clientScopeKey, holder)) {
				shutdownApiClient(holder.apiClient)
			}
		}
	}

	protected boolean isExpired(SessionClientHolder holder, long now = System.currentTimeMillis()) {
		return holder == null || (now - holder.lastUsedAt) > SESSION_CLIENT_TTL_MS
	}

	/**
	 * Normalize a base URL by stripping any trailing path segments that overlap with our
	 * endpoint constants (e.g. /v1, /v1/, /api). Users often paste URLs like
	 * "http://host:port/v1" but our path constants already include /v1, which would
	 * result in double-prefixed paths like /v1/v1/models.
	 */
	protected static String normalizeBaseUrl(String baseUrl) {
		if (!baseUrl) return baseUrl
		String url = baseUrl.trim()
		// Strip trailing slash(es)
		while (url.endsWith('/')) {
			url = url.substring(0, url.length() - 1)
		}
		// Strip known path suffixes that overlap with our endpoint constants
		if (url.endsWith('/v1')) {
			url = url.substring(0, url.length() - 3)
		} else if (url.endsWith('/api')) {
			url = url.substring(0, url.length() - 4)
		}
		// Strip trailing slash(es) again after suffix removal
		while (url.endsWith('/')) {
			url = url.substring(0, url.length() - 1)
		}
		return url
	}

	/**
	 * Build request options. When token is null or empty, apiToken is not set
	 * so no Authorization header is sent (supporting unauthenticated local LLM servers).
	 */
	protected HttpApiClient.RequestOptions buildRequestOptions(String token, Map<CharSequence, CharSequence> headers, Object body, Integer readTimeout) {
		HttpApiClient.RequestOptions options = new HttpApiClient.RequestOptions(
			headers          : headers,
			ignoreSSL        : true,
			connectionTimeout: DEFAULT_CONNECTION_TIMEOUT,
			readTimeout      : readTimeout
		)
		if (token) {
			options.apiToken = token
		}
		if (body != null) {
			options.body = body
		}
		return options
	}

	protected String buildErrorMessage(ServiceResponse response) {
		if (response == null) {
			return 'Unknown error'
		}
		Map responseData = response.data instanceof Map ? response.data as Map : [:]
		String errorMessage = responseData.error?.toString() ?: responseData.message?.toString() ?: responseData.msg?.toString() ?: response.content ?: response.error ?: response.msg
		String errorCode = response.errorCode ?: response.statusCode
		if (errorCode && errorMessage) {
			return "API returned ${errorCode}: ${errorMessage}"
		}
		if (errorCode) {
			return "API returned ${errorCode}"
		}
		return errorMessage ?: 'Unknown error'
	}

	protected String readErrorBody(CloseableHttpResponse response) {
		try {
			if (response?.entity) {
				return EntityUtils.toString(response.entity)
			}
		} catch (Exception ignored) {
		}
		return null
	}
}
