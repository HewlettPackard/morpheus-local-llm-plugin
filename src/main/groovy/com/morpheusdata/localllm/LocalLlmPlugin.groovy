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

import com.morpheusdata.core.Plugin
import groovy.util.logging.Slf4j

/**
 * Plugin entrypoint for Local LLM Engine integrations.
 * Registers two LlmProviders:
 * <ul>
 *   <li>{@link OllamaProvider} - for connecting to Ollama instances</li>
 *   <li>{@link OpenAICompatibleProvider} - for connecting to any OpenAI-compatible API</li>
 * </ul>
 *
 * @author Morpheus
 */
@Slf4j
class LocalLlmPlugin extends Plugin {

	@Override
	String getCode() {
		return 'morpheus-local-llm-plugin'
	}

	@Override
	void initialize() {
		this.setName('Local LLM')
		OllamaProvider ollamaProvider = new OllamaProvider(this, morpheus)
		this.pluginProviders.put(ollamaProvider.code, ollamaProvider)
		OpenAICompatibleProvider openAIProvider = new OpenAICompatibleProvider(this, morpheus)
		this.pluginProviders.put(openAIProvider.code, openAIProvider)
	}

	@Override
	void onDestroy() {
		// cleanup if needed
	}
}
