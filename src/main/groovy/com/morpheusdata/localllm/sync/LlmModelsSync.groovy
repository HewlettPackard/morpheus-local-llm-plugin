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
package com.morpheusdata.localllm.sync

import com.morpheusdata.localllm.LocalLlmApiService
import com.morpheusdata.core.MorpheusContext
import com.morpheusdata.core.data.DataQuery
import com.morpheusdata.core.util.SyncTask
import com.morpheusdata.model.llm.LlmIntegration
import com.morpheusdata.model.llm.LlmModel
import groovy.util.logging.Slf4j
import io.reactivex.rxjava3.core.Observable

/**
 * Synchronizes LLM models from a remote API into the Morpheus database.
 * Used by both OllamaProvider and OpenAICompatibleProvider.
 * Uses the Morpheus SyncTask framework: adds new models, disables removed models (soft-delete),
 * and re-enables/updates models that reappear.
 *
 * @author Morpheus
 */
@Slf4j
class LlmModelsSync {

	protected final MorpheusContext morpheusContext
	protected final LlmIntegration llmIntegration
	protected final String providerCode

	LlmModelsSync(LlmIntegration llmIntegration) {
		this(null, llmIntegration, null)
	}

	LlmModelsSync(MorpheusContext morpheusContext, LlmIntegration llmIntegration, String providerCode) {
		this.morpheusContext = morpheusContext
		this.llmIntegration = llmIntegration
		this.providerCode = providerCode
	}

	/**
	 * Execute model sync using an API response that has already been fetched.
	 * The modelBuilder closure transforms the raw API response map into a collection of LlmModel objects.
	 */
	Map execute(MorpheusContext morpheusContext, String providerCode, Map apiResult, Closure<Collection<LlmModel>> modelBuilder) {
		return new LlmModelsSync(morpheusContext, llmIntegration, providerCode).execute(apiResult, modelBuilder)
	}

	/**
	 * Execute model sync from a pre-fetched API result map.
	 */
	Map execute(Map apiResult, Closure<Collection<LlmModel>> modelBuilder) {
		if (apiResult?.success) {
			Collection<LlmModel> freshModels = []
			if (modelBuilder) {
				Map apiResponse = apiResult.data instanceof Map ? apiResult.data as Map : [:]
				def builtModels = modelBuilder.call(apiResponse)
				if (builtModels instanceof Collection) {
					freshModels = builtModels as Collection<LlmModel>
				}
			}
			execute(freshModels)
		} else {
			log.warn("Failed to refresh models: ${apiResult?.msg}")
		}
		return apiResult ?: [success: false, msg: 'No API result']
	}

	void execute(Collection<LlmModel> freshModels) {
		if (!morpheusContext || !providerCode) {
			log.warn("Skipping model sync: sync context is incomplete")
			return
		}
		if (!llmIntegration?.id) {
			log.warn("Skipping model sync: LLM integration is not persisted")
			return
		}
		Collection<LlmModel> normalizedFreshModels = freshModels ?: []
		DataQuery query = new DataQuery().withFilter('providerCode', providerCode).withFilter('llmIntegration.id', llmIntegration.id)
		SyncTask<LlmModel, LlmModel, LlmModel> syncTask = new SyncTask<>(morpheusContext.llm.model.list(query), normalizedFreshModels)
		syncTask.addMatchFunction { LlmModel existingModel, LlmModel freshModel ->
			existingModel.code == freshModel.code
		}.onDelete { List<LlmModel> removeList ->
			disableMissingModels(removeList)
		}.onAdd { List<LlmModel> addList ->
			addMissingModels(addList)
		}.withLoadObjectDetails { List<SyncTask.UpdateItemDto<LlmModel, LlmModel>> updateItems ->
			Observable.fromIterable(updateItems.collect { SyncTask.UpdateItemDto<LlmModel, LlmModel> updateItem ->
				new SyncTask.UpdateItem<LlmModel, LlmModel>(existingItem: updateItem.existingItem, masterItem: updateItem.masterItem)
			})
		}.onUpdate { List<SyncTask.UpdateItem<LlmModel, LlmModel>> updateList ->
			updateMatchedModels(updateList)
		}.start()
	}

	protected void disableMissingModels(List<LlmModel> removeList) {
		List<LlmModel> saveList = []
		removeList?.each { LlmModel model ->
			if (model?.enabled) {
				log.debug("Disabling removed model: ${model.code}")
				model.enabled = false
				saveList << model
			}
		}
		if (saveList) {
			morpheusContext.llm.model.bulkSave(saveList).blockingGet()
		}
	}

	protected void addMissingModels(List<LlmModel> addList) {
		if (!addList) {
			return
		}
		addList.each { LlmModel model ->
			model.llmIntegration = llmIntegration
			model.enabled = model.enabled != false
			log.debug("Adding new model: ${model.code}")
		}
		morpheusContext.llm.model.bulkCreate(addList).blockingGet()
	}

	protected void updateMatchedModels(List<SyncTask.UpdateItem<LlmModel, LlmModel>> updateList) {
		List<LlmModel> saveList = []
		updateList?.each { SyncTask.UpdateItem<LlmModel, LlmModel> updateItem ->
			LlmModel existingModel = updateItem.existingItem
			LlmModel freshModel = updateItem.masterItem
			boolean changed = false

			if (existingModel.llmIntegration?.id != llmIntegration.id) {
				existingModel.llmIntegration = llmIntegration
				changed = true
			}
			if (!existingModel.enabled) {
				log.info("Re-enabling model: ${freshModel.code}")
				existingModel.enabled = true
				changed = true
			}
			if (existingModel.contextWindow != freshModel.contextWindow) {
				existingModel.contextWindow = freshModel.contextWindow
				changed = true
			}
			if (existingModel.name != freshModel.name) {
				existingModel.name = freshModel.name
				changed = true
			}
			if (existingModel.modelType != freshModel.modelType) {
				existingModel.modelType = freshModel.modelType
				changed = true
			}
			if (existingModel.maxOutputTokens != freshModel.maxOutputTokens) {
				existingModel.maxOutputTokens = freshModel.maxOutputTokens
				changed = true
			}
			Map existingMetadata = existingModel.metadata ?: [:]
			Map freshMetadata = freshModel.metadata ?: [:]
			if (existingMetadata != freshMetadata) {
				existingModel.metadata = freshMetadata
				existingModel.configMap = freshMetadata
				changed = true
			}
			if (changed) {
				saveList << existingModel
			}
		}
		if (saveList) {
			morpheusContext.llm.model.bulkSave(saveList).blockingGet()
		}
	}
}
