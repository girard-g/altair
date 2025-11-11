use altair_core::{error::*, types::*};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::backend::{InferenceBackend, LlamaCppBackend};

/// Manages concurrent inference requests with per-model queuing
pub struct InferenceExecutor {
    /// The inference backend (llama.cpp wrapper)
    backend: Arc<LlamaCppBackend>,

    /// Loaded models registry
    models: Arc<RwLock<HashMap<ModelId, ModelState>>>,

    /// Semaphore to limit concurrent inference operations
    concurrency_limit: Arc<Semaphore>,

    /// Maximum concurrent inference requests
    max_concurrent: usize,
}

struct ModelState {
    spec: ModelSpec,
    #[allow(dead_code)]
    model_id: ModelId,
    active_requests: usize,
    /// Track if model is in a failed state
    failed: bool,
}

impl InferenceExecutor {
    pub fn new(backend: Arc<LlamaCppBackend>, max_concurrent: usize) -> Result<Self> {
        Ok(Self {
            backend,
            models: Arc::new(RwLock::new(HashMap::new())),
            concurrency_limit: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        })
    }

    /// Get or load a model - idempotent model loading
    /// Returns existing model_id if already loaded, otherwise loads the model
    pub async fn get_or_load_model(&self, spec: ModelSpec) -> Result<ModelId> {
        // First check if model with this name is already loaded
        {
            let models = self.models.read();
            for (id, state) in models.iter() {
                if state.spec.name == spec.name && state.spec.path == spec.path {
                    tracing::debug!("Model {} already loaded with ID {}", spec.name, id);
                    return Ok(*id);
                }
            }
        }

        // Model not loaded, proceed with loading
        self.load_model(spec).await
    }

    /// Load a model and register it
    pub async fn load_model(&self, spec: ModelSpec) -> Result<ModelId> {
        tracing::info!("Loading model: {}", spec.name);

        // Load model through backend
        let model_id = self.backend.load_model(&spec).await?;

        // Register in state
        {
            let mut models = self.models.write();
            if models.contains_key(&model_id) {
                return Err(AltairError::ModelAlreadyLoaded(spec.name.clone()));
            }
            models.insert(
                model_id,
                ModelState {
                    spec: spec.clone(),
                    model_id,
                    active_requests: 0,
                    failed: false,
                },
            );
        }

        tracing::info!("Model loaded successfully: {} (ID: {})", spec.name, model_id);
        Ok(model_id)
    }

    /// Unload a model
    pub async fn unload_model(&self, model_id: ModelId) -> Result<()> {
        tracing::info!("Unloading model: {}", model_id);

        // Check if model has active requests
        {
            let models = self.models.read();
            if let Some(state) = models.get(&model_id) {
                if state.active_requests > 0 {
                    return Err(AltairError::InferenceFailed(format!(
                        "Cannot unload model with {} active requests",
                        state.active_requests
                    )));
                }
            }
        }

        // Unload through backend with proper error handling
        match self.backend.unload_model(model_id).await {
            Ok(()) => {
                // Successfully unloaded, remove from registry
                self.models.write().remove(&model_id);
                tracing::info!("Model unloaded: {}", model_id);
                Ok(())
            }
            Err(e) => {
                // Mark model as failed in registry so it can be cleaned up later
                let mut models = self.models.write();
                if let Some(state) = models.get_mut(&model_id) {
                    state.failed = true;
                    tracing::error!(
                        "Failed to unload model {}, marked as failed: {:?}",
                        model_id,
                        e
                    );
                }
                Err(e)
            }
        }
    }

    /// Execute inference request with streaming
    /// Returns a channel receiver for tokens and the final completion response
    pub async fn infer_streaming(
        &self,
        model_id: ModelId,
        request: InferenceRequest,
    ) -> Result<(tokio::sync::mpsc::UnboundedReceiver<TokenResponse>, tokio::task::JoinHandle<Result<CompletionResponse>>)> {
        // Verify model is loaded and not in failed state
        {
            let models = self.models.read();
            match models.get(&model_id) {
                None => return Err(AltairError::ModelNotFound(format!("{}", model_id))),
                Some(state) if state.failed => {
                    return Err(AltairError::InferenceFailed(format!(
                        "Model {} is in failed state and cannot be used",
                        model_id
                    )));
                }
                Some(_) => {} // Model is OK
            }
        }

        // Create channel for token streaming
        let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel();

        // Clone necessary data for spawned task
        let backend = self.backend.clone();
        let models = self.models.clone();

        // Acquire semaphore permit BEFORE incrementing counter to prevent race condition
        let permit = self
            .concurrency_limit
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| AltairError::InferenceFailed(e.to_string()))?;

        // NOW increment active requests after successful semaphore acquisition
        {
            let mut models = self.models.write();
            if let Some(state) = models.get_mut(&model_id) {
                state.active_requests += 1;
            }
        }

        // Spawn inference task
        let handle = tokio::spawn(async move {
            let result = backend.generate_streaming(model_id, request, token_tx).await;

            // Decrement active requests
            {
                let mut models = models.write();
                if let Some(state) = models.get_mut(&model_id) {
                    state.active_requests = state.active_requests.saturating_sub(1);
                }
            }

            // Release permit
            drop(permit);

            result
        });

        Ok((token_rx, handle))
    }

    /// Get model information
    pub fn get_model_info(&self, model_id: ModelId) -> Option<ModelSpec> {
        let models = self.models.read();
        models.get(&model_id).map(|state| state.spec.clone())
    }

    /// List all loaded models
    pub fn list_models(&self) -> Vec<(ModelId, ModelSpec)> {
        let models = self.models.read();
        models
            .iter()
            .map(|(id, state)| (*id, state.spec.clone()))
            .collect()
    }

    /// Get statistics
    pub fn get_stats(&self) -> ExecutorStats {
        let models = self.models.read();
        let total_active_requests: usize = models.values().map(|s| s.active_requests).sum();

        ExecutorStats {
            models_loaded: models.len(),
            active_requests: total_active_requests,
            available_slots: self
                .max_concurrent
                .saturating_sub(total_active_requests),
        }
    }

    /// Clean up failed models that have no active requests
    /// This is a recovery mechanism for models that failed to unload
    pub async fn cleanup_failed_models(&self) -> Result<usize> {
        let failed_models: Vec<ModelId> = {
            let models = self.models.read();
            models
                .iter()
                .filter(|(_, state)| state.failed && state.active_requests == 0)
                .map(|(id, _)| *id)
                .collect()
        };

        let count = failed_models.len();
        for model_id in failed_models {
            tracing::info!("Cleaning up failed model: {}", model_id);
            // Force remove from registry
            self.models.write().remove(&model_id);
        }

        if count > 0 {
            tracing::info!("Cleaned up {} failed models", count);
        }

        Ok(count)
    }
}

#[derive(Debug, Clone)]
pub struct ExecutorStats {
    pub models_loaded: usize,
    pub active_requests: usize,
    pub available_slots: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LlamaCppBackend;
    use crate::gpu::detect_gpu_backend;

    #[tokio::test]
    async fn test_executor_creation() {
        let gpu_backend = detect_gpu_backend();
        let backend = Arc::new(LlamaCppBackend::new(gpu_backend).unwrap());
        let executor = InferenceExecutor::new(backend, 4).unwrap();

        let stats = executor.get_stats();
        assert_eq!(stats.models_loaded, 0);
        assert_eq!(stats.active_requests, 0);
        assert_eq!(stats.available_slots, 4);
    }
}
