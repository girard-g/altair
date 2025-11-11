use altair_core::{error::*, types::*};
use async_trait::async_trait;
use llama_cpp_2::llama_backend::LlamaBackend;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::llama::LlamaInstance;

/// Trait for inference backends
/// This will be implemented by the llama.cpp wrapper
// #[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Load a model
    async fn load_model(&self, spec: &ModelSpec) -> Result<ModelId>;

    /// Unload a model
    async fn unload_model(&self, model_id: ModelId) -> Result<()>;

    /// Generate text with streaming via channel
    async fn generate_streaming(
        &self,
        model_id: ModelId,
        request: InferenceRequest,
        token_tx: mpsc::UnboundedSender<TokenResponse>,
    ) -> Result<CompletionResponse>;

    /// Check if model is loaded
    fn is_model_loaded(&self, model_id: ModelId) -> bool;
}

/// llama.cpp backend implementation with actual model management
pub struct LlamaCppBackend {
    gpu_backend: GpuBackend,
    backend: Arc<LlamaBackend>,
    models: Arc<RwLock<HashMap<ModelId, Arc<LlamaInstance>>>>,
}

impl LlamaCppBackend {
    pub fn new(gpu_backend: GpuBackend) -> Result<Self> {
        // Initialize llama backend once (global initialization)
        let backend = Arc::new(LlamaBackend::init().map_err(|e| {
            AltairError::InferenceFailed(format!("Failed to initialize llama backend: {:?}", e))
        })?);

        Ok(Self {
            gpu_backend,
            backend,
            models: Arc::new(RwLock::new(HashMap::new())),
        })
    }
}

// #[async_trait]
impl InferenceBackend for LlamaCppBackend {
    async fn load_model(&self, spec: &ModelSpec) -> Result<ModelId> {
        tracing::info!("Loading model: {} from {:?}", spec.name, spec.path);

        // Note: The executor layer checks for duplicate models by name/path
        // This backend layer just handles the actual model loading

        // Load model using llama.cpp with context size from spec
        let instance = LlamaInstance::load(
            self.backend.clone(),
            &spec.path,
            self.gpu_backend,
            spec.context_size,
        )?;
        let model_id = ModelId::new();

        // Store in models map
        self.models.write().insert(model_id, Arc::new(instance));

        tracing::info!(
            "Successfully loaded model {} with ID {}",
            spec.name,
            model_id
        );
        Ok(model_id)
    }

    async fn unload_model(&self, model_id: ModelId) -> Result<()> {
        tracing::info!("Unloading model: {}", model_id);

        // Remove from models map
        if self.models.write().remove(&model_id).is_some() {
            tracing::info!("Successfully unloaded model {}", model_id);
            Ok(())
        } else {
            Err(AltairError::ModelNotFound(format!(
                "Model ID: {}",
                model_id
            )))
        }
    }

    async fn generate_streaming(
        &self,
        model_id: ModelId,
        request: InferenceRequest,
        token_tx: mpsc::UnboundedSender<TokenResponse>,
    ) -> Result<CompletionResponse> {
        tracing::info!("Generating text for model: {}", model_id);

        // Get the model instance
        let instance = {
            let models = self.models.read();
            models
                .get(&model_id)
                .cloned()
                .ok_or_else(|| AltairError::ModelNotFound(format!("Model ID: {}", model_id)))?
        };

        // Run inference using llama.cpp
        instance.generate_streaming(request, token_tx).await
    }

    fn is_model_loaded(&self, model_id: ModelId) -> bool {
        self.models.read().contains_key(&model_id)
    }
}
