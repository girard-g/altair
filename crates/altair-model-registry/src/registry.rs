use altair_core::{error::*, types::*};
use std::path::PathBuf;

/// Model registry interface
pub trait ModelRegistry: Send + Sync {
    /// Resolve a model reference to full model information
    fn resolve_model(&self, model_ref: &str) -> Result<ModelInfo>;

    /// List all downloaded models
    fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Get model file path
    fn get_model_path(&self, model_ref: &str) -> Result<PathBuf>;

    /// Delete a model
    fn delete_model(&self, model_ref: &str) -> Result<()>;
}

/// Local filesystem model registry
pub struct LocalModelRegistry {
    models_dir: PathBuf,
}

impl LocalModelRegistry {
    pub fn new(models_dir: PathBuf) -> Result<Self> {
        // Create models directory if it doesn't exist
        std::fs::create_dir_all(&models_dir)?;

        Ok(Self { models_dir })
    }
}

impl ModelRegistry for LocalModelRegistry {
    fn resolve_model(&self, model_ref: &str) -> Result<ModelInfo> {
        // TODO: Implement model resolution
        Err(AltairError::ModelNotFound(model_ref.to_string()))
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();

        // Scan models directory for .gguf files
        if self.models_dir.exists() {
            for entry in std::fs::read_dir(&self.models_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let metadata = std::fs::metadata(&path)?;

                    models.push(ModelInfo {
                        name,
                        size: metadata.len(),
                        digest: "sha256:placeholder".to_string(),
                        modified_at: chrono::DateTime::<chrono::Utc>::from(
                            metadata.modified()?,
                        )
                        .to_rfc3339(),
                    });
                }
            }
        }

        Ok(models)
    }

    fn get_model_path(&self, model_ref: &str) -> Result<PathBuf> {
        // Simple implementation: look for {model_ref}.gguf
        let path = self.models_dir.join(format!("{}.gguf", model_ref));

        if path.exists() {
            Ok(path)
        } else {
            Err(AltairError::ModelNotFound(model_ref.to_string()))
        }
    }

    fn delete_model(&self, model_ref: &str) -> Result<()> {
        let path = self.get_model_path(model_ref)?;
        std::fs::remove_file(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_local_registry_creation() {
        let temp_dir = TempDir::new().unwrap();
        let registry = LocalModelRegistry::new(temp_dir.path().to_path_buf()).unwrap();

        // Should create directory if it doesn't exist
        assert!(temp_dir.path().exists());

        // List should return empty for new registry
        let models = registry.list_models().unwrap();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_model_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let registry = LocalModelRegistry::new(temp_dir.path().to_path_buf()).unwrap();

        let result = registry.get_model_path("nonexistent");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AltairError::ModelNotFound(_)));
    }

    #[test]
    fn test_get_model_path() {
        let temp_dir = TempDir::new().unwrap();
        let registry = LocalModelRegistry::new(temp_dir.path().to_path_buf()).unwrap();

        // Create a dummy model file
        let model_path = temp_dir.path().join("test-model.gguf");
        std::fs::write(&model_path, b"dummy model data").unwrap();

        // Should find the model
        let found_path = registry.get_model_path("test-model").unwrap();
        assert_eq!(found_path, model_path);
    }
}
