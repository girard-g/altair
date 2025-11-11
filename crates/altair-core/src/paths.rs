//! Path utilities for Altair model storage

use std::path::PathBuf;

/// Get the Altair home directory (~/.altair)
pub fn get_altair_home() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".altair")
}

/// Get the models directory (~/.altair/models)
pub fn get_models_dir() -> PathBuf {
    get_altair_home().join("models")
}

/// Get the full path to a specific model file
pub fn get_model_path(model_name: &str) -> PathBuf {
    get_models_dir().join(model_name)
}

/// Get the config directory (~/.altair/config)
pub fn get_config_dir() -> PathBuf {
    get_altair_home().join("config")
}

/// Get the logs directory (~/.altair/logs)
pub fn get_logs_dir() -> PathBuf {
    get_altair_home().join("logs")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_construction() {
        let models_dir = get_models_dir();
        assert!(models_dir.ends_with(".altair/models"));

        let model_path = get_model_path("test.gguf");
        assert!(model_path.ends_with(".altair/models/test.gguf"));
    }
}
