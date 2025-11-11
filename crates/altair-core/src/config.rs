use crate::error::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub gpu: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 11434,
            gpu: "auto".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub dir: PathBuf,
    pub keep_alive: String,
    pub max_loaded: usize,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            dir: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".altair")
                .join("models"),
            keep_alive: "15m".to_string(),
            max_loaded: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_concurrent_requests: usize,
    pub max_batch_size: usize,
    pub request_timeout_secs: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 100,
            max_batch_size: 32,
            request_timeout_secs: 300, // 5 minutes
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub require_auth: bool,
    pub bearer_token: Option<String>,
    pub rate_limit_per_minute: u32,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_auth: false,
            bearer_token: None,
            rate_limit_per_minute: 60,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn load(path: &PathBuf) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config = toml::from_str(&contents)
            .map_err(|e| AltairError::InvalidConfig(e.to_string()))?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let contents = toml::to_string_pretty(self)
            .map_err(|e| AltairError::InvalidConfig(e.to_string()))?;
        std::fs::write(path, contents)?;
        Ok(())
    }
}
