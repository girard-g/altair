use altair_core::ChatMessage;
use serde::{Deserialize, Serialize};

/// Ollama version response
#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub version: String,
}

/// Ollama generate request
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    #[allow(dead_code)]
    pub prompt: String,

    #[serde(default)]
    #[allow(dead_code)]
    pub stream: bool,

    #[serde(default)]
    #[allow(dead_code)]
    pub options: Option<GenerateOptions>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateOptions {
    #[allow(dead_code)]
    pub temperature: Option<f32>,
    #[allow(dead_code)]
    pub top_p: Option<f32>,
    #[allow(dead_code)]
    pub top_k: Option<u32>,
    #[allow(dead_code)]
    pub num_predict: Option<u32>,
}

/// Ollama chat request
#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default)]
    #[allow(dead_code)]
    pub stream: bool,

    #[serde(default)]
    #[allow(dead_code)]
    pub options: Option<GenerateOptions>,
}

/// Ollama pull request
#[derive(Debug, Deserialize)]
pub struct PullRequest {
    #[allow(dead_code)]
    pub name: String,

    #[serde(default)]
    #[allow(dead_code)]
    pub insecure: bool,

    #[serde(default)]
    #[allow(dead_code)]
    pub stream: bool,
}

/// Ollama show request
#[derive(Debug, Deserialize)]
pub struct ShowRequest {
    #[allow(dead_code)]
    pub name: String,
}

/// Ollama delete request
#[derive(Debug, Deserialize)]
pub struct DeleteRequest {
    #[allow(dead_code)]
    pub model: String,
}

/// Models list response
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
}

/// Running processes response
#[derive(Debug, Serialize)]
pub struct ProcessesResponse {
    pub models: Vec<RunningModel>,
}

#[derive(Debug, Serialize)]
pub struct RunningModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub expires_at: String,
}

/// Show model response
#[derive(Debug, Serialize)]
pub struct ShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
}
