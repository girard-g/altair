use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Unique identifier for a loaded model instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub uuid::Uuid);

impl ModelId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for ModelId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Model metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Model name (e.g., "llama2:7b")
    pub name: String,

    /// Path to model file
    pub path: PathBuf,

    /// Context window size
    pub context_size: u32,

    /// Number of GPU layers to offload (None = auto-detect)
    pub gpu_layers: Option<u32>,

    /// RoPE frequency base
    pub rope_freq_base: Option<f32>,

    /// RoPE frequency scale
    pub rope_freq_scale: Option<f32>,
}

/// Inference request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,

    /// Optional structured chat messages (for chat template)
    #[serde(default)]
    pub messages: Vec<ChatMessage>,

    /// Whether to apply chat template from model metadata
    #[serde(default)]
    pub apply_chat_template: bool,

    #[serde(default)]
    pub max_tokens: Option<u32>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub top_k: Option<u32>,

    #[serde(default)]
    pub stop_sequences: Vec<String>,

    #[serde(default)]
    pub stream: bool,
}

/// Chat message for chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default)]
    pub max_tokens: Option<u32>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub top_k: Option<u32>,

    #[serde(default)]
    pub stream: bool,
}

/// Token generation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub token: String,
    pub token_id: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f32>,
}

/// Final completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
    pub tokens_generated: u32,
    pub tokens_per_second: f32,
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StopReason {
    #[serde(rename = "max_tokens")]
    MaxTokens,

    #[serde(rename = "stop_sequence")]
    StopSequence,

    #[serde(rename = "end_of_text")]
    EndOfText,
}

/// CUDA device information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory_bytes: u64,
}

/// Metal device information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub recommended_max_working_set_size: u64,
}

/// GPU backend detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GpuBackend {
    #[serde(rename = "cuda")]
    Cuda {
        device_count: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        devices: Option<Vec<CudaDeviceInfo>>,
    },

    #[serde(rename = "metal")]
    Metal {
        #[serde(skip_serializing_if = "Option::is_none")]
        device_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        device_memory_gb: Option<u64>,
    },

    #[serde(rename = "cpu")]
    None,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Cuda { device_count, devices } => {
                write!(f, "CUDA ({} devices)", device_count)?;
                if let Some(devs) = devices {
                    if let Some(first) = devs.first() {
                        write!(f, " - {}", first.name)?;
                    }
                }
                Ok(())
            }
            GpuBackend::Metal { device_name, device_memory_gb } => {
                write!(f, "Metal")?;
                if let Some(name) = device_name {
                    write!(f, " - {}", name)?;
                }
                if let Some(mem_gb) = device_memory_gb {
                    write!(f, " ({}GB)", mem_gb)?;
                }
                Ok(())
            }
            GpuBackend::None => write!(f, "CPU"),
        }
    }
}

/// Model information for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub modified_at: String,
}

/// Download progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub status: DownloadStatus,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub files_completed: usize,
    pub total_files: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DownloadStatus {
    #[serde(rename = "downloading")]
    Downloading,

    #[serde(rename = "verifying")]
    Verifying,

    #[serde(rename = "completed")]
    Completed,

    #[serde(rename = "failed")]
    Failed,
}

impl DownloadProgress {
    pub fn percentage(&self) -> Option<f32> {
        self.total_bytes.map(|total| {
            if total > 0 {
                (self.bytes_downloaded as f32 / total as f32) * 100.0
            } else {
                0.0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_creation() {
        let id1 = ModelId::new();
        let id2 = ModelId::new();
        assert_ne!(id1, id2, "ModelIds should be unique");
    }

    #[test]
    fn test_download_progress_percentage() {
        let progress = DownloadProgress {
            model_id: "test".to_string(),
            status: DownloadStatus::Downloading,
            bytes_downloaded: 50,
            total_bytes: Some(100),
            files_completed: 0,
            total_files: 1,
        };
        assert_eq!(progress.percentage(), Some(50.0));

        let progress_no_total = DownloadProgress {
            model_id: "test".to_string(),
            status: DownloadStatus::Downloading,
            bytes_downloaded: 50,
            total_bytes: None,
            files_completed: 0,
            total_files: 1,
        };
        assert_eq!(progress_no_total.percentage(), None);
    }

    #[test]
    fn test_stop_reason_serialization() {
        let reason = StopReason::MaxTokens;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"max_tokens\"");
    }

    #[test]
    fn test_gpu_backend_display() {
        let cuda = GpuBackend::Cuda {
            device_count: 2,
            devices: None,
        };
        assert_eq!(cuda.to_string(), "CUDA (2 devices)");

        let cuda_with_info = GpuBackend::Cuda {
            device_count: 1,
            devices: Some(vec![CudaDeviceInfo {
                device_id: 0,
                name: "NVIDIA RTX 4090".to_string(),
                compute_capability: (8, 9),
                total_memory_bytes: 24 * 1024 * 1024 * 1024,
            }]),
        };
        assert!(cuda_with_info.to_string().contains("NVIDIA RTX 4090"));

        let metal = GpuBackend::Metal {
            device_name: None,
            device_memory_gb: None,
        };
        assert_eq!(metal.to_string(), "Metal");

        let metal_with_info = GpuBackend::Metal {
            device_name: Some("Apple M3 Max".to_string()),
            device_memory_gb: Some(64),
        };
        assert!(metal_with_info.to_string().contains("Apple M3 Max"));

        let cpu = GpuBackend::None;
        assert_eq!(cpu.to_string(), "CPU");
    }
}
