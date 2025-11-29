use thiserror::Error;

/// Main error type for Altair
#[derive(Error, Debug)]
pub enum AltairError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model already loaded: {0}")]
    ModelAlreadyLoaded(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid model file: {0}")]
    InvalidModelFile(String),

    #[error("GPU initialization failed: {0}")]
    GpuInitFailed(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    #[error("Request timeout")]
    RequestTimeout,

    #[error("Download failed: {0}")]
    DownloadFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("File system error: {0}")]
    FileSystemError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("FFI error: {0}")]
    Ffi(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Authentication failed")]
    AuthFailed,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("GPU detection failed: {0}")]
    GpuDetectionFailed(String),

    #[error("CUDA not available: {0}")]
    CudaNotAvailable(String),

    #[error("Metal not available: {0}")]
    MetalNotAvailable(String),

    #[error("GPU validation failed: {0}")]
    GpuValidationFailed(String),
}

pub type Result<T> = std::result::Result<T, AltairError>;
