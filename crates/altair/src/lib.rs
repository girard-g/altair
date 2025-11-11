//! Altair - High-Performance Local LLM Runtime
//!
//! Altair is a modular, high-performance Rust-based LLM runtime designed as a drop-in
//! replacement for Ollama. It features concurrent request handling, GPU acceleration
//! (CUDA/Metal), and both Ollama-compatible and OpenAI-compatible APIs.
//!
//! # Example
//!
//! ```ignore
//! // This example shows the full server setup
//! // Run with: cargo run --bin altair -- serve
//!
//! use altair::*;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize GPU backend
//!     let gpu_backend = altair_inference::detect_gpu_backend();
//!     altair_inference::init_gpu_backend(gpu_backend)?;
//!
//!     // Create inference backend
//!     let backend = Arc::new(altair_inference::LlamaCppBackend::new(gpu_backend)?);
//!
//!     // Create executor
//!     let executor = Arc::new(altair_inference::InferenceExecutor::new(backend, 4)?);
//!
//!     // Create API state
//!     let state = altair_api::AppState::new(executor);
//!
//!     // Create router
//!     let app = altair_api::create_router(state);
//!
//!     // Start server
//!     let listener = tokio::net::TcpListener::bind("127.0.0.1:11434").await?;
//!     axum::serve(listener, app).await?;
//!
//!     Ok(())
//! }
//! ```

// Re-export main components
pub use altair_api;
pub use altair_core;
pub use altair_inference;
pub use altair_model_registry;
