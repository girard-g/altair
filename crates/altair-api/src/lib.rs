//! HTTP API layer for Altair

pub mod error;
pub mod ollama;
pub mod state;
pub mod streaming;

use axum::Router;
use tower_http::{compression::CompressionLayer, cors::CorsLayer, trace::TraceLayer};

pub use error::ApiError;
pub use state::AppState;

/// Create the main API router with all endpoints
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Mount Ollama-compatible API
        .nest("/api", ollama::create_routes())
        // TODO: Add native v1 API routes
        // .nest("/v1", native::create_routes())
        // Middleware
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
