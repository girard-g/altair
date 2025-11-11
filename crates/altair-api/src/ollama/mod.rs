mod handlers;
mod types;

use axum::{
    routing::{delete, get, post},
    Router,
};
use handlers::*;

use crate::AppState;

/// Create Ollama-compatible API routes
pub fn create_routes() -> Router<AppState> {
    Router::new()
        .route("/version", get(version))
        .route("/tags", get(list_models))
        .route("/show", post(show_model))
        .route("/generate", post(generate))
        .route("/chat", post(chat))
        .route("/pull", post(pull_model))
        .route("/delete", delete(delete_model))
        .route("/ps", get(list_running))
}
