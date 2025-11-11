use altair_inference::InferenceExecutor;
use std::sync::Arc;

/// Application state shared across all handlers
#[derive(Clone)]
pub struct AppState {
    pub executor: Arc<InferenceExecutor>,
}

impl AppState {
    pub fn new(executor: Arc<InferenceExecutor>) -> Self {
        Self { executor }
    }
}
