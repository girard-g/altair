use altair_core::{error::*, types::*};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio::sync::mpsc;

/// Wrapper for streaming token responses
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<TokenResponse>> + Send>>;

/// Create a streaming response channel
pub fn create_token_stream() -> (TokenSender, TokenStream) {
    let (tx, rx) = mpsc::unbounded_channel();

    let stream = async_stream::stream! {
        let mut rx = rx;
        while let Some(token) = rx.recv().await {
            yield Ok(token);
        }
    };

    (TokenSender { tx }, Box::pin(stream))
}

/// Sender for streaming tokens
pub struct TokenSender {
    tx: mpsc::UnboundedSender<TokenResponse>,
}

impl TokenSender {
    pub fn send(&self, token: TokenResponse) -> Result<()> {
        self.tx
            .send(token)
            .map_err(|_| AltairError::InferenceFailed("Stream closed".into()))
    }
}

/// Ollama-compatible streaming response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaStreamResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,

    // Metadata (when done=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
}
