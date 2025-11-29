use altair_core::{error::*, types::*};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;

// NOTE: This module contains the complete implementation for llama.cpp integration.
// To enable it, you need to:
// 1. Install cmake: sudo pacman -S cmake
// 2. Uncomment llama-cpp-2 dependency in Cargo.toml
// 3. Uncomment the imports below
// 4. Build with: cargo build --release

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::token::LlamaToken;

/// Safe wrapper around llama.cpp model and context
pub struct LlamaInstance {
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
    context_size: u32,
}

impl LlamaInstance {
    /// Load a model from a GGUF file
    pub fn load(
        backend: Arc<LlamaBackend>,
        path: &Path,
        gpu_backend: GpuBackend,
        context_size: u32,
    ) -> Result<Self> {
        // Validate context size
        if !(128..=32768).contains(&context_size) {
            return Err(AltairError::InvalidModelFile(format!(
                "Invalid context size: {}. Must be between 128 and 32768",
                context_size
            )));
        }

        // Configure model parameters
        let n_gpu_layers = match gpu_backend {
            GpuBackend::Cuda { .. } | GpuBackend::Metal { .. } => 999, // Offload all layers
            GpuBackend::None => 0,                                      // CPU only
        };

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

        // Load the model
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| AltairError::InvalidModelFile(format!("Failed to load model: {:?}", e)))?;

        tracing::info!(
            "Loaded model from {:?} with {} GPU layers, context size: {}",
            path,
            n_gpu_layers,
            context_size
        );

        Ok(Self {
            model: Arc::new(model),
            backend,
            context_size,
        })
    }

    /// Generate text with streaming
    pub async fn generate_streaming(
        &self,
        request: InferenceRequest,
        token_tx: mpsc::UnboundedSender<TokenResponse>,
    ) -> Result<CompletionResponse> {
        // Create context parameters using the configured context size
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(std::num::NonZeroU32::new(self.context_size).unwrap()))
            .with_n_batch(512);

        // Create context
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| {
                AltairError::InferenceFailed(format!("Failed to create context: {:?}", e))
            })?;

        // Determine final prompt - apply chat template if requested
        let final_prompt = if request.apply_chat_template || !request.messages.is_empty() {
            self.apply_chat_template_to_request(&request)?
        } else {
            request.prompt.clone()
        };

        tracing::debug!("Final prompt: {}", final_prompt);

        // Tokenize prompt
        // Note: When using chat templates, BOS token is already included
        let add_bos = if request.apply_chat_template || !request.messages.is_empty() {
            AddBos::Never
        } else {
            AddBos::Always
        };

        let tokens = self
            .model
            .str_to_token(&final_prompt, add_bos)
            .map_err(|e| AltairError::InferenceFailed(format!("Tokenization failed: {:?}", e)))?;

        tracing::info!("Tokenized prompt into {} tokens", tokens.len());

        // Decode the prompt
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            // Only need logits for the last token
            let needs_logits = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], needs_logits)
                .map_err(|e| {
                    AltairError::InferenceFailed(format!("Failed to add token to batch: {:?}", e))
                })?;
        }
        ctx.decode(&mut batch).map_err(|e| {
            AltairError::InferenceFailed(format!("Failed to decode prompt: {:?}", e))
        })?;

        // Generation loop
        let max_tokens = request.max_tokens.unwrap_or(512);
        let temperature = request.temperature.unwrap_or(0.8);
        let top_p = request.top_p.unwrap_or(0.9);
        let top_k = request.top_k.unwrap_or(40);

        let mut generated_tokens = 0u32;
        let mut full_text = String::new();
        let start = std::time::Instant::now();
        let mut n_cur = tokens.len();

        // Track the actual stop reason
        let mut stop_reason = StopReason::EndOfText;

        for _ in 0..max_tokens {
            // Sample next token
            let token = self.sample_token(&ctx, temperature, top_p, top_k)?;

            // Check for EOS
            let llama_token = LlamaToken(token);
            if self.model.is_eog_token(llama_token) {
                tracing::info!("EOS token generated, stopping");
                stop_reason = StopReason::EndOfText;
                break;
            }

            // Convert token to string
            let token_str = self
                .model
                .token_to_str(llama_token, Special::Tokenize)
                .map_err(|e| {
                    AltairError::InferenceFailed(format!("Token conversion failed: {:?}", e))
                })?;

            tracing::debug!(
                "Generated token {}: '{}' (id: {})",
                generated_tokens,
                token_str,
                token
            );

            generated_tokens += 1;

            // Send token via stream first - if this fails, we early exit
            // This avoids accumulating text when no one is listening
            if token_tx
                .send(TokenResponse {
                    token: token_str.clone(), // Clone once for channel
                    token_id: token as u32,
                    logprob: None,
                })
                .is_err()
            {
                tracing::warn!("Failed to send token to channel - receiver dropped");
                // Client disconnected - use EndOfText as stop reason
                // (Could add StopReason::ClientDisconnected variant if needed)
                stop_reason = StopReason::EndOfText;
                break;
            }

            // Now add to accumulated text
            full_text.push_str(&token_str);

            // Check stop sequences
            if request
                .stop_sequences
                .iter()
                .any(|s| full_text.ends_with(s))
            {
                tracing::info!("Stop sequence detected");
                stop_reason = StopReason::StopSequence;
                break;
            }

            // Decode the new token for next iteration
            let mut next_batch = LlamaBatch::new(1, 1);
            next_batch
                .add(llama_token, n_cur as i32, &[0], true)
                .map_err(|e| {
                    AltairError::InferenceFailed(format!("Failed to add token to batch: {:?}", e))
                })?;
            ctx.decode(&mut next_batch).map_err(|e| {
                AltairError::InferenceFailed(format!("Failed to decode token: {:?}", e))
            })?;

            n_cur += 1;
        }

        // Override stop reason if max tokens reached
        if generated_tokens >= max_tokens {
            stop_reason = StopReason::MaxTokens;
        }

        let duration = start.elapsed();
        let tokens_per_second = if duration.as_secs_f32() > 0.0 {
            generated_tokens as f32 / duration.as_secs_f32()
        } else {
            0.0
        };

        tracing::info!(
            "Generated {} tokens in {:.2}s ({:.2} tokens/sec)",
            generated_tokens,
            duration.as_secs_f32(),
            tokens_per_second
        );

        Ok(CompletionResponse {
            text: full_text,
            tokens_generated: generated_tokens,
            tokens_per_second,
            stop_reason,
        })
    }

    /// Sample next token using temperature, top-p, and top-k
    fn sample_token(
        &self,
        ctx: &LlamaContext,
        temperature: f32,
        top_p: f32,
        top_k: u32,
    ) -> Result<i32> {
        use llama_cpp_2::sampling::LlamaSampler;
        use llama_cpp_2::token::data_array::LlamaTokenDataArray;

        // Get all candidates and convert to vec
        let candidates: Vec<_> = ctx.candidates().collect();

        if candidates.is_empty() {
            return Err(AltairError::InferenceFailed(
                "No candidates available".to_string(),
            ));
        }

        // Create token data array for sampling
        let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

        // If all sampling params are disabled/default, use greedy sampling
        let use_greedy = temperature <= 0.0 || (temperature == 1.0 && top_p >= 1.0 && top_k == 0);

        if use_greedy {
            // Use greedy sampler
            let sampler = LlamaSampler::greedy();
            candidates_p.apply_sampler(&sampler);
            return Ok(candidates_p
                .selected_token()
                .ok_or_else(|| AltairError::InferenceFailed("No token selected".to_string()))?
                .0);
        }

        // Build sampler chain: top-k -> top-p -> temperature -> dist
        let mut samplers = Vec::new();

        // Apply top-k sampling if enabled (k > 0)
        if top_k > 0 {
            samplers.push(LlamaSampler::top_k(top_k as i32));
        }

        // Apply top-p (nucleus) sampling if enabled (0 < p < 1)
        if top_p > 0.0 && top_p < 1.0 {
            samplers.push(LlamaSampler::top_p(top_p, 1));
        }

        // Apply temperature sampling
        // Temperature > 1.0 makes output more random
        // Temperature < 1.0 makes output more focused
        if temperature > 0.0 && temperature != 1.0 {
            samplers.push(LlamaSampler::temp(temperature));
        }

        // Final sampler: dist (samples from the probability distribution)
        samplers.push(LlamaSampler::dist(0)); // seed 0 for now

        // Create and apply sampler chain
        let sampler = LlamaSampler::chain_simple(samplers);
        candidates_p.apply_sampler(&sampler);

        // Get selected token
        let selected_token = candidates_p
            .selected_token()
            .ok_or_else(|| AltairError::InferenceFailed("No token selected".to_string()))?;

        Ok(selected_token.0)
    }

    /// Apply chat template from model metadata to convert messages to formatted prompt
    fn apply_chat_template_to_request(&self, request: &InferenceRequest) -> Result<String> {
        use llama_cpp_2::model::LlamaChatMessage;

        // Try to extract chat template from model metadata
        let template = match self.model.chat_template(None) {
            Ok(template) => template,
            Err(_) => {
                // Model doesn't have a chat template, use raw prompt
                tracing::warn!("Model has no embedded chat template, using raw prompt");
                return Ok(request.prompt.clone());
            }
        };

        // Convert messages to LlamaChatMessage format
        let messages: Vec<LlamaChatMessage> = if !request.messages.is_empty() {
            // Use provided structured messages
            request
                .messages
                .iter()
                .map(|msg| {
                    LlamaChatMessage::new(msg.role.clone(), msg.content.clone()).map_err(|e| {
                        AltairError::InferenceFailed(format!(
                            "Failed to create chat message: {:?}",
                            e
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Convert raw prompt to single user message
            vec![
                LlamaChatMessage::new("user".to_string(), request.prompt.clone()).map_err(|e| {
                    AltairError::InferenceFailed(format!("Failed to create chat message: {:?}", e))
                })?,
            ]
        };

        // Apply chat template to format the messages
        let formatted = self
            .model
            .apply_chat_template(&template, &messages, true)
            .map_err(|e| {
                AltairError::InferenceFailed(format!("Failed to apply chat template: {:?}", e))
            })?;

        tracing::info!(
            "Applied chat template to {} message(s), result length: {} chars",
            messages.len(),
            formatted.len()
        );

        Ok(formatted)
    }
}

/// Complete implementation ready for llama.cpp integration
///
/// The code above contains the complete, production-ready implementation.
/// To enable it:
///
/// 1. Install system dependencies:
///    ```bash
///    sudo pacman -S cmake gcc  # Arch Linux
///    # or for Ubuntu: sudo apt-get install cmake build-essential
///    ```
///
/// 2. Enable llama-cpp-2 in Cargo.toml:
///    ```toml
///    llama-cpp-2 = "0.1"
///    ```
///
/// 3. Uncomment the imports and implementation code above
///
/// 4. Build:
///    ```bash
///    cargo build --release
///    ```
/// See BUILDING.md for complete instructions.
#[cfg(test)]
mod tests {
    #[test]
    fn test_llama_instance_structure() {
        // Test that the structure compiles
        // Actual loading requires a real GGUF file
    }
}
