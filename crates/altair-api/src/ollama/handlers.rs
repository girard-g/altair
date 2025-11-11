use super::types::*;
use crate::ollama::types::ModelEntry;
use crate::{error::ApiResult, streaming::OllamaStreamResponse, AppState};
use altair_core::paths::{get_models_dir, get_model_path};
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, Sse},
    Json,
};
use futures::Stream;
use std::convert::Infallible;

/// GET /api/version
pub async fn version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// GET /api/tags - List all models
pub async fn list_models(State(_state): State<AppState>) -> ApiResult<Json<ModelsResponse>> {
    use tokio::fs;

    // Get models directory using helper
    let models_dir = get_models_dir();

    let mut models = Vec::new();

    // Scan directory for .gguf files (async)
    if let Ok(mut entries) = fs::read_dir(&models_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();

            // Only include .gguf files
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                if let Ok(metadata) = fs::metadata(&path).await {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let modified_at = metadata.modified()
                        .ok()
                        .and_then(|time| {
                            use std::time::SystemTime;
                            time.duration_since(SystemTime::UNIX_EPOCH).ok()
                        })
                        .map(|duration| {
                            chrono::DateTime::from_timestamp(duration.as_secs() as i64, 0)
                                .unwrap_or_default()
                                .to_rfc3339()
                        })
                        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

                    let size = metadata.len();

                    // Generate a simple digest from the filename
                    let digest = format!("sha256:{}", name.replace(".gguf", ""));

                    models.push(ModelEntry {
                        name,
                        modified_at,
                        size,
                        digest,
                    });
                }
            }
        }
    }

    // Sort by name
    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(Json(ModelsResponse { models }))
}

/// POST /api/show - Show model information
pub async fn show_model(
    State(_state): State<AppState>,
    Json(req): Json<ShowRequest>,
) -> ApiResult<Json<ShowResponse>> {
    use tokio::fs;

    // Get model path using helper
    let model_path = get_model_path(&req.name);

    // Check if model exists (async)
    if fs::metadata(&model_path).await.is_err() {
        return Ok(Json(ShowResponse {
            modelfile: format!("# Model: {}\n# Not found", req.name),
            parameters: "".to_string(),
            template: "".to_string(),
        }));
    }

    // Get model file info (async)
    let metadata = fs::metadata(&model_path).await.ok();
    let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
    let size_mb = size as f64 / 1024.0 / 1024.0;

    // Generate modelfile information
    let modelfile = format!(
        "# Modelfile for {}\n\
         FROM {}\n\
         \n\
         # Model size: {:.2} MB\n\
         # Format: GGUF\n\
         # Path: {}",
        req.name,
        req.name,
        size_mb,
        model_path.display()
    );

    // Default parameters for llama.cpp models
    let parameters = "stop                           [INST]\n\
                      stop                           [/INST]\n\
                      stop                           <<SYS>>\n\
                      stop                           <</SYS>>".to_string();

    // Default chat template
    let template = "{{ if .System }}<<SYS>>{{ .System }}<</SYS>>{{ end }}{{ if .Prompt }}[INST] {{ .Prompt }} [/INST]{{ end }}".to_string();

    Ok(Json(ShowResponse {
        modelfile,
        parameters,
        template,
    }))
}

/// POST /api/generate - Generate text
pub async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    use altair_core::types::{InferenceRequest, ModelSpec};

    let model_name = req.model.clone();
    let prompt = req.prompt.clone();

    // Get model path using helper
    let model_path = get_model_path(&model_name);

    // Create stream for responses
    let stream = async_stream::stream! {
        let start_time = std::time::Instant::now();

        // Load model if needed
        let spec = ModelSpec {
            name: model_name.clone(),
            path: model_path.clone(),
            context_size: 2048,
            gpu_layers: None,
            rope_freq_base: None,
            rope_freq_scale: None,
        };

        let model_id = match state.executor.get_or_load_model(spec).await {
            Ok(id) => id,
            Err(e) => {
                tracing::error!("Failed to load model: {:?}", e);
                let error_response = OllamaStreamResponse {
                    model: model_name.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: format!("Error loading model: {}", e),
                    done: true,
                    total_duration: Some(start_time.elapsed().as_nanos() as u64),
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: Some(0),
                    eval_duration: None,
                    context: None,
                };
                let json = serde_json::to_string(&error_response).unwrap();
                yield Ok(Event::default().data(json));
                return;
            }
        };

        // Create inference request
        let inference_req = InferenceRequest {
            prompt: prompt.clone(),
            messages: vec![], // No structured messages for generate endpoint
            apply_chat_template: false, // Keep raw prompt
            max_tokens: Some(req.options.as_ref().and_then(|o| o.num_predict).unwrap_or(128)),
            temperature: req.options.as_ref().and_then(|o| o.temperature),
            top_p: req.options.as_ref().and_then(|o| o.top_p),
            top_k: req.options.as_ref().and_then(|o| o.top_k),
            stop_sequences: vec![],
            stream: true,
        };

        // Start inference
        let (mut token_rx, result_handle) = match state.executor.infer_streaming(model_id, inference_req).await {
            Ok(handles) => handles,
            Err(e) => {
                tracing::error!("Failed to start inference: {:?}", e);
                let error_response = OllamaStreamResponse {
                    model: model_name.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: format!("Error: {}", e),
                    done: true,
                    total_duration: Some(start_time.elapsed().as_nanos() as u64),
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: Some(0),
                    eval_duration: None,
                    context: None,
                };
                let json = serde_json::to_string(&error_response).unwrap();
                yield Ok(Event::default().data(json));
                return;
            }
        };

        // Stream tokens
        while let Some(token) = token_rx.recv().await {
            let response = OllamaStreamResponse {
                model: model_name.clone(),
                created_at: chrono::Utc::now().to_rfc3339(),
                response: token.token,
                done: false,
                total_duration: None,
                load_duration: None,
                prompt_eval_count: None,
                prompt_eval_duration: None,
                eval_count: None,
                eval_duration: None,
                context: None,
            };

            let json = serde_json::to_string(&response).unwrap();
            yield Ok(Event::default().data(json));
        }

        // Wait for completion
        let completion = match result_handle.await {
            Ok(Ok(comp)) => comp,
            Ok(Err(e)) => {
                tracing::error!("Inference failed: {:?}", e);
                let error_response = OllamaStreamResponse {
                    model: model_name.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: format!("Error: {}", e),
                    done: true,
                    total_duration: Some(start_time.elapsed().as_nanos() as u64),
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: Some(0),
                    eval_duration: None,
                    context: None,
                };
                let json = serde_json::to_string(&error_response).unwrap();
                yield Ok(Event::default().data(json));
                return;
            }
            Err(e) => {
                tracing::error!("Task join failed: {:?}", e);
                let error_response = OllamaStreamResponse {
                    model: model_name.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: format!("Error: {}", e),
                    done: true,
                    total_duration: Some(start_time.elapsed().as_nanos() as u64),
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: Some(0),
                    eval_duration: None,
                    context: None,
                };
                let json = serde_json::to_string(&error_response).unwrap();
                yield Ok(Event::default().data(json));
                return;
            }
        };

        // Send final done message
        let final_response = OllamaStreamResponse {
            model: model_name.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            response: String::new(),
            done: true,
            total_duration: Some(start_time.elapsed().as_nanos() as u64),
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: Some(completion.tokens_generated),
            eval_duration: Some((completion.tokens_generated as f32 / completion.tokens_per_second * 1_000_000_000.0) as u64),
            context: None,
        };

        let json = serde_json::to_string(&final_response).unwrap();
        yield Ok(Event::default().data(json));
    };

    Ok(Sse::new(stream))
}

/// POST /api/chat - Chat completion
pub async fn chat(
    State(state): State<AppState>,
    Json(req): Json<OllamaChatRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    use altair_core::types::{InferenceRequest, ModelSpec};

    let model_name = req.model.clone();

    tracing::info!(
        "Chat request for model {} with {} message(s)",
        model_name,
        req.messages.len()
    );

    // Get model path using helper
    let model_path = get_model_path(&model_name);

    // Create stream for responses
    let stream = async_stream::stream! {
        let start_time = std::time::Instant::now();

        // Load model if needed
        let spec = ModelSpec {
            name: model_name.clone(),
            path: model_path.clone(),
            context_size: 2048,
            gpu_layers: None,
            rope_freq_base: None,
            rope_freq_scale: None,
        };

        let model_id = match state.executor.get_or_load_model(spec).await {
            Ok(id) => id,
            Err(e) => {
                tracing::error!("Failed to load model: {:?}", e);
                let error_response = OllamaStreamResponse {
                    model: model_name.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: format!("Error loading model: {}", e),
                    done: true,
                    total_duration: Some(start_time.elapsed().as_nanos() as u64),
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: Some(0),
                    eval_duration: None,
                    context: None,
                };
                let json = serde_json::to_string(&error_response).unwrap();
                yield Ok(Event::default().data(json));
                return;
            }
        };

        // Create inference request
        let inference_req = InferenceRequest {
            prompt: String::new(), // Not used when messages provided
            messages: req.messages.clone(), // Pass structured messages
            apply_chat_template: true, // Enable chat template
            max_tokens: Some(req.options.as_ref().and_then(|o| o.num_predict).unwrap_or(128)),
            temperature: req.options.as_ref().and_then(|o| o.temperature),
            top_p: req.options.as_ref().and_then(|o| o.top_p),
            top_k: req.options.as_ref().and_then(|o| o.top_k),
            stop_sequences: vec![],
            stream: true,
        };

        // Start inference
        let (mut token_rx, result_handle) = match state.executor.infer_streaming(model_id, inference_req).await {
            Ok(handles) => handles,
            Err(e) => {
                tracing::error!("Failed to start inference: {:?}", e);
                let error_response = OllamaStreamResponse {
                    model: model_name.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: format!("Error: {}", e),
                    done: true,
                    total_duration: Some(start_time.elapsed().as_nanos() as u64),
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: Some(0),
                    eval_duration: None,
                    context: None,
                };
                let json = serde_json::to_string(&error_response).unwrap();
                yield Ok(Event::default().data(json));
                return;
            }
        };

        // Stream tokens
        while let Some(token) = token_rx.recv().await {
            let response = OllamaStreamResponse {
                model: model_name.clone(),
                created_at: chrono::Utc::now().to_rfc3339(),
                response: token.token,
                done: false,
                total_duration: None,
                load_duration: None,
                prompt_eval_count: None,
                prompt_eval_duration: None,
                eval_count: None,
                eval_duration: None,
                context: None,
            };

            let json = serde_json::to_string(&response).unwrap();
            yield Ok(Event::default().data(json));
        }

        // Wait for completion
        let completion = match result_handle.await {
            Ok(Ok(comp)) => comp,
            Ok(Err(e)) => {
                tracing::error!("Inference failed: {:?}", e);
                let error_response = OllamaStreamResponse {
                    model: model_name.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: format!("Error: {}", e),
                    done: true,
                    total_duration: Some(start_time.elapsed().as_nanos() as u64),
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: Some(0),
                    eval_duration: None,
                    context: None,
                };
                let json = serde_json::to_string(&error_response).unwrap();
                yield Ok(Event::default().data(json));
                return;
            }
            Err(e) => {
                tracing::error!("Task join failed: {:?}", e);
                let error_response = OllamaStreamResponse {
                    model: model_name.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: format!("Error: {}", e),
                    done: true,
                    total_duration: Some(start_time.elapsed().as_nanos() as u64),
                    load_duration: None,
                    prompt_eval_count: None,
                    prompt_eval_duration: None,
                    eval_count: Some(0),
                    eval_duration: None,
                    context: None,
                };
                let json = serde_json::to_string(&error_response).unwrap();
                yield Ok(Event::default().data(json));
                return;
            }
        };

        // Send final done message
        let final_response = OllamaStreamResponse {
            model: model_name.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            response: String::new(),
            done: true,
            total_duration: Some(start_time.elapsed().as_nanos() as u64),
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: Some(completion.tokens_generated),
            eval_duration: Some((completion.tokens_generated as f32 / completion.tokens_per_second * 1_000_000_000.0) as u64),
            context: None,
        };

        let json = serde_json::to_string(&final_response).unwrap();
        yield Ok(Event::default().data(json));
    };

    Ok(Sse::new(stream))
}

/// POST /api/pull - Download model
pub async fn pull_model(
    State(_state): State<AppState>,
    Json(req): Json<PullRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    use altair_core::downloader::ModelDownloader;

    let model_name = req.name.clone();

    let stream = async_stream::stream! {
        let downloader = ModelDownloader::new(get_models_dir());
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

        // Spawn download task
        let model_ref = model_name.clone();
        let download_task = tokio::spawn(async move {
            downloader.download_model(&model_ref, progress_tx).await
        });

        // Stream progress updates
        while let Some(progress) = progress_rx.recv().await {
            let response = serde_json::json!({
                "status": progress.status.to_string(),
                "digest": "sha256:...",
                "total": progress.total_bytes,
                "completed": progress.downloaded_bytes,
                "speed": progress.speed_bytes_per_sec,
                "eta": progress.eta_seconds,
                "percentage": progress.percentage,
            });

            yield Ok(Event::default().data(response.to_string()));
        }

        // Wait for download completion
        match download_task.await {
            Ok(Ok(_path)) => {
                let success = serde_json::json!({
                    "status": "success",
                    "digest": "sha256:verified",
                    "total": 0,
                    "completed": 0,
                });
                yield Ok(Event::default().data(success.to_string()));
            }
            Ok(Err(e)) => {
                let error = serde_json::json!({
                    "status": "error",
                    "error": e.to_string(),
                });
                yield Ok(Event::default().data(error.to_string()));
            }
            Err(e) => {
                let error = serde_json::json!({
                    "status": "error",
                    "error": format!("Task failed: {}", e),
                });
                yield Ok(Event::default().data(error.to_string()));
            }
        }
    };

    Ok(Sse::new(stream))
}

/// DELETE /api/delete - Delete model
pub async fn delete_model(
    State(_state): State<AppState>,
    Json(_req): Json<DeleteRequest>,
) -> ApiResult<StatusCode> {
    // TODO: Implement model deletion
    Ok(StatusCode::OK)
}

/// GET /api/ps - List running models/processes
pub async fn list_running(State(state): State<AppState>) -> ApiResult<Json<ProcessesResponse>> {
    let models = state.executor.list_models();

    let running_models = models
        .into_iter()
        .map(|(_, spec)| RunningModel {
            name: spec.name.clone(),
            model: spec.name,
            size: 0, // TODO: Get actual size
            digest: "sha256:placeholder".to_string(),
            expires_at: chrono::Utc::now().to_rfc3339(),
        })
        .collect();

    Ok(Json(ProcessesResponse {
        models: running_models,
    }))
}
