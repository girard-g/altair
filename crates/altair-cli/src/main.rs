use altair_api::{create_router, AppState};
use altair_inference::{detect_gpu_backend, init_gpu_backend, InferenceExecutor, LlamaCppBackend};
use altair_model_registry::{LocalModelRegistry, ModelRegistry};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "altair")]
#[command(about = "High-performance local LLM runtime", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, global = true, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the API server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(long, default_value = "11434")]
        port: u16,

        /// GPU backend (auto, cuda, metal, cpu)
        #[arg(long, default_value = "auto")]
        gpu: String,

        /// Maximum concurrent inference requests
        #[arg(long, default_value = "4")]
        max_concurrent: usize,
    },

    /// Run a one-off inference
    Run {
        /// Model name or path
        model: String,

        /// Prompt text
        prompt: String,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Download a model
    Pull {
        /// Model reference (e.g., "llama2:7b" or "username/model")
        model: String,
    },

    /// List downloaded models
    List {
        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Delete a model
    Rm {
        /// Model name
        model: String,
    },

    /// Show model information
    Show {
        /// Model name
        model: String,
    },

    /// List running models
    Ps {
        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    match cli.command {
        Commands::Serve {
            host,
            port,
            gpu,
            max_concurrent,
        } => {
            serve_command(host, port, gpu, max_concurrent).await?;
        }
        Commands::Run {
            model,
            prompt,
            format,
        } => {
            run_command(model, prompt, format).await?;
        }
        Commands::Pull { model } => {
            pull_command(model).await?;
        }
        Commands::List { format } => {
            list_command(format).await?;
        }
        Commands::Rm { model } => {
            rm_command(model).await?;
        }
        Commands::Show { model } => {
            show_command(model).await?;
        }
        Commands::Ps { format } => {
            ps_command(format).await?;
        }
    }

    Ok(())
}

async fn serve_command(
    host: String,
    port: u16,
    gpu: String,
    max_concurrent: usize,
) -> anyhow::Result<()> {
    tracing::info!("Starting Altair LLM Runtime");

    // Detect or select GPU backend
    let gpu_backend = if gpu == "auto" {
        detect_gpu_backend()
    } else {
        match gpu.as_str() {
            "cuda" => altair_core::GpuBackend::Cuda {
                device_count: 1,
                devices: None,
            },
            "metal" => altair_core::GpuBackend::Metal {
                device_name: None,
                device_memory_gb: None,
            },
            "cpu" => altair_core::GpuBackend::None,
            _ => {
                tracing::warn!("Unknown GPU backend '{}', using auto-detection", gpu);
                detect_gpu_backend()
            }
        }
    };

    tracing::info!("Using GPU backend: {}", gpu_backend);

    // Initialize GPU backend
    init_gpu_backend(gpu_backend.clone())?;

    // Create inference backend
    let backend = Arc::new(LlamaCppBackend::new(gpu_backend.clone())?);

    // Create inference executor
    let executor = Arc::new(InferenceExecutor::new(backend, max_concurrent)?);

    // Create application state
    let state = AppState::new(executor);

    // Create API router
    let app = create_router(state);

    // Start server
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("✓ Server listening on {}", addr);
    tracing::info!("✓ GPU backend: {}", gpu_backend);
    tracing::info!("✓ Max concurrent requests: {}", max_concurrent);

    // Create graceful shutdown signal
    let shutdown_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
        tracing::info!("Received shutdown signal, draining requests...");
    };

    // Serve with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    tracing::info!("Shutting down gracefully...");

    Ok(())
}

async fn run_command(model: String, prompt: String, _format: String) -> anyhow::Result<()> {
    use altair_core::types::*;
    use altair_inference::{gpu::detect_gpu_backend, InferenceBackend, LlamaCppBackend};
    use std::path::PathBuf;

    tracing::info!("Running inference with model: {}", model);

    // Find the model file
    let models_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".altair")
        .join("models");

    let mut model = model.clone();

    if !model.contains(".gguf") {
        model.push_str(".gguf");
    }

    let model_path = models_dir.join(&model);

    if !model_path.exists() {
        eprintln!("Error: Model file not found: {}", model_path.display());
        eprintln!("Available models in {}:", models_dir.display());
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().and_then(|s| s.to_str()) == Some("gguf") {
                    eprintln!("  - {}", entry.file_name().to_string_lossy());
                }
            }
        }
        return Err(anyhow::anyhow!("Model not found"));
    }

    let gpu_backend = detect_gpu_backend();
    println!("GPU Backend: {}", gpu_backend);

    let backend = LlamaCppBackend::new(gpu_backend)?;

    println!("Loading model from: {}", model_path.display());
    let spec = ModelSpec {
        name: model.clone(),
        path: model_path,
        context_size: 2048,
        gpu_layers: None,
        rope_freq_base: None,
        rope_freq_scale: None,
    };

    let model_id = backend.load_model(&spec).await?;
    println!("Model loaded successfully!");

    let request = InferenceRequest {
        prompt,
        messages: vec![],
        apply_chat_template: true,
        max_tokens: Some(9999),
        temperature: Some(0.8),
        top_p: Some(0.9),
        top_k: Some(40),
        stop_sequences: vec![],
        stream: true,
    };

    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel::<TokenResponse>();

    let print_task = tokio::spawn(async move {
        print!("\nResponse: ");
        while let Some(token) = token_rx.recv().await {
            print!("{}", token.token);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!();
    });

    let result = backend
        .generate_streaming(model_id, request, token_tx)
        .await?;

    print_task.await?;

    println!("\nStatistics:");
    println!("  Tokens generated: {}", result.tokens_generated);
    println!("  Tokens/second: {:.2}", result.tokens_per_second);
    println!("  Stop reason: {:?}", result.stop_reason);

    Ok(())
}

async fn pull_command(model: String) -> anyhow::Result<()> {
    use altair_core::downloader::{DownloadStatus, ModelDownloader};
    use altair_core::paths::get_models_dir;
    use indicatif::{ProgressBar, ProgressStyle};

    println!("Pulling model: {}", model);

    let downloader = ModelDownloader::new(get_models_dir());
    let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap()
            .progress_chars("#>-")
    );

    // Spawn download task
    let model_ref = model.clone();
    let download_task =
        tokio::spawn(async move { downloader.download_model(&model_ref, progress_tx).await });

    // Update progress bar
    while let Some(progress) = progress_rx.recv().await {
        match progress.status {
            DownloadStatus::Starting => {
                pb.set_message("Starting download...");
            }
            DownloadStatus::Downloading => {
                pb.set_length(progress.total_bytes);
                pb.set_position(progress.downloaded_bytes);
            }
            DownloadStatus::Verifying => {
                pb.set_message("Verifying SHA256...");
            }
            DownloadStatus::Complete => {
                pb.finish_with_message("Download complete!");
            }
            DownloadStatus::Error(ref e) => {
                pb.abandon_with_message(format!("Error: {}", e));
            }
        }
    }

    // Wait for completion
    match download_task.await? {
        Ok(path) => {
            println!("Model saved to: {:?}", path);
            Ok(())
        }
        Err(e) => Err(anyhow::anyhow!("Download failed: {}", e)),
    }
}

async fn list_command(format: String) -> anyhow::Result<()> {
    let models_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".altair")
        .join("models");

    let registry = LocalModelRegistry::new(models_dir)?;
    let models = registry.list_models()?;

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&models)?);
    } else {
        println!("NAME\t\t\tSIZE\t\tMODIFIED");
        println!("{}", "-".repeat(80));
        for model in models {
            println!(
                "{}\t\t{}\t\t{}",
                model.name,
                format_size(model.size),
                model.modified_at
            );
        }
    }

    Ok(())
}

async fn rm_command(model: String) -> anyhow::Result<()> {
    tracing::info!("Deleting model: {}", model);

    let models_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".altair")
        .join("models");

    let registry = LocalModelRegistry::new(models_dir)?;
    registry.delete_model(&model)?;

    println!("✓ Model deleted: {}", model);

    Ok(())
}

async fn show_command(model: String) -> anyhow::Result<()> {
    tracing::info!("Showing model info: {}", model);

    // TODO: Implement model info display
    println!("Model: {}", model);
    println!("[Model info not implemented yet]");

    Ok(())
}

async fn ps_command(_format: String) -> anyhow::Result<()> {
    // TODO: Implement running models listing
    println!("Running models:");
    println!("[Not implemented yet]");

    Ok(())
}

fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}
