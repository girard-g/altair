# Altair

**High-Performance Local LLM Runtime**

Altair is a modular, high-performance Rust-based LLM runtime designed as a drop-in replacement for Ollama. It features concurrent request handling, GPU acceleration (CUDA/Metal), and both Ollama-compatible and OpenAI-compatible APIs.

## Features

- 🚀 **High Performance**: Continuous batching, efficient memory management, >100 tokens/sec
- 🔄 **Concurrent Processing**: Handle 100+ simultaneous requests efficiently
- 🎯 **Dual API**: Ollama-compatible + OpenAI-compatible endpoints
- 🎮 **GPU Acceleration**: CUDA (Windows/Linux) and Metal (macOS) support
- 📦 **Model Management**: Download from HuggingFace, automatic caching
- 🔌 **Plug & Play**: Drop-in replacement for Ollama in existing projects
- 🌊 **Streaming First**: Server-Sent Events and NDJSON streaming
- 📊 **Observability**: Prometheus metrics, OpenTelemetry tracing

## GPU Support Status

Altair supports GPU acceleration on multiple platforms:

| Backend | Platform | Status | Notes |
|---------|----------|--------|-------|
| **CUDA** | Linux, Windows | ✅ Fully Supported | NVIDIA GPUs, driver 450.80.02+ |
| **Metal** | macOS | ✅ Fully Supported | Apple Silicon & AMD GPUs, macOS 10.15+ |
| **CPU** | All | ✅ Fully Supported | Automatic fallback |
| ROCm | Linux | ⏳ Planned | AMD GPUs |
| Vulkan | Cross-platform | ⏳ Planned | Universal GPU support |

**Performance**:
- CUDA/Metal: 50-100+ tokens/sec (varies by model and GPU)
- CPU: 5-20 tokens/sec (varies by model and CPU)

**Build with GPU support**:
```bash
# CUDA (NVIDIA GPUs on Linux/Windows)
cargo build --release --features cuda

# Metal (Apple Silicon/AMD on macOS)
cargo build --release --features metal
```

See [GPU Backend Guide](docs/GPU_BACKEND_GUIDE.md) for detailed setup instructions.

## Quick Start

```bash
# Install
cargo install altair-cli

# Start server
altair serve

# Download a model
altair pull llama2:7b

# Run inference
altair run llama2:7b "Hello, how are you?"
```

## API Compatibility

### Ollama API
- `POST /api/generate` - Text generation
- `POST /api/chat` - Chat completions
- `POST /api/pull` - Download models
- `GET /api/tags` - List models
- `DELETE /api/delete` - Remove models
- `GET /api/ps` - Running processes

### OpenAI-Compatible API
- `POST /v1/completions` - Completions
- `POST /v1/chat/completions` - Chat
- `POST /v1/embeddings` - Embeddings
- `GET /v1/models` - List models

## Architecture

Altair is built with a modular architecture:

- **altair-core**: Core types, errors, configuration
- **altair-inference**: llama.cpp FFI wrapper, inference engine
- **altair-api**: HTTP API layer (Axum)
- **altair-model-registry**: HuggingFace integration, downloads
- **altair-cli**: Command-line interface
- **altair**: Umbrella crate

## Development

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Start development server
cargo run --bin altair-cli -- serve
```

## License

MIT OR Apache-2.0
