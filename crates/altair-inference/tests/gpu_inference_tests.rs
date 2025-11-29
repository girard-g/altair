//! End-to-end GPU inference tests
//!
//! These tests verify that inference actually works with GPU backends.
//! They are marked with #[ignore] by default because they require a model file.
//!
//! To run these tests:
//! 1. Download a small test model (e.g., TinyLlama or similar GGUF model)
//! 2. Set the MODEL_PATH environment variable to point to the model
//! 3. Run: cargo test --package altair-inference --test gpu_inference_tests -- --ignored

use altair_inference::*;
use std::path::PathBuf;

/// Get test model path from environment or use default
fn get_test_model_path() -> Option<PathBuf> {
    std::env::var("TEST_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            // Try some common locations for test models
            let common_paths = vec![
                "models/tinyllama-1.1b-chat-v0.6.Q4_K_M.gguf",
                "../models/tinyllama-1.1b-chat-v0.6.Q4_K_M.gguf",
                "../../models/tinyllama-1.1b-chat-v0.6.Q4_K_M.gguf",
            ];

            for path in common_paths {
                let pb = PathBuf::from(path);
                if pb.exists() {
                    return Some(pb);
                }
            }

            None
        })
}

#[tokio::test]
#[ignore] // Requires model file - run with --ignored
async fn test_inference_with_cpu() {
    use altair_core::types::*;
    use altair_core::GpuBackend;

    let model_path = match get_test_model_path() {
        Some(path) => path,
        None => {
            println!("⊘ Skipping test - no model file found");
            println!("  Set TEST_MODEL_PATH environment variable to run this test");
            return;
        }
    };

    println!("Using model: {:?}", model_path);

    // Create CPU backend
    let backend = LlamaCppBackend::new(GpuBackend::None).expect("Failed to create CPU backend");

    // Load model
    let spec = ModelSpec {
        name: "test-model".to_string(),
        path: model_path,
        context_size: 2048,
        gpu_layers: None,
        rope_freq_base: None,
        rope_freq_scale: None,
    };

    let model_id = backend.load_model(&spec).await.expect("Failed to load model");
    println!("✓ Model loaded with ID: {}", model_id);

    // Create inference request
    let request = InferenceRequest {
        prompt: "Once upon a time".to_string(),
        messages: vec![],
        apply_chat_template: false,
        max_tokens: Some(20),
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(40),
        stop_sequences: vec![],
        stream: false,
    };

    // Create channel for tokens
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    // Run inference
    let inference_task = tokio::spawn(async move {
        backend.generate_streaming(model_id, request, tx).await
    });

    // Collect tokens
    while let Some(token_resp) = rx.recv().await {
        print!("{}", token_resp.token);
    }
    println!();

    // Wait for completion
    let result = inference_task.await.expect("Inference task panicked");
    match result {
        Ok(completion) => {
            println!("✓ Inference completed:");
            println!("  Tokens generated: {}", completion.tokens_generated);
            println!("  Tokens/sec: {:.2}", completion.tokens_per_second);
            println!("  Stop reason: {:?}", completion.stop_reason);
            assert!(completion.tokens_generated > 0, "Should generate at least one token");
        }
        Err(e) => panic!("Inference failed: {}", e),
    }
}

#[tokio::test]
#[ignore] // Requires model file and GPU - run with --ignored
async fn test_inference_with_detected_gpu() {
    use altair_core::types::*;

    let model_path = match get_test_model_path() {
        Some(path) => path,
        None => {
            println!("⊘ Skipping test - no model file found");
            return;
        }
    };

    // Detect GPU backend
    let gpu_backend = detect_gpu_backend();
    println!("Testing inference with backend: {}", gpu_backend);

    // Skip if no GPU detected
    if matches!(gpu_backend, altair_core::GpuBackend::None) {
        println!("⊘ Skipping test - no GPU detected");
        return;
    }

    // Initialize GPU backend
    if let Err(e) = init_gpu_backend(gpu_backend.clone()) {
        println!("⊘ Skipping test - GPU init failed: {}", e);
        return;
    }

    // Create backend with GPU
    let backend = match LlamaCppBackend::new(gpu_backend) {
        Ok(b) => b,
        Err(e) => {
            println!("⊘ Skipping test - backend creation failed: {}", e);
            return;
        }
    };

    // Load model
    let spec = ModelSpec {
        name: "test-model-gpu".to_string(),
        path: model_path,
        context_size: 2048,
        gpu_layers: None, // Will be determined by backend
        rope_freq_base: None,
        rope_freq_scale: None,
    };

    let model_id = backend.load_model(&spec).await.expect("Failed to load model");
    println!("✓ Model loaded with GPU acceleration");

    // Create inference request
    let request = InferenceRequest {
        prompt: "The quick brown fox".to_string(),
        messages: vec![],
        apply_chat_template: false,
        max_tokens: Some(20),
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(40),
        stop_sequences: vec![],
        stream: false,
    };

    // Create channel for tokens
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    // Run inference
    let inference_task = tokio::spawn(async move {
        backend.generate_streaming(model_id, request, tx).await
    });

    // Collect tokens
    while let Some(token_resp) = rx.recv().await {
        print!("{}", token_resp.token);
    }
    println!();

    // Wait for completion
    let result = inference_task.await.expect("Inference task panicked");
    match result {
        Ok(completion) => {
            println!("✓ GPU inference completed:");
            println!("  Tokens generated: {}", completion.tokens_generated);
            println!("  Tokens/sec: {:.2}", completion.tokens_per_second);
            println!("  Stop reason: {:?}", completion.stop_reason);

            // GPU inference should be reasonably fast
            assert!(
                completion.tokens_per_second > 0.1,
                "Tokens/sec should be positive"
            );
        }
        Err(e) => panic!("GPU inference failed: {}", e),
    }
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore] // Requires model file and CUDA GPU - run with --ignored
async fn test_inference_with_cuda() {
    use altair_core::types::*;
    use altair_core::GpuBackend;

    let model_path = match get_test_model_path() {
        Some(path) => path,
        None => {
            println!("⊘ Skipping test - no model file found");
            return;
        }
    };

    // Detect and verify CUDA
    let backend_type = detect_gpu_backend();
    if !matches!(backend_type, GpuBackend::Cuda { .. }) {
        println!("⊘ Skipping test - CUDA not detected");
        return;
    }

    println!("Running CUDA inference test with: {}", backend_type);

    // Create CUDA backend
    let backend = match LlamaCppBackend::new(backend_type) {
        Ok(b) => b,
        Err(e) => {
            println!("⊘ Skipping test - CUDA backend creation failed: {}", e);
            return;
        }
    };

    // Load and run inference
    let spec = ModelSpec {
        name: "test-cuda".to_string(),
        path: model_path,
        context_size: 2048,
        gpu_layers: None,
        rope_freq_base: None,
        rope_freq_scale: None,
    };

    let model_id = backend.load_model(&spec).await.expect("Failed to load model");

    let request = InferenceRequest {
        prompt: "Hello, CUDA!".to_string(),
        messages: vec![],
        apply_chat_template: false,
        max_tokens: Some(10),
        temperature: Some(0.7),
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        stream: false,
    };

    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let result = backend.generate_streaming(model_id, request, tx).await;

    match result {
        Ok(completion) => {
            println!("✓ CUDA inference successful");
            println!("  Tokens/sec: {:.2}", completion.tokens_per_second);
        }
        Err(e) => panic!("CUDA inference failed: {}", e),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[tokio::test]
#[ignore] // Requires model file and Metal GPU - run with --ignored
async fn test_inference_with_metal() {
    use altair_core::types::*;
    use altair_core::GpuBackend;

    let model_path = match get_test_model_path() {
        Some(path) => path,
        None => {
            println!("⊘ Skipping test - no model file found");
            return;
        }
    };

    // Detect and verify Metal
    let backend_type = detect_gpu_backend();
    if !matches!(backend_type, GpuBackend::Metal { .. }) {
        println!("⊘ Skipping test - Metal not detected");
        return;
    }

    println!("Running Metal inference test with: {}", backend_type);

    // Create Metal backend
    let backend = match LlamaCppBackend::new(backend_type) {
        Ok(b) => b,
        Err(e) => {
            println!("⊘ Skipping test - Metal backend creation failed: {}", e);
            return;
        }
    };

    // Load and run inference
    let spec = ModelSpec {
        name: "test-metal".to_string(),
        path: model_path,
        context_size: 2048,
        gpu_layers: None,
        rope_freq_base: None,
        rope_freq_scale: None,
    };

    let model_id = backend.load_model(&spec).await.expect("Failed to load model");

    let request = InferenceRequest {
        prompt: "Hello, Metal!".to_string(),
        messages: vec![],
        apply_chat_template: false,
        max_tokens: Some(10),
        temperature: Some(0.7),
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        stream: false,
    };

    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let result = backend.generate_streaming(model_id, request, tx).await;

    match result {
        Ok(completion) => {
            println!("✓ Metal inference successful");
            println!("  Tokens/sec: {:.2}", completion.tokens_per_second);
        }
        Err(e) => panic!("Metal inference failed: {}", e),
    }
}
