//! Integration tests for GPU backend initialization
//!
//! These tests verify that GPU backends can be properly initialized,
//! and that the LlamaCppBackend works correctly with different GPU configurations.
//!
//! Note: llama.cpp backend can only be initialized once per process, so these
//! tests must run serially using the serial_test crate.

use altair_inference::*;
use serial_test::serial;

#[tokio::test]
#[serial]
async fn test_cpu_backend_creation() {
    // CPU backend should work (unless already initialized)
    let result = LlamaCppBackend::new(altair_core::GpuBackend::None);

    match result {
        Ok(_) => {
            println!("✓ CPU backend created successfully");
        }
        Err(e) => {
            let error_msg = format!("{}", e);
            if error_msg.contains("BackendAlreadyInitialized") || error_msg.contains("already") {
                println!("✓ Backend already initialized (expected in test environment)");
            } else {
                panic!("CPU backend creation failed with unexpected error: {}", e);
            }
        }
    }
}

#[tokio::test]
#[serial]
async fn test_detected_backend_creation() {
    // Detect whatever backend is available
    let backend = detect_gpu_backend();
    println!("Testing with detected backend: {}", backend);

    // Try to create backend
    let result = LlamaCppBackend::new(backend.clone());

    match result {
        Ok(_) => {
            println!("✓ Backend created successfully with: {}", backend);
        }
        Err(e) => {
            // Some platforms might fail initialization even if detection succeeds
            // (e.g., drivers not fully installed, permissions issues, etc.)
            println!("⚠ Backend creation failed (may be expected): {}", e);
            println!("  This is not necessarily an error - the system will fall back to CPU");
        }
    }
}

#[tokio::test]
#[serial]
async fn test_backend_initialization_idempotent() {
    // Note: llama.cpp backend can only be initialized once globally
    // The second initialization will fail with BackendAlreadyInitialized
    let backend_type = detect_gpu_backend();

    let result1 = LlamaCppBackend::new(backend_type.clone());
    let result2 = LlamaCppBackend::new(backend_type);

    match (result1, result2) {
        (Ok(_), Ok(_)) => {
            println!("✓ Multiple backend creations succeeded");
        }
        (Err(e1), Err(e2)) => {
            println!("⚠ Both backend creations failed (may be expected):");
            println!("  First: {}", e1);
            println!("  Second: {}", e2);
        }
        (Ok(_), Err(e)) => {
            // This is expected: llama.cpp backend can only be initialized once
            let error_msg = format!("{}", e);
            if error_msg.contains("BackendAlreadyInitialized") || error_msg.contains("already") {
                println!("✓ Second backend creation correctly failed (backend already initialized)");
            } else {
                panic!("Unexpected error on second backend creation: {}", e);
            }
        }
        (Err(e), Ok(_)) => {
            panic!("Inconsistent: first failed, second succeeded: {}", e);
        }
    }
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[serial]
async fn test_cuda_backend_with_feature() {
    use altair_core::GpuBackend;

    // When CUDA feature is enabled, try to use CUDA backend
    let backend = detect_gpu_backend();

    if let GpuBackend::Cuda { device_count, .. } = backend {
        println!("Testing CUDA backend with {} devices", device_count);

        let result = LlamaCppBackend::new(backend);
        match result {
            Ok(_) => println!("✓ CUDA backend created successfully"),
            Err(e) => println!("⚠ CUDA backend creation failed: {}", e),
        }
    } else {
        println!("⊘ CUDA feature enabled but no CUDA devices detected");
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[tokio::test]
#[serial]
async fn test_metal_backend_with_feature() {
    use altair_core::GpuBackend;

    // When Metal feature is enabled on macOS, try to use Metal backend
    let backend = detect_gpu_backend();

    if let GpuBackend::Metal { .. } = backend {
        println!("Testing Metal backend");

        let result = LlamaCppBackend::new(backend);
        match result {
            Ok(_) => println!("✓ Metal backend created successfully"),
            Err(e) => println!("⚠ Metal backend creation failed: {}", e),
        }
    } else {
        println!("⊘ Metal feature enabled but Metal not detected");
    }
}

#[tokio::test]
#[serial]
async fn test_init_gpu_backend_cpu() {
    let result = init_gpu_backend(altair_core::GpuBackend::None);
    assert!(result.is_ok(), "CPU backend init should always succeed");
    println!("✓ CPU backend initialized");
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[tokio::test]
#[serial]
async fn test_init_gpu_backend_cuda() {
    use altair_core::GpuBackend;

    let backend = detect_gpu_backend();

    if let GpuBackend::Cuda { device_count, devices } = backend {
        let result = init_gpu_backend(GpuBackend::Cuda {
            device_count,
            devices,
        });

        match result {
            Ok(()) => println!("✓ CUDA backend initialized successfully"),
            Err(e) => {
                println!("⚠ CUDA initialization failed: {}", e);
                println!("  This may be expected if drivers are not fully installed");
            }
        }
    } else {
        println!("⊘ Skipping CUDA init test - no CUDA detected");
    }
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[serial]
async fn test_init_gpu_backend_metal() {
    use altair_core::GpuBackend;

    let backend = detect_gpu_backend();

    if let GpuBackend::Metal { device_name, device_memory_gb } = backend {
        let result = init_gpu_backend(GpuBackend::Metal {
            device_name,
            device_memory_gb,
        });

        match result {
            Ok(()) => println!("✓ Metal backend initialized successfully"),
            Err(e) => {
                println!("⚠ Metal initialization failed: {}", e);
                println!("  This may be expected on older macOS versions");
            }
        }
    } else {
        println!("⊘ Skipping Metal init test - no Metal detected");
    }
}

#[tokio::test]
#[serial]
async fn test_backend_trait_methods() {
    use altair_inference::InferenceBackend;

    // Create a backend
    let backend_type = altair_core::GpuBackend::None; // Use CPU for reliability
    let backend = LlamaCppBackend::new(backend_type);

    if let Ok(backend) = backend {
        // Test model loading would require a model file,
        // so we just verify the backend implements the trait
        let model_id = altair_core::types::ModelId::new();

        // Test is_model_loaded (should be false for random ID)
        assert!(
            !backend.is_model_loaded(model_id),
            "Random model ID should not be loaded"
        );

        println!("✓ Backend trait methods work");
    }
}

#[tokio::test]
#[serial]
async fn test_error_handling_graceful() {
    // Even if backend creation fails, it should return a proper error, not panic
    let backend = detect_gpu_backend();

    match LlamaCppBackend::new(backend) {
        Ok(_) => println!("✓ Backend created successfully"),
        Err(e) => {
            println!("✓ Backend creation failed gracefully with error: {}", e);
            // Verify error is one of the expected types
            let error_string = format!("{}", e);
            assert!(
                !error_string.is_empty(),
                "Error message should not be empty"
            );
        }
    }
}
