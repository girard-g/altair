//! Integration tests for GPU detection
//!
//! These tests verify that GPU detection works correctly across different
//! platforms and gracefully handles cases where GPUs are not available.

use altair_inference::*;

#[test]
fn test_detect_gpu_backend_no_panic() {
    // GPU detection should never panic, regardless of hardware
    let backend = detect_gpu_backend();
    println!("Detected backend: {}", backend);

    // Verify it's one of the valid types
    match backend {
        altair_core::GpuBackend::Cuda { device_count, devices } => {
            assert!(device_count > 0, "CUDA device count must be positive");
            println!("✓ CUDA detected with {} device(s)", device_count);
            if let Some(devs) = devices {
                assert_eq!(devs.len() as u32, device_count, "Device count mismatch");
            }
        }
        altair_core::GpuBackend::Metal { device_name, device_memory_gb } => {
            println!("✓ Metal detected");
            if let Some(name) = device_name {
                println!("  Device: {}", name);
            }
            if let Some(mem) = device_memory_gb {
                println!("  Memory: {}GB", mem);
            }
        }
        altair_core::GpuBackend::None => {
            println!("✓ CPU-only mode (no GPU detected)");
        }
    }
}

#[test]
fn test_cpu_fallback_graceful() {
    // Even if no GPU is detected, the system should work
    let backend = detect_gpu_backend();

    // Initialization should succeed for any backend
    let init_result = init_gpu_backend(backend);

    match init_result {
        Ok(()) => println!("✓ Backend initialized successfully"),
        Err(e) => {
            // Some platforms may not support certain backends
            println!("Backend init failed (may be expected): {}", e);
        }
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[test]
fn test_cuda_detection_on_supported_platform() {
    use altair_core::GpuBackend;

    // On Windows/Linux, try CUDA detection
    let backend = detect_gpu_backend();

    match backend {
        GpuBackend::Cuda { device_count, devices } => {
            println!("CUDA detected:");
            println!("  Device count: {}", device_count);

            if let Some(devs) = devices {
                for dev in devs {
                    println!(
                        "  Device {}: {} (Compute {}.{}, {}GB)",
                        dev.device_id,
                        dev.name,
                        dev.compute_capability.0,
                        dev.compute_capability.1,
                        dev.total_memory_bytes / (1024 * 1024 * 1024)
                    );

                    // Sanity checks
                    assert!(!dev.name.is_empty(), "Device name should not be empty");
                    assert!(
                        dev.total_memory_bytes > 0,
                        "Device memory should be positive"
                    );
                }
            }
        }
        GpuBackend::None => {
            println!("No CUDA detected on this system (expected if no NVIDIA GPU)");
        }
        _ => {
            // On Linux, we might get Metal if running in a VM or unusual config
            println!("Unexpected backend type: {}", backend);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_detection_on_macos() {
    use altair_core::GpuBackend;

    // On macOS, we should detect Metal (unless very old system)
    let backend = detect_gpu_backend();

    match backend {
        GpuBackend::Metal { device_name, device_memory_gb } => {
            println!("Metal detected:");
            if let Some(name) = device_name {
                println!("  Device: {}", name);
                assert!(!name.is_empty(), "Metal device name should not be empty");
            }
            if let Some(mem) = device_memory_gb {
                println!("  Memory: {}GB", mem);
                assert!(mem > 0, "Metal device memory should be positive");
            }
        }
        GpuBackend::None => {
            println!("No Metal detected (unusual for modern macOS, but possible on very old systems)");
        }
        _ => {
            println!("Unexpected backend type on macOS: {}", backend);
        }
    }
}

#[test]
fn test_gpu_backend_display() {
    // Test the Display implementation
    use altair_core::GpuBackend;

    let cpu = GpuBackend::None;
    let cpu_str = cpu.to_string();
    assert_eq!(cpu_str, "CPU");

    let cuda = GpuBackend::Cuda {
        device_count: 2,
        devices: None,
    };
    let cuda_str = cuda.to_string();
    assert!(cuda_str.contains("CUDA"));
    assert!(cuda_str.contains("2 devices"));

    let metal = GpuBackend::Metal {
        device_name: Some("Apple M3".to_string()),
        device_memory_gb: Some(32),
    };
    let metal_str = metal.to_string();
    assert!(metal_str.contains("Metal"));
    assert!(metal_str.contains("Apple M3"));
}

#[test]
fn test_multiple_detections_consistent() {
    // Running detection multiple times should give consistent results
    let backend1 = detect_gpu_backend();
    let backend2 = detect_gpu_backend();

    // Compare backends (we can't use PartialEq directly due to Vec in CudaDeviceInfo)
    match (&backend1, &backend2) {
        (
            altair_core::GpuBackend::Cuda { device_count: c1, .. },
            altair_core::GpuBackend::Cuda { device_count: c2, .. },
        ) => {
            assert_eq!(c1, c2, "CUDA device count should be consistent");
        }
        (altair_core::GpuBackend::Metal { .. }, altair_core::GpuBackend::Metal { .. }) => {
            // Both detected Metal, consistent
        }
        (altair_core::GpuBackend::None, altair_core::GpuBackend::None) => {
            // Both detected no GPU, consistent
        }
        _ => {
            panic!(
                "Inconsistent detection: {:?} vs {:?}",
                backend1, backend2
            );
        }
    }
}

#[test]
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn test_cuda_device_info_sanity() {
    use altair_core::{CudaDeviceInfo, GpuBackend};

    // Create a sample device info for testing
    let device = CudaDeviceInfo {
        device_id: 0,
        name: "Test GPU".to_string(),
        compute_capability: (7, 5),
        total_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
    };

    assert_eq!(device.device_id, 0);
    assert_eq!(device.name, "Test GPU");
    assert_eq!(device.compute_capability, (7, 5));
    assert_eq!(device.total_memory_bytes, 8 * 1024 * 1024 * 1024);

    // Test creating a GpuBackend with this device
    let backend = GpuBackend::Cuda {
        device_count: 1,
        devices: Some(vec![device]),
    };

    match backend {
        GpuBackend::Cuda { device_count, devices } => {
            assert_eq!(device_count, 1);
            assert!(devices.is_some());
            let devs = devices.unwrap();
            assert_eq!(devs.len(), 1);
            assert_eq!(devs[0].name, "Test GPU");
        }
        _ => panic!("Expected CUDA backend"),
    }
}
