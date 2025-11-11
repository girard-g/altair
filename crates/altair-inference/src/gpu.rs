use altair_core::{error::*, GpuBackend};

/// Detect available GPU backend on the current system
pub fn detect_gpu_backend() -> GpuBackend {
    // Try CUDA first (Windows/Linux)
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        if let Some(device_count) = detect_cuda() {
            tracing::info!("Detected CUDA with {} device(s)", device_count);
            return GpuBackend::Cuda { device_count };
        }
    }

    // Try Metal (macOS)
    #[cfg(target_os = "macos")]
    {
        if detect_metal() {
            tracing::info!("Detected Metal backend");
            return GpuBackend::Metal;
        }
    }

    // Fallback to CPU
    tracing::info!("No GPU detected, using CPU");
    GpuBackend::None
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn detect_cuda() -> Option<u32> {
    // TODO: Implement CUDA detection
    // For now, we'll try to detect CUDA by checking for nvidia-smi or CUDA runtime

    // Method 1: Check nvidia-smi
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=count")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            if let Ok(count_str) = String::from_utf8(output.stdout) {
                if let Ok(count) = count_str.trim().parse::<u32>() {
                    if count > 0 {
                        return Some(count);
                    }
                }
            }
        }
    }

    // Method 2: Check for CUDA environment variables
    if std::env::var("CUDA_PATH").is_ok() || std::env::var("CUDA_HOME").is_ok() {
        // Assume at least one device if CUDA is installed
        return Some(1);
    }

    None
}

#[cfg(target_os = "macos")]
fn detect_metal() -> bool {
    // TODO: Implement proper Metal detection using metal-rs
    // For now, assume Metal is available on macOS (it's standard on modern Macs)

    // Check macOS version - Metal requires 10.11+
    if let Ok(output) = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
    {
        if output.status.success() {
            if let Ok(version) = String::from_utf8(output.stdout) {
                // Simple check: if we can get the version, Metal is likely available
                return !version.is_empty();
            }
        }
    }

    // Conservative fallback
    true
}

#[allow(dead_code)]
fn detect_metal_placeholder() -> bool {
    detect_metal()
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn detect_cuda() -> Option<u32> {
    None
}

#[cfg(not(target_os = "macos"))]
fn detect_metal() -> bool {
    false
}

/// Initialize GPU backend
pub fn init_gpu_backend(backend: GpuBackend) -> Result<()> {
    match backend {
        GpuBackend::Cuda { device_count } => {
            tracing::info!("Initializing CUDA backend with {} device(s)", device_count);
            // TODO: Initialize CUDA via llama.cpp
            Ok(())
        }
        GpuBackend::Metal => {
            tracing::info!("Initializing Metal backend");
            // TODO: Initialize Metal via llama.cpp
            Ok(())
        }
        GpuBackend::None => {
            tracing::info!("Running on CPU");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        let backend = detect_gpu_backend();
        println!("Detected GPU backend: {}", backend);
        // This test just ensures detection doesn't crash
    }
}
