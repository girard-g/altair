use altair_core::{error::*, types::*, GpuBackend};
use std::path::Path;

/// Detect available GPU backend on the current system
pub fn detect_gpu_backend() -> GpuBackend {
    // Try CUDA first (Windows/Linux)
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        match detect_cuda() {
            Ok(Some((device_count, devices))) => {
                tracing::info!("Detected CUDA with {} device(s)", device_count);
                if let Some(ref devs) = devices {
                    for dev in devs {
                        tracing::info!(
                            "  Device {}: {} (Compute {}.{}, {}MB)",
                            dev.device_id,
                            dev.name,
                            dev.compute_capability.0,
                            dev.compute_capability.1,
                            dev.total_memory_bytes / (1024 * 1024)
                        );
                    }
                }
                return GpuBackend::Cuda {
                    device_count,
                    devices,
                };
            }
            Ok(None) => {
                tracing::debug!("CUDA not detected");
            }
            Err(e) => {
                tracing::warn!("CUDA detection failed: {}", e);
            }
        }
    }

    // Try Metal (macOS)
    #[cfg(target_os = "macos")]
    {
        match detect_metal() {
            Ok(Some((device_name, device_memory_gb))) => {
                tracing::info!(
                    "Detected Metal backend: {} ({}GB)",
                    device_name.as_deref().unwrap_or("Unknown"),
                    device_memory_gb.unwrap_or(0)
                );
                return GpuBackend::Metal {
                    device_name,
                    device_memory_gb,
                };
            }
            Ok(None) => {
                tracing::debug!("Metal not detected");
            }
            Err(e) => {
                tracing::warn!("Metal detection failed: {}", e);
            }
        }
    }

    // Fallback to CPU
    tracing::info!("No GPU detected, using CPU");
    GpuBackend::None
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn detect_cuda() -> Result<Option<(u32, Option<Vec<CudaDeviceInfo>>)>> {
    tracing::debug!("Attempting CUDA detection");

    // Method 1: Try nvidia-smi for detailed device info
    if let Ok(device_info) = detect_cuda_via_nvidia_smi() {
        if !device_info.is_empty() {
            let device_count = device_info.len() as u32;
            tracing::debug!("CUDA detected via nvidia-smi: {} devices", device_count);
            return Ok(Some((device_count, Some(device_info))));
        }
    }

    // Method 2: Check for /dev/nvidia* devices on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(Some(count)) = detect_cuda_via_dev_nodes() {
            tracing::debug!("CUDA detected via /dev/nvidia* nodes: {} devices", count);
            return Ok(Some((count, None)));
        }
    }

    // Method 3: Check for CUDA environment variables
    if let Ok(Some(count)) = detect_cuda_via_env_vars() {
        tracing::debug!("CUDA detected via environment variables: {} devices", count);
        return Ok(Some((count, None)));
    }

    tracing::debug!("No CUDA devices detected");
    Ok(None)
}

/// Detect CUDA devices using nvidia-smi
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn detect_cuda_via_nvidia_smi() -> Result<Vec<CudaDeviceInfo>> {
    // Query nvidia-smi for device information
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=index,name,compute_cap,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
        .map_err(|e| {
            AltairError::GpuDetectionFailed(format!("Failed to execute nvidia-smi: {}", e))
        })?;

    if !output.status.success() {
        return Err(AltairError::GpuDetectionFailed(
            "nvidia-smi command failed".to_string(),
        ));
    }

    let output_str = String::from_utf8(output.stdout).map_err(|e| {
        AltairError::GpuDetectionFailed(format!("Invalid UTF-8 from nvidia-smi: {}", e))
    })?;

    let mut devices = Vec::new();
    for (line_num, line) in output_str.lines().enumerate() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 4 {
            // Parse device ID with proper error handling
            let device_id = match parts[0].parse::<u32>() {
                Ok(id) => id,
                Err(e) => {
                    tracing::warn!(
                        "Skipping device on line {}: invalid device ID '{}': {}",
                        line_num + 1,
                        parts[0],
                        e
                    );
                    continue;
                }
            };

            let name = parts[1].to_string();

            // Parse compute capability (e.g., "8.9")
            let compute_cap: Vec<&str> = parts[2].split('.').collect();
            let compute_capability = if compute_cap.len() == 2 {
                let major = match compute_cap[0].parse::<u32>() {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(
                            "Skipping device on line {}: invalid compute capability major '{}': {}",
                            line_num + 1,
                            compute_cap[0],
                            e
                        );
                        continue;
                    }
                };
                let minor = match compute_cap[1].parse::<u32>() {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(
                            "Skipping device on line {}: invalid compute capability minor '{}': {}",
                            line_num + 1,
                            compute_cap[1],
                            e
                        );
                        continue;
                    }
                };
                (major, minor)
            } else {
                tracing::warn!(
                    "Skipping device on line {}: invalid compute capability format '{}'",
                    line_num + 1,
                    parts[2]
                );
                continue;
            };

            // Parse memory in MB and convert to bytes
            let memory_mb = match parts[3].parse::<u64>() {
                Ok(mem) => mem,
                Err(e) => {
                    tracing::warn!(
                        "Skipping device on line {}: invalid memory size '{}': {}",
                        line_num + 1,
                        parts[3],
                        e
                    );
                    continue;
                }
            };

            // Validate memory is not zero
            if memory_mb == 0 {
                tracing::warn!(
                    "Skipping device '{}': reported 0MB memory",
                    name
                );
                continue;
            }

            let total_memory_bytes = memory_mb * 1024 * 1024;

            devices.push(CudaDeviceInfo {
                device_id,
                name,
                compute_capability,
                total_memory_bytes,
            });
        }
    }

    if devices.is_empty() {
        Err(AltairError::GpuDetectionFailed(
            "No CUDA devices found by nvidia-smi".to_string(),
        ))
    } else {
        Ok(devices)
    }
}

/// Detect CUDA via /dev/nvidia* device nodes (Linux only)
#[cfg(target_os = "linux")]
fn detect_cuda_via_dev_nodes() -> Result<Option<u32>> {
    // Check for /dev/nvidia0, /dev/nvidia1, etc.
    let mut count = 0;
    for i in 0..16 {
        // Check up to 16 devices
        let dev_path = format!("/dev/nvidia{}", i);
        if Path::new(&dev_path).exists() {
            count += 1;
        } else {
            break;
        }
    }

    if count > 0 {
        Ok(Some(count))
    } else {
        Ok(None)
    }
}

/// Detect CUDA via environment variables
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn detect_cuda_via_env_vars() -> Result<Option<u32>> {
    // Check for CUDA installation paths
    if std::env::var("CUDA_PATH").is_ok() || std::env::var("CUDA_HOME").is_ok() {
        // We know CUDA is installed, but we don't know device count
        // Return 1 as a conservative estimate
        return Ok(Some(1));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
fn detect_metal() -> Result<Option<(Option<String>, Option<u64>)>> {
    tracing::debug!("Attempting Metal detection");

    // Check macOS version - Metal requires 10.15+
    if let Ok(version) = get_macos_version() {
        if !is_metal_compatible_version(&version) {
            tracing::warn!("macOS version {} does not support Metal (requires 10.15+)", version);
            return Ok(None);
        }
    }

    // Try to get Metal device info using metal-rs if feature is enabled
    #[cfg(feature = "metal")]
    {
        match detect_metal_via_framework() {
            Ok(Some(info)) => return Ok(Some(info)),
            Ok(None) => return Ok(None),
            Err(e) => {
                tracing::warn!("Metal framework detection failed: {}", e);
                // Fall through to fallback detection
            }
        }
    }

    // Fallback: assume Metal is available on modern macOS
    tracing::debug!("Using fallback Metal detection");
    Ok(Some((Some("Metal GPU".to_string()), None)))
}

/// Get macOS version string
#[cfg(target_os = "macos")]
fn get_macos_version() -> Result<String> {
    let output = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .map_err(|e| AltairError::GpuDetectionFailed(format!("Failed to get macOS version: {}", e)))?;

    if !output.status.success() {
        return Err(AltairError::GpuDetectionFailed(
            "sw_vers command failed".to_string(),
        ));
    }

    String::from_utf8(output.stdout)
        .map(|s| s.trim().to_string())
        .map_err(|e| AltairError::GpuDetectionFailed(format!("Invalid UTF-8 from sw_vers: {}", e)))
}

/// Check if macOS version supports Metal (10.15+)
#[cfg(target_os = "macos")]
fn is_metal_compatible_version(version: &str) -> bool {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.is_empty() {
        return false;
    }

    // Parse major version
    if let Ok(major) = parts[0].parse::<u32>() {
        // macOS 11+ (Big Sur and later) always support Metal
        if major >= 11 {
            return true;
        }

        // macOS 10.15+ (Catalina and later) support Metal
        if major == 10 && parts.len() > 1 {
            if let Ok(minor) = parts[1].parse::<u32>() {
                return minor >= 15;
            }
        }
    }

    false
}

/// Detect Metal using metal-rs framework (when feature is enabled)
#[cfg(all(target_os = "macos", feature = "metal"))]
fn detect_metal_via_framework() -> Result<Option<(Option<String>, Option<u64>)>> {
    // This will be implemented when metal-rs dependency is added
    // For now, return None to fall back to simple detection
    tracing::debug!("Metal framework detection not yet implemented");
    Ok(None)
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn detect_cuda() -> Result<Option<(u32, Option<Vec<CudaDeviceInfo>>)>> {
    Ok(None)
}

#[cfg(not(target_os = "macos"))]
fn detect_metal() -> Result<Option<(Option<String>, Option<u64>)>> {
    Ok(None)
}

/// Initialize GPU backend with validation
pub fn init_gpu_backend(backend: GpuBackend) -> Result<()> {
    match backend {
        GpuBackend::Cuda { device_count, ref devices } => {
            tracing::info!("Initializing CUDA backend with {} device(s)", device_count);

            // Pre-initialization validation
            validate_cuda_environment()?;

            // Log device information
            if let Some(devs) = devices {
                for dev in devs {
                    tracing::info!(
                        "  CUDA Device {}: {} (Compute {}.{}, {}GB RAM)",
                        dev.device_id,
                        dev.name,
                        dev.compute_capability.0,
                        dev.compute_capability.1,
                        dev.total_memory_bytes / (1024 * 1024 * 1024)
                    );

                    // Warn if compute capability is too low
                    if dev.compute_capability.0 < 6 {
                        tracing::warn!(
                            "Device {} has compute capability {}.{}, which may not be supported",
                            dev.device_id,
                            dev.compute_capability.0,
                            dev.compute_capability.1
                        );
                    }
                }
            }

            // Note: Actual CUDA initialization happens in LlamaBackend::init()
            // which is called in LlamaCppBackend::new() (backend.rs:43)
            tracing::info!("✓ CUDA environment validated and ready");
            Ok(())
        }
        GpuBackend::Metal { ref device_name, ref device_memory_gb } => {
            tracing::info!("Initializing Metal backend");

            // Pre-initialization validation
            validate_metal_environment()?;

            // Log device information
            if let Some(name) = device_name {
                tracing::info!("  Metal Device: {}", name);
            }
            if let Some(mem_gb) = device_memory_gb {
                tracing::info!("  Metal Memory: {}GB", mem_gb);
            }

            // Note: Actual Metal initialization happens in LlamaBackend::init()
            tracing::info!("✓ Metal environment validated and ready");
            Ok(())
        }
        GpuBackend::None => {
            tracing::info!("Running on CPU (no GPU acceleration)");
            Ok(())
        }
    }
}

/// Validate CUDA environment before initialization
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn validate_cuda_environment() -> Result<()> {
    tracing::debug!("Validating CUDA environment");

    // Check if nvidia-smi is available (indicates drivers are installed)
    if std::process::Command::new("nvidia-smi")
        .arg("--version")
        .output()
        .is_err()
    {
        return Err(AltairError::CudaNotAvailable(
            "nvidia-smi not found. Please install NVIDIA drivers".to_string(),
        ));
    }

    // Check for CUDA_VISIBLE_DEVICES (if set, validate it)
    if let Ok(visible_devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
        if visible_devices.is_empty() {
            tracing::warn!("CUDA_VISIBLE_DEVICES is set but empty - no CUDA devices will be used");
        } else {
            tracing::debug!("CUDA_VISIBLE_DEVICES: {}", visible_devices);
        }
    }

    tracing::debug!("CUDA environment validation passed");
    Ok(())
}

/// Validate Metal environment before initialization
#[cfg(target_os = "macos")]
fn validate_metal_environment() -> Result<()> {
    tracing::debug!("Validating Metal environment");

    // Check macOS version
    if let Ok(version) = get_macos_version() {
        if !is_metal_compatible_version(&version) {
            return Err(AltairError::MetalNotAvailable(format!(
                "macOS version {} does not support Metal. Requires macOS 10.15+",
                version
            )));
        }
        tracing::debug!("macOS version {} supports Metal", version);
    }

    tracing::debug!("Metal environment validation passed");
    Ok(())
}

// Stub implementations for non-CUDA platforms
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn validate_cuda_environment() -> Result<()> {
    Err(AltairError::CudaNotAvailable(
        "CUDA not supported on this platform".to_string(),
    ))
}

// Stub implementation for non-macOS platforms
#[cfg(not(target_os = "macos"))]
fn validate_metal_environment() -> Result<()> {
    Err(AltairError::MetalNotAvailable(
        "Metal not supported on this platform".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        let backend = detect_gpu_backend();
        println!("Detected GPU backend: {}", backend);
        // This test just ensures detection doesn't crash

        // Verify it returns one of the valid backend types
        match backend {
            GpuBackend::Cuda { device_count, .. } => {
                assert!(device_count > 0, "CUDA device count should be > 0");
            }
            GpuBackend::Metal { .. } => {
                // Metal detected, all good
            }
            GpuBackend::None => {
                // CPU fallback, all good
            }
        }
    }

    #[test]
    fn test_cpu_backend_init() {
        // CPU backend should always initialize successfully
        let result = init_gpu_backend(GpuBackend::None);
        assert!(result.is_ok(), "CPU backend init should succeed");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_version_check() {
        // This should work on any macOS system
        let version_result = get_macos_version();
        if let Ok(version) = version_result {
            println!("macOS version: {}", version);
            // Just verify we can parse it
            assert!(!version.is_empty());
        }
    }

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    #[test]
    fn test_cuda_detection_methods() {
        // Test individual CUDA detection methods
        // These may fail if CUDA is not installed, which is fine

        // Test nvidia-smi method
        let nvidia_smi_result = detect_cuda_via_nvidia_smi();
        match nvidia_smi_result {
            Ok(devices) => println!("nvidia-smi detected {} devices", devices.len()),
            Err(e) => println!("nvidia-smi detection failed (expected if no CUDA): {}", e),
        }

        // Test env var method
        let env_result = detect_cuda_via_env_vars();
        match env_result {
            Ok(Some(count)) => println!("Env vars indicate {} CUDA devices", count),
            Ok(None) => println!("No CUDA detected via env vars"),
            Err(e) => println!("Env var detection failed: {}", e),
        }
    }
}
