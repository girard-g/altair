//! Model downloader with progress tracking, resume support, and SHA256 verification

use crate::error::*;
use futures_util::StreamExt;
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

/// Status of the download operation
#[derive(Debug, Clone)]
pub enum DownloadStatus {
    /// Download is starting
    Starting,
    /// Download is in progress
    Downloading,
    /// Verifying file integrity
    Verifying,
    /// Download completed successfully
    Complete,
    /// Download failed with error
    Error(String),
}

impl ToString for DownloadStatus {
    fn to_string(&self) -> String {
        match self {
            Self::Starting => "starting".to_string(),
            Self::Downloading => "downloading".to_string(),
            Self::Verifying => "verifying".to_string(),
            Self::Complete => "success".to_string(),
            Self::Error(e) => format!("error: {}", e),
        }
    }
}

/// Progress information for download operations
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Current status
    pub status: DownloadStatus,
    /// Total bytes to download
    pub total_bytes: u64,
    /// Bytes downloaded so far
    pub downloaded_bytes: u64,
    /// Download speed in bytes per second
    pub speed_bytes_per_sec: f64,
    /// Estimated time remaining in seconds
    pub eta_seconds: Option<u64>,
    /// Download percentage (0-100)
    pub percentage: f32,
}

/// Model downloader with resume support and SHA256 verification
pub struct ModelDownloader {
    client: Client,
    models_dir: PathBuf,
}

impl ModelDownloader {
    /// Create a new model downloader
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            client: Client::new(),
            models_dir,
        }
    }

    /// Download a model from HuggingFace or direct URL
    ///
    /// Formats supported:
    /// - `TheBloke/Llama-2-7B-Chat-GGUF` (auto-detects .gguf file)
    /// - `TheBloke/Llama-2-7B-Chat-GGUF:llama-2-7b-chat.Q4_K_M.gguf` (specific file)
    /// - `https://huggingface.co/.../model.gguf` (direct URL)
    pub async fn download_model(
        &self,
        model_ref: &str,
        progress_tx: mpsc::UnboundedSender<DownloadProgress>,
    ) -> Result<PathBuf> {
        // Parse model reference
        let (url, filename) = self.parse_model_ref(model_ref).await?;

        let dest_path = self.models_dir.join(&filename);

        // Create models directory if it doesn't exist
        fs::create_dir_all(&self.models_dir)
            .await
            .map_err(|e| AltairError::FileSystemError(e.to_string()))?;

        // Check if already exists
        if dest_path.exists() {
            progress_tx.send(DownloadProgress {
                status: DownloadStatus::Complete,
                total_bytes: 0,
                downloaded_bytes: 0,
                speed_bytes_per_sec: 0.0,
                eta_seconds: None,
                percentage: 100.0,
            }).ok();
            return Ok(dest_path);
        }

        // Send starting status
        progress_tx.send(DownloadProgress {
            status: DownloadStatus::Starting,
            total_bytes: 0,
            downloaded_bytes: 0,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
            percentage: 0.0,
        }).ok();

        // Download with resume support
        self.download_with_resume(&url, &dest_path, &progress_tx).await?;

        // Verify SHA256 if available
        if let Some(hash) = self.get_sha256_hash(&url).await? {
            progress_tx.send(DownloadProgress {
                status: DownloadStatus::Verifying,
                total_bytes: 0,
                downloaded_bytes: 0,
                speed_bytes_per_sec: 0.0,
                eta_seconds: None,
                percentage: 100.0,
            }).ok();

            if !self.verify_sha256(&dest_path, &hash).await? {
                // Delete invalid file
                fs::remove_file(&dest_path).await.ok();
                return Err(AltairError::InvalidModelFile(
                    "SHA256 verification failed".to_string()
                ));
            }
        }

        // Send complete status
        progress_tx.send(DownloadProgress {
            status: DownloadStatus::Complete,
            total_bytes: 0,
            downloaded_bytes: 0,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
            percentage: 100.0,
        }).ok();

        Ok(dest_path)
    }

    /// Parse model reference into URL and filename
    async fn parse_model_ref(&self, model_ref: &str) -> Result<(String, String)> {
        if model_ref.starts_with("http://") || model_ref.starts_with("https://") {
            // Direct URL
            let filename = model_ref.split('/').last()
                .ok_or_else(|| AltairError::InvalidModelFile("Invalid URL".to_string()))?
                .to_string();
            Ok((model_ref.to_string(), filename))
        } else if model_ref.contains('/') {
            // HuggingFace format
            let (repo, file) = if model_ref.contains(':') {
                let parts: Vec<&str> = model_ref.split(':').collect();
                (parts[0], Some(parts[1]))
            } else {
                (model_ref, None)
            };

            self.resolve_huggingface_model(repo, file).await
        } else {
            Err(AltairError::InvalidModelFile(
                "Invalid model reference format".to_string()
            ))
        }
    }

    /// Resolve HuggingFace model repository to a download URL
    async fn resolve_huggingface_model(
        &self,
        repo: &str,
        filename: Option<&str>,
    ) -> Result<(String, String)> {
        // If filename specified, use it directly
        if let Some(file) = filename {
            let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, file);
            return Ok((url, file.to_string()));
        }

        // Otherwise, find first .gguf file in repo
        let api_url = format!("https://huggingface.co/api/models/{}/tree/main", repo);
        let response: serde_json::Value = self.client
            .get(&api_url)
            .send()
            .await
            .map_err(|e| AltairError::NetworkError(e.to_string()))?
            .json()
            .await
            .map_err(|e| AltairError::NetworkError(e.to_string()))?;

        // Find .gguf files
        let gguf_file = response
            .as_array()
            .ok_or_else(|| AltairError::InvalidModelFile("Invalid API response".to_string()))?
            .iter()
            .find_map(|f| {
                f["path"].as_str()
                    .filter(|p| p.ends_with(".gguf"))
                    .map(String::from)
            })
            .ok_or_else(|| AltairError::ModelNotFound("No .gguf file found in repository".to_string()))?;

        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, gguf_file);
        Ok((url, gguf_file))
    }

    /// Download file with resume support using HTTP Range headers
    async fn download_with_resume(
        &self,
        url: &str,
        dest_path: &Path,
        progress_tx: &mpsc::UnboundedSender<DownloadProgress>,
    ) -> Result<()> {
        let temp_path = dest_path.with_extension("partial");

        // Check if partial file exists
        let start_byte = if temp_path.exists() {
            fs::metadata(&temp_path).await
                .map_err(|e| AltairError::FileSystemError(e.to_string()))?
                .len()
        } else {
            0
        };

        // Build request with Range header for resume
        let mut request = self.client.get(url);
        if start_byte > 0 {
            request = request.header("Range", format!("bytes={}-", start_byte));
        }

        let response = request.send().await
            .map_err(|e| AltairError::NetworkError(e.to_string()))?;

        // Check for error status
        if !response.status().is_success() && response.status().as_u16() != 206 {
            return Err(AltairError::NetworkError(
                format!("HTTP error: {}", response.status())
            ));
        }

        let total_size = response.content_length().unwrap_or(0) + start_byte;

        // Open file for writing (append if resuming)
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&temp_path)
            .await
            .map_err(|e| AltairError::FileSystemError(e.to_string()))?;

        let mut downloaded = start_byte;
        let mut stream = response.bytes_stream();
        let start_time = std::time::Instant::now();
        let mut last_update = start_time;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AltairError::NetworkError(e.to_string()))?;
            file.write_all(&chunk).await
                .map_err(|e| AltairError::FileSystemError(e.to_string()))?;

            downloaded += chunk.len() as u64;

            // Update progress every 100ms
            let now = std::time::Instant::now();
            if now.duration_since(last_update).as_millis() >= 100 {
                let elapsed = now.duration_since(start_time).as_secs_f64();
                let speed = if elapsed > 0.0 {
                    (downloaded - start_byte) as f64 / elapsed
                } else {
                    0.0
                };
                let remaining = total_size.saturating_sub(downloaded);
                let eta = if speed > 0.0 {
                    Some((remaining as f64 / speed) as u64)
                } else {
                    None
                };

                progress_tx.send(DownloadProgress {
                    status: DownloadStatus::Downloading,
                    total_bytes: total_size,
                    downloaded_bytes: downloaded,
                    speed_bytes_per_sec: speed,
                    eta_seconds: eta,
                    percentage: if total_size > 0 {
                        (downloaded as f32 / total_size as f32) * 100.0
                    } else {
                        0.0
                    },
                }).ok();

                last_update = now;
            }
        }

        file.flush().await
            .map_err(|e| AltairError::FileSystemError(e.to_string()))?;

        // Rename temp file to final destination
        fs::rename(&temp_path, dest_path).await
            .map_err(|e| AltairError::FileSystemError(e.to_string()))?;

        Ok(())
    }

    /// Attempt to get SHA256 hash from remote server
    async fn get_sha256_hash(&self, base_url: &str) -> Result<Option<String>> {
        let sha_url = format!("{}.sha256", base_url);
        match self.client.get(&sha_url).send().await {
            Ok(response) if response.status().is_success() => {
                let hash = response.text().await
                    .map_err(|e| AltairError::NetworkError(e.to_string()))?;
                // Extract just the hash (first part before any whitespace)
                let hash = hash.trim().split_whitespace().next()
                    .unwrap_or(hash.trim())
                    .to_string();
                Ok(Some(hash))
            }
            _ => Ok(None),
        }
    }

    /// Verify SHA256 hash of downloaded file
    async fn verify_sha256(&self, file_path: &Path, expected_hash: &str) -> Result<bool> {
        let mut file = fs::File::open(file_path).await
            .map_err(|e| AltairError::FileSystemError(e.to_string()))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192];

        loop {
            let n = tokio::io::AsyncReadExt::read(&mut file, &mut buffer).await
                .map_err(|e| AltairError::FileSystemError(e.to_string()))?;
            if n == 0 { break; }
            hasher.update(&buffer[..n]);
        }

        let result = hasher.finalize();
        let computed_hash = format!("{:x}", result);

        Ok(computed_hash == expected_hash.trim().to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_status_to_string() {
        assert_eq!(DownloadStatus::Starting.to_string(), "starting");
        assert_eq!(DownloadStatus::Downloading.to_string(), "downloading");
        assert_eq!(DownloadStatus::Verifying.to_string(), "verifying");
        assert_eq!(DownloadStatus::Complete.to_string(), "success");
    }

    #[tokio::test]
    async fn test_parse_direct_url() {
        let downloader = ModelDownloader::new(PathBuf::from("/tmp/models"));
        let result = downloader.parse_model_ref("https://example.com/model.gguf").await;
        assert!(result.is_ok());
        let (url, filename) = result.unwrap();
        assert_eq!(url, "https://example.com/model.gguf");
        assert_eq!(filename, "model.gguf");
    }
}
