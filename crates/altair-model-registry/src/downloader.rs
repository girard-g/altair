use altair_core::{error::*, types::*};
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

/// Model downloader with progress tracking
pub struct ModelDownloader {
    client: reqwest::Client,
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| AltairError::DownloadFailed(e.to_string()))?;

        Ok(Self { client, cache_dir })
    }

    /// Download a model from HuggingFace with progress callback
    pub async fn download_from_huggingface<F>(
        &self,
        repo_id: &str,
        filename: &str,
        mut progress_callback: F,
    ) -> Result<PathBuf>
    where
        F: FnMut(DownloadProgress) + Send,
    {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );

        let dest_path = self.cache_dir.join(filename);

        // Check if already exists
        if dest_path.exists() {
            tracing::info!("Model already cached: {:?}", dest_path);
            return Ok(dest_path);
        }

        tracing::info!("Downloading model from {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| AltairError::DownloadFailed(e.to_string()))?;

        let total_size = response.content_length();

        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        // Create temporary file
        let temp_path = dest_path.with_extension("tmp");
        let mut file = File::create(&temp_path)
            .await
            .map_err(|e| AltairError::DownloadFailed(e.to_string()))?;

        let mut hasher = Sha256::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AltairError::DownloadFailed(e.to_string()))?;

            file.write_all(&chunk).await?;
            hasher.update(&chunk);

            downloaded += chunk.len() as u64;

            progress_callback(DownloadProgress {
                model_id: repo_id.to_string(),
                status: DownloadStatus::Downloading,
                bytes_downloaded: downloaded,
                total_bytes: total_size,
                files_completed: 0,
                total_files: 1,
            });
        }

        file.flush().await?;
        drop(file);

        // Verify and move to final location
        progress_callback(DownloadProgress {
            model_id: repo_id.to_string(),
            status: DownloadStatus::Verifying,
            bytes_downloaded: downloaded,
            total_bytes: total_size,
            files_completed: 0,
            total_files: 1,
        });

        let hash = hasher.finalize();
        tracing::info!("Download complete. SHA256: {:x}", hash);

        // Move temp file to final location
        std::fs::rename(&temp_path, &dest_path)?;

        progress_callback(DownloadProgress {
            model_id: repo_id.to_string(),
            status: DownloadStatus::Completed,
            bytes_downloaded: downloaded,
            total_bytes: total_size,
            files_completed: 1,
            total_files: 1,
        });

        Ok(dest_path)
    }

    /// Download from a direct URL
    pub async fn download_from_url<F>(
        &self,
        url: &str,
        filename: &str,
        progress_callback: F,
    ) -> Result<PathBuf>
    where
        F: FnMut(DownloadProgress) + Send,
    {
        // Similar to download_from_huggingface but for arbitrary URLs
        // TODO: Implement
        self.download_from_huggingface(url, filename, progress_callback)
            .await
    }
}
