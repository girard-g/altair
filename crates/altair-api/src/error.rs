use altair_core::AltairError;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// API error wrapper for proper HTTP responses
pub struct ApiError(pub AltairError);

impl From<AltairError> for ApiError {
    fn from(err: AltairError) -> Self {
        Self(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            // Client errors (4xx)
            AltairError::ModelNotFound(ref msg) => (StatusCode::NOT_FOUND, msg.clone()),
            AltairError::ModelAlreadyLoaded(ref msg) => (StatusCode::CONFLICT, msg.clone()),
            AltairError::InvalidModelFile(ref msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AltairError::InvalidConfig(ref msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AltairError::Serialization(ref e) => {
                (StatusCode::BAD_REQUEST, format!("Invalid request format: {}", e))
            }
            AltairError::RequestTimeout => {
                (StatusCode::REQUEST_TIMEOUT, "Request timeout".to_string())
            }
            AltairError::AuthFailed => {
                (StatusCode::UNAUTHORIZED, "Authentication failed".to_string())
            }
            AltairError::RateLimitExceeded => {
                (StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded".to_string())
            }

            // Service unavailable (503) - temporary failures
            AltairError::GpuInitFailed(ref msg) => (StatusCode::SERVICE_UNAVAILABLE, msg.clone()),
            AltairError::OutOfMemory(ref msg) => (StatusCode::SERVICE_UNAVAILABLE, msg.clone()),
            AltairError::GpuDetectionFailed(ref msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, msg.clone())
            }
            AltairError::CudaNotAvailable(ref msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, msg.clone())
            }
            AltairError::MetalNotAvailable(ref msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, msg.clone())
            }
            AltairError::GpuValidationFailed(ref msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, msg.clone())
            }

            // Bad gateway (502) - upstream failures
            AltairError::NetworkError(ref msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            AltairError::DownloadFailed(ref msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            AltairError::Http(ref msg) => (StatusCode::BAD_GATEWAY, msg.clone()),

            // Internal errors (500)
            AltairError::InferenceFailed(ref msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, msg.clone())
            }
            AltairError::FileSystemError(ref msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, msg.clone())
            }
            AltairError::Io(ref e) => {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("I/O error: {}", e))
            }
            AltairError::Ffi(ref msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        (status, Json(json!({ "error": message }))).into_response()
    }
}

/// Helper to convert Result<T, AltairError> to Result<T, ApiError>
pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;
    use altair_core::AltairError;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    #[test]
    fn test_error_status_codes() {
        let err = ApiError(AltairError::ModelNotFound("test".to_string()));
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let err = ApiError(AltairError::RequestTimeout);
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);

        let err = ApiError(AltairError::AuthFailed);
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let err = ApiError(AltairError::RateLimitExceeded);
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
