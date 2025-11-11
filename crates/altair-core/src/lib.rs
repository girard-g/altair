//! Core types, traits, and utilities for Altair

pub mod config;
pub mod downloader;
pub mod error;
pub mod paths;
pub mod types;

pub use error::{AltairError, Result};
pub use paths::*;
pub use types::*;
