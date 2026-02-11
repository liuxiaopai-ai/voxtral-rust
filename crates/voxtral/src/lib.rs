//! Voxtral Realtime (Rust) core library.
//!
//! This crate will provide:
//! - Audio frontend (resample + incremental log-mel)
//! - Streaming pipeline (encoder + adapter + decoder)
//! - Tokenization (Tekken decode)

pub mod audio;
pub mod constants;
pub mod conv;
pub mod decoder;
pub mod kv;
pub mod math;
pub mod mel;
pub mod model;
pub mod ops;
pub mod params;
pub mod pipeline;
pub mod stream;
pub mod tokenizer;
pub mod weights;

/// A simple placeholder error type used until we settle module boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoxtralError {
    NotImplemented(&'static str),
}

impl std::fmt::Display for VoxtralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented(what) => write!(f, "not implemented: {what}"),
        }
    }
}

impl std::error::Error for VoxtralError {}
