//! Model parameter file (`params.json`) parsing.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralParams {
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub use_biases: bool,
    pub causal: bool,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub vocab_size: usize,
    pub tied_embeddings: bool,
    pub sliding_window: usize,
    pub multimodal: Multimodal,
    #[serde(default)]
    pub ada_rms_norm_t_cond: bool,
    #[serde(default)]
    pub ada_rms_norm_t_cond_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Multimodal {
    pub whisper_model_args: WhisperModelArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WhisperModelArgs {
    pub encoder_args: EncoderArgs,
    pub downsample_args: DownsampleArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DownsampleArgs {
    pub downsample_factor: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderArgs {
    pub audio_encoding_args: AudioEncodingArgs,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub use_biases: bool,
    pub rope_theta: f32,
    pub causal: bool,
    pub norm_eps: f32,
    pub sliding_window: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncodingArgs {
    pub sampling_rate: u32,
    pub frame_rate: f32,
    pub num_mel_bins: usize,
    pub hop_length: usize,
    pub window_size: usize,
    pub global_log_mel_max: f32,
    pub transcription_format: String,
}

impl VoxtralParams {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let params: Self = serde_json::from_str(json).context("parse params.json")?;
        params.validate()?;
        Ok(params)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let json = std::fs::read_to_string(path_ref)
            .with_context(|| format!("read {}", path_ref.display()))?;
        Self::from_json_str(&json)
    }

    pub fn audio_encoding(&self) -> &AudioEncodingArgs {
        &self
            .multimodal
            .whisper_model_args
            .encoder_args
            .audio_encoding_args
    }

    pub fn transcription_delay_tokens(&self, delay_ms: usize) -> usize {
        let frame_rate = self.audio_encoding().frame_rate;
        let delay_s = (delay_ms as f32) / 1000.0;
        (delay_s * frame_rate).round() as usize
    }

    pub fn validate(&self) -> Result<()> {
        let enc = &self.multimodal.whisper_model_args.encoder_args;
        let audio = &enc.audio_encoding_args;

        anyhow::ensure!(self.n_heads > 0, "decoder n_heads must be > 0");
        anyhow::ensure!(self.head_dim > 0, "decoder head_dim must be > 0");
        anyhow::ensure!(
            self.n_heads * self.head_dim >= self.dim,
            "decoder heads/head_dim do not cover decoder dim"
        );
        anyhow::ensure!(enc.n_heads > 0, "encoder n_heads must be > 0");
        anyhow::ensure!(enc.head_dim > 0, "encoder head_dim must be > 0");
        anyhow::ensure!(
            enc.n_heads * enc.head_dim >= enc.dim,
            "encoder heads/head_dim do not cover encoder dim"
        );
        anyhow::ensure!(audio.sampling_rate > 0, "sampling_rate must be > 0");
        anyhow::ensure!(audio.frame_rate > 0.0, "frame_rate must be > 0");
        anyhow::ensure!(audio.hop_length > 0, "hop_length must be > 0");
        anyhow::ensure!(audio.window_size > 0, "window_size must be > 0");
        anyhow::ensure!(
            self.multimodal
                .whisper_model_args
                .downsample_args
                .downsample_factor
                > 0,
            "downsample_factor must be > 0"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::VoxtralParams;

    #[test]
    fn parse_params_smoke() {
        let json = r#"
        {
          "dim": 3072,
          "n_layers": 26,
          "head_dim": 128,
          "hidden_dim": 9216,
          "n_heads": 32,
          "n_kv_heads": 8,
          "use_biases": false,
          "causal": true,
          "rope_theta": 1000000.0,
          "norm_eps": 1e-05,
          "vocab_size": 131072,
          "tied_embeddings": true,
          "sliding_window": 8192,
          "multimodal": {
            "whisper_model_args": {
              "encoder_args": {
                "audio_encoding_args": {
                  "sampling_rate": 16000,
                  "frame_rate": 12.5,
                  "num_mel_bins": 128,
                  "hop_length": 160,
                  "window_size": 400,
                  "global_log_mel_max": 1.5,
                  "transcription_format": "streaming"
                },
                "dim": 1280,
                "n_layers": 32,
                "head_dim": 64,
                "hidden_dim": 5120,
                "n_heads": 32,
                "n_kv_heads": 32,
                "use_biases": true,
                "rope_theta": 1000000.0,
                "causal": true,
                "norm_eps": 1e-05,
                "sliding_window": 750
              },
              "downsample_args": {
                "downsample_factor": 4
              }
            }
          },
          "ada_rms_norm_t_cond": true,
          "ada_rms_norm_t_cond_dim": 32
        }
        "#;
        let p = VoxtralParams::from_json_str(json).expect("params parse");
        assert_eq!(p.dim, 3072);
        assert_eq!(p.multimodal.whisper_model_args.encoder_args.dim, 1280);
        assert_eq!(p.audio_encoding().num_mel_bins, 128);
        assert_eq!(p.transcription_delay_tokens(480), 6);
        assert!(p.ada_rms_norm_t_cond);
    }
}
