//! High-level model asset loading from a model directory.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::params::VoxtralParams;
use crate::tokenizer::TekkenTokenizer;
use crate::weights::WeightStore;

#[derive(Debug)]
pub struct ModelMetadata {
    pub params: VoxtralParams,
    pub tokenizer: TekkenTokenizer,
}

#[derive(Debug)]
pub struct ModelBundle {
    pub metadata: ModelMetadata,
    pub weights: WeightStore,
}

fn params_path(dir: &Path) -> PathBuf {
    dir.join("params.json")
}

fn tokenizer_path(dir: &Path) -> PathBuf {
    dir.join("tekken.json")
}

fn weights_path(dir: &Path) -> PathBuf {
    dir.join("consolidated.safetensors")
}

impl ModelMetadata {
    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let params = VoxtralParams::from_path(params_path(dir)).context("load params.json")?;
        let tokenizer =
            TekkenTokenizer::from_path(tokenizer_path(dir)).context("load tekken.json")?;
        Ok(Self { params, tokenizer })
    }
}

impl ModelBundle {
    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let metadata = ModelMetadata::load_from_dir(dir)?;
        let weights =
            WeightStore::open(weights_path(dir)).context("load consolidated.safetensors")?;
        Ok(Self { metadata, weights })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use safetensors::tensor::{Dtype, View, serialize_to_file};

    use super::{ModelBundle, ModelMetadata};

    #[derive(Debug, Clone)]
    struct TestTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for TestTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<[u8]> {
            Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn tmp_dir() -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        p.push(format!("voxtral-model-test-{nanos}"));
        std::fs::create_dir_all(&p).expect("mkdir");
        p
    }

    fn write_fixture_model_dir(dir: &Path) {
        let params = r#"
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
              "downsample_args": { "downsample_factor": 4 }
            }
          },
          "ada_rms_norm_t_cond": true,
          "ada_rms_norm_t_cond_dim": 32
        }
        "#;
        std::fs::write(dir.join("params.json"), params).expect("write params");

        let tekken = r#"
        {
          "config": {
            "default_vocab_size": 8,
            "default_num_special_tokens": 4
          },
          "special_tokens": [
            {"rank": 1, "token_str": "<s>", "is_control": true},
            {"rank": 2, "token_str": "</s>", "is_control": true},
            {"rank": 3, "token_str": "[STREAMING_PAD]", "is_control": true}
          ],
          "vocab": [
            {"rank": 0, "token_bytes": "QQ=="}
          ],
          "audio": {
            "sampling_rate": 16000,
            "frame_rate": 12.5,
            "transcription_delay_ms": 480,
            "streaming_n_left_pad_tokens": 32
          }
        }
        "#;
        std::fs::write(dir.join("tekken.json"), tekken).expect("write tekken");

        let tensor = TestTensor {
            dtype: Dtype::F32,
            shape: vec![2],
            data: [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
        };
        serialize_to_file(
            vec![(
                "mm_streams_embeddings.embedding_module.tok_embeddings.weight",
                tensor,
            )],
            &None,
            &dir.join("consolidated.safetensors"),
        )
        .expect("write safetensors");
    }

    #[test]
    fn loads_metadata_from_model_dir() {
        let dir = tmp_dir();
        write_fixture_model_dir(&dir);

        let meta = ModelMetadata::load_from_dir(&dir).expect("metadata");
        assert_eq!(meta.params.dim, 3072);
        assert_eq!(meta.tokenizer.streaming_pad_id(), Some(3));
        assert_eq!(
            meta.tokenizer.default_prompt_ids().expect("prompt").len(),
            39
        );

        std::fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn loads_full_bundle_with_weights() {
        let dir = tmp_dir();
        write_fixture_model_dir(&dir);

        let bundle = ModelBundle::load_from_dir(&dir).expect("bundle");
        let names = bundle.weights.names().expect("names");
        assert_eq!(names.len(), 1);
        assert_eq!(
            names[0],
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight"
        );

        std::fs::remove_dir_all(dir).expect("cleanup");
    }
}
