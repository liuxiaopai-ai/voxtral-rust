//! Tekken tokenizer (`tekken.json`) decode support.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct SpecialToken {
    pub id: u32,
    pub text: String,
    pub is_control: bool,
}

#[derive(Debug, Clone)]
pub struct TekkenAudioConfig {
    pub sampling_rate: u32,
    pub frame_rate: f32,
    pub transcription_delay_ms: usize,
    pub streaming_n_left_pad_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct TekkenTokenizer {
    default_vocab_size: usize,
    special_token_count: usize,
    token_bytes_by_id: Vec<Option<Vec<u8>>>,
    special_tokens: Vec<SpecialToken>,
    special_lookup: HashMap<String, u32>,
    audio: Option<TekkenAudioConfig>,
}

#[derive(Debug, Deserialize)]
struct TekkenFile {
    config: TekkenConfig,
    vocab: Vec<TekkenVocabEntry>,
    special_tokens: Vec<TekkenSpecialEntry>,
    #[serde(default)]
    audio: Option<TekkenAudioRaw>,
}

#[derive(Debug, Deserialize)]
struct TekkenConfig {
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct TekkenVocabEntry {
    rank: usize,
    token_bytes: String,
}

#[derive(Debug, Deserialize)]
struct TekkenSpecialEntry {
    rank: usize,
    token_str: String,
    is_control: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct TekkenAudioRaw {
    sampling_rate: u32,
    frame_rate: f32,
    transcription_delay_ms: usize,
    streaming_n_left_pad_tokens: usize,
}

impl TekkenTokenizer {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let file: TekkenFile = serde_json::from_str(json).context("parse tekken.json")?;
        anyhow::ensure!(
            file.config.default_vocab_size > 0,
            "default_vocab_size must be > 0"
        );

        let mut special_slots = vec![None; file.config.default_num_special_tokens];
        let mut special_lookup = HashMap::<String, u32>::new();
        for entry in file.special_tokens {
            if entry.rank >= special_slots.len() {
                continue;
            }
            let id = entry.rank as u32;
            special_lookup.insert(entry.token_str.clone(), id);
            special_slots[entry.rank] = Some(SpecialToken {
                id,
                text: entry.token_str,
                is_control: entry.is_control,
            });
        }

        let mut special_tokens = Vec::with_capacity(special_slots.len());
        for (rank, slot) in special_slots.into_iter().enumerate() {
            special_tokens.push(slot.unwrap_or(SpecialToken {
                id: rank as u32,
                text: format!("<missing-special-{rank}>"),
                is_control: true,
            }));
        }

        let mut token_bytes_by_id = vec![None; file.config.default_vocab_size];
        for entry in file.vocab {
            let token_id = file.config.default_num_special_tokens + entry.rank;
            if token_id >= token_bytes_by_id.len() {
                continue;
            }
            let bytes = STANDARD
                .decode(entry.token_bytes.as_bytes())
                .with_context(|| format!("decode base64 for vocab rank {}", entry.rank))?;
            token_bytes_by_id[token_id] = Some(bytes);
        }

        let audio = file.audio.map(|a| TekkenAudioConfig {
            sampling_rate: a.sampling_rate,
            frame_rate: a.frame_rate,
            transcription_delay_ms: a.transcription_delay_ms,
            streaming_n_left_pad_tokens: a.streaming_n_left_pad_tokens,
        });

        Ok(Self {
            default_vocab_size: file.config.default_vocab_size,
            special_token_count: file.config.default_num_special_tokens,
            token_bytes_by_id,
            special_tokens,
            special_lookup,
            audio,
        })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let json = std::fs::read_to_string(path_ref)
            .with_context(|| format!("read {}", path_ref.display()))?;
        Self::from_json_str(&json)
    }

    pub fn default_vocab_size(&self) -> usize {
        self.default_vocab_size
    }

    pub fn special_token_count(&self) -> usize {
        self.special_token_count
    }

    pub fn special_id(&self, text: &str) -> Option<u32> {
        self.special_lookup.get(text).copied()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.special_id("<s>")
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.special_id("</s>")
    }

    pub fn streaming_pad_id(&self) -> Option<u32> {
        self.special_id("[STREAMING_PAD]")
    }

    pub fn audio(&self) -> Option<&TekkenAudioConfig> {
        self.audio.as_ref()
    }

    pub fn transcription_delay_tokens(&self) -> Option<usize> {
        let audio = self.audio()?;
        let seconds = (audio.transcription_delay_ms as f32) / 1000.0;
        Some((seconds * audio.frame_rate).round() as usize)
    }

    pub fn default_prompt_ids(&self) -> Option<Vec<u32>> {
        let bos = self.bos_id()?;
        let pad = self.streaming_pad_id()?;
        let audio = self.audio()?;
        let delay = self.transcription_delay_tokens()?;
        let len = audio.streaming_n_left_pad_tokens + delay;

        let mut prompt = Vec::with_capacity(1 + len);
        prompt.push(bos);
        prompt.extend(std::iter::repeat_n(pad, len));
        Some(prompt)
    }

    pub fn decode_token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        let idx = usize::try_from(token_id).ok()?;
        if idx >= self.default_vocab_size {
            return None;
        }

        if idx < self.special_token_count {
            let st = self.special_tokens.get(idx)?;
            if st.is_control {
                None
            } else {
                Some(st.text.as_bytes())
            }
        } else {
            self.token_bytes_by_id.get(idx)?.as_deref()
        }
    }

    pub fn decode_to_utf8_lossy(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::<u8>::new();
        for &id in token_ids {
            if let Some(piece) = self.decode_token_bytes(id) {
                bytes.extend_from_slice(piece);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::TekkenTokenizer;

    #[test]
    fn tokenizer_decodes_basic_tokens() {
        let json = r#"
        {
          "config": {
            "default_vocab_size": 16,
            "default_num_special_tokens": 4
          },
          "special_tokens": [
            {"rank": 0, "token_str": "<unk>", "is_control": true},
            {"rank": 1, "token_str": "<s>", "is_control": true},
            {"rank": 2, "token_str": "</s>", "is_control": true},
            {"rank": 3, "token_str": "[STREAMING_PAD]", "is_control": true}
          ],
          "vocab": [
            {"rank": 0, "token_bytes": "QQ=="},
            {"rank": 1, "token_bytes": "Qg=="},
            {"rank": 2, "token_bytes": "Qw=="}
          ],
          "audio": {
            "sampling_rate": 16000,
            "frame_rate": 12.5,
            "transcription_delay_ms": 480,
            "streaming_n_left_pad_tokens": 32
          }
        }
        "#;
        let t = TekkenTokenizer::from_json_str(json).expect("tokenizer parse");
        assert_eq!(t.default_vocab_size(), 16);
        assert_eq!(t.special_token_count(), 4);
        assert_eq!(t.bos_id(), Some(1));
        assert_eq!(t.eos_id(), Some(2));
        assert_eq!(t.streaming_pad_id(), Some(3));
        assert_eq!(t.transcription_delay_tokens(), Some(6));

        // IDs 4/5/6 map to ranks 0/1/2 regular vocab (A/B/C).
        let decoded = t.decode_to_utf8_lossy(&[1, 4, 5, 6, 2]);
        assert_eq!(decoded, "ABC");
    }

    #[test]
    fn default_prompt_has_expected_length() {
        let json = r#"
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
          "vocab": [],
          "audio": {
            "sampling_rate": 16000,
            "frame_rate": 12.5,
            "transcription_delay_ms": 480,
            "streaming_n_left_pad_tokens": 32
          }
        }
        "#;
        let t = TekkenTokenizer::from_json_str(json).expect("tokenizer parse");
        let prompt = t.default_prompt_ids().expect("prompt");
        assert_eq!(prompt.len(), 1 + 32 + 6);
        assert_eq!(prompt[0], 1);
        assert!(prompt[1..].iter().all(|&id| id == 3));
    }
}
