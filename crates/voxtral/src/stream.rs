//! Streaming scheduling helpers (prompt construction and final flush padding).

#[derive(Debug, Clone, Copy)]
pub struct StreamPromptConfig {
    pub left_pad_tokens: usize,
    pub delay_tokens: usize,
    pub eos_extra_tokens: usize,
}

impl Default for StreamPromptConfig {
    fn default() -> Self {
        Self {
            left_pad_tokens: 32,
            delay_tokens: 6, // 480ms @ 12.5Hz
            eos_extra_tokens: 10,
        }
    }
}

impl StreamPromptConfig {
    pub fn right_pad_tokens(self) -> usize {
        self.delay_tokens + 1 + self.eos_extra_tokens
    }
}

pub fn build_prompt_ids(bos_id: u32, streaming_pad_id: u32, cfg: StreamPromptConfig) -> Vec<u32> {
    let mut prompt = Vec::with_capacity(1 + cfg.left_pad_tokens + cfg.delay_tokens);
    prompt.push(bos_id);
    prompt.extend(std::iter::repeat_n(
        streaming_pad_id,
        cfg.left_pad_tokens + cfg.delay_tokens,
    ));
    prompt
}

pub fn decode_ready(total_adapter_tokens: usize, prompt_len: usize) -> bool {
    total_adapter_tokens >= prompt_len
}

/// Number of generated text tokens possible for current adapter token budget.
///
/// Offline schedule equivalent:
/// - first generated token at position `L-1`
/// - then one token per position `L..N-1`
///
///   => `N - L + 1` when `N >= L`, else `0`
pub fn decodable_token_budget(total_adapter_tokens: usize, prompt_len: usize) -> usize {
    if total_adapter_tokens < prompt_len {
        0
    } else {
        total_adapter_tokens - prompt_len + 1
    }
}

/// Compute zero samples to append when finishing the stream.
///
/// Mirrors the reference strategy:
/// 1. Align to `samples_per_token` boundary
/// 2. Add `(delay + 1 + eos_extra)` token steps of zero audio.
pub fn finish_zero_padding_samples(
    consumed_samples: usize,
    samples_per_token: usize,
    cfg: StreamPromptConfig,
) -> usize {
    debug_assert!(samples_per_token > 0);
    let rem = consumed_samples % samples_per_token;
    let align = if rem == 0 { 0 } else { samples_per_token - rem };
    align + cfg.right_pad_tokens() * samples_per_token
}

#[cfg(test)]
mod tests {
    use super::{
        StreamPromptConfig, build_prompt_ids, decodable_token_budget, decode_ready,
        finish_zero_padding_samples,
    };

    #[test]
    fn prompt_and_budget_formulas_match_reference_schedule() {
        let cfg = StreamPromptConfig::default();
        let prompt = build_prompt_ids(1, 32, cfg);
        assert_eq!(prompt.len(), 1 + 32 + 6);
        assert_eq!(prompt[0], 1);
        assert!(prompt[1..].iter().all(|&id| id == 32));

        assert!(!decode_ready(prompt.len() - 1, prompt.len()));
        assert!(decode_ready(prompt.len(), prompt.len()));
        assert_eq!(decodable_token_budget(prompt.len() - 1, prompt.len()), 0);
        assert_eq!(decodable_token_budget(prompt.len(), prompt.len()), 1);
        assert_eq!(decodable_token_budget(prompt.len() + 9, prompt.len()), 10);
    }

    #[test]
    fn finish_padding_matches_alignment_plus_right_budget() {
        let cfg = StreamPromptConfig::default();
        // 3.5 tokens consumed with 1280 samples/token.
        let consumed = 3 * 1280 + 640;
        let pad = finish_zero_padding_samples(consumed, 1280, cfg);

        // Align 640 samples + (6+1+10)*1280
        assert_eq!(pad, 640 + 17 * 1280);
    }
}
