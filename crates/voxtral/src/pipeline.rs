//! Streaming pipeline state machine (model-agnostic planner).
//!
//! This module intentionally does not run neural network compute yet. It tracks:
//! - prompt construction
//! - adapter-token progress
//! - decodable token budget
//! - end-of-stream right-padding plan

use crate::stream::{
    StreamPromptConfig, build_prompt_ids, decodable_token_budget, finish_zero_padding_samples,
};

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub bos_id: u32,
    pub streaming_pad_id: u32,
    pub samples_per_token: usize,
    pub prompt: StreamPromptConfig,
}

impl PipelineConfig {
    pub fn prompt_ids(&self) -> Vec<u32> {
        build_prompt_ids(self.bos_id, self.streaming_pad_id, self.prompt)
    }
}

#[derive(Debug, Clone)]
pub struct PipelineState {
    cfg: PipelineConfig,
    prompt_ids: Vec<u32>,
    adapter_tokens_total: usize,
    generated_tokens: usize,
    finished: bool,
}

impl PipelineState {
    #[must_use]
    pub fn new(cfg: PipelineConfig) -> Self {
        let prompt_ids = cfg.prompt_ids();
        Self {
            cfg,
            prompt_ids,
            adapter_tokens_total: 0,
            generated_tokens: 0,
            finished: false,
        }
    }

    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    pub fn adapter_tokens_total(&self) -> usize {
        self.adapter_tokens_total
    }

    pub fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    pub fn push_adapter_tokens(&mut self, n_tokens: usize) {
        if n_tokens == 0 {
            return;
        }
        self.adapter_tokens_total += n_tokens;
    }

    pub fn decodable_now(&self) -> usize {
        let total_budget = decodable_token_budget(self.adapter_tokens_total, self.prompt_ids.len());
        total_budget.saturating_sub(self.generated_tokens)
    }

    pub fn mark_decoded(&mut self, n_tokens: usize) {
        debug_assert!(n_tokens <= self.decodable_now());
        self.generated_tokens += n_tokens;
    }

    pub fn finish_padding_samples(&mut self, consumed_audio_samples: usize) -> usize {
        self.finished = true;
        finish_zero_padding_samples(
            consumed_audio_samples,
            self.cfg.samples_per_token,
            self.cfg.prompt,
        )
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use crate::constants::RAW_AUDIO_SAMPLES_PER_TOKEN;
    use crate::stream::StreamPromptConfig;

    use super::{PipelineConfig, PipelineState};

    #[test]
    fn planner_tracks_budget_across_chunks() {
        let cfg = PipelineConfig {
            bos_id: 1,
            streaming_pad_id: 32,
            samples_per_token: RAW_AUDIO_SAMPLES_PER_TOKEN,
            prompt: StreamPromptConfig::default(),
        };
        let mut st = PipelineState::new(cfg);

        assert_eq!(st.prompt_ids().len(), 39);
        assert_eq!(st.decodable_now(), 0);

        // Not enough adapter tokens yet.
        st.push_adapter_tokens(20);
        assert_eq!(st.decodable_now(), 0);

        // Reach prompt length => first generated token becomes available.
        st.push_adapter_tokens(19);
        assert_eq!(st.adapter_tokens_total(), 39);
        assert_eq!(st.decodable_now(), 1);

        st.mark_decoded(1);
        assert_eq!(st.generated_tokens(), 1);
        assert_eq!(st.decodable_now(), 0);

        // 10 more adapter positions => 10 more decodable tokens.
        st.push_adapter_tokens(10);
        assert_eq!(st.decodable_now(), 10);
        st.mark_decoded(7);
        assert_eq!(st.decodable_now(), 3);
    }

    #[test]
    fn planner_finish_padding_formula() {
        let cfg = PipelineConfig {
            bos_id: 1,
            streaming_pad_id: 32,
            samples_per_token: RAW_AUDIO_SAMPLES_PER_TOKEN,
            prompt: StreamPromptConfig::default(),
        };
        let mut st = PipelineState::new(cfg);
        let pad = st.finish_padding_samples(3 * 1280 + 640);
        assert!(st.is_finished());
        assert_eq!(pad, 640 + 17 * 1280);
    }
}
