//! Minimal decoder layer step (single-token incremental forward).

use anyhow::Result;

use crate::kv::RollingKvCache;
use crate::math::{rms_norm_rows, rope_interleaved_inplace, silu_inplace};
use crate::ops::{AttentionShape, add_inplace, attention_gqa_step, linear};

#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
}

impl DecoderConfig {
    pub fn validate(self) -> Result<()> {
        anyhow::ensure!(self.dim > 0, "dim must be > 0");
        anyhow::ensure!(self.hidden_dim > 0, "hidden_dim must be > 0");
        anyhow::ensure!(self.n_heads > 0, "n_heads must be > 0");
        anyhow::ensure!(self.n_kv_heads > 0, "n_kv_heads must be > 0");
        anyhow::ensure!(self.head_dim > 0, "head_dim must be > 0");
        anyhow::ensure!(
            self.n_heads.is_multiple_of(self.n_kv_heads),
            "n_heads must be divisible by n_kv_heads"
        );
        anyhow::ensure!(
            self.n_heads * self.head_dim >= self.dim,
            "n_heads * head_dim must cover dim"
        );
        anyhow::ensure!(
            self.sliding_window > 0,
            "sliding_window must be > 0 for rolling attention cache"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DecoderLayerWeights {
    pub attention_norm: Vec<f32>, // [dim]
    pub wq: Vec<f32>,             // [n_heads*head_dim, dim]
    pub wk: Vec<f32>,             // [n_kv_heads*head_dim, dim]
    pub wv: Vec<f32>,             // [n_kv_heads*head_dim, dim]
    pub wo: Vec<f32>,             // [dim, n_heads*head_dim]
    pub ffn_norm: Vec<f32>,       // [dim]
    pub w1: Vec<f32>,             // [hidden_dim, dim]
    pub w2: Vec<f32>,             // [dim, hidden_dim]
    pub w3: Vec<f32>,             // [hidden_dim, dim]
}

impl DecoderLayerWeights {
    pub fn validate(&self, cfg: DecoderConfig) -> Result<()> {
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        anyhow::ensure!(
            self.attention_norm.len() == cfg.dim,
            "attention_norm shape mismatch"
        );
        anyhow::ensure!(self.wq.len() == q_dim * cfg.dim, "wq shape mismatch");
        anyhow::ensure!(self.wk.len() == kv_dim * cfg.dim, "wk shape mismatch");
        anyhow::ensure!(self.wv.len() == kv_dim * cfg.dim, "wv shape mismatch");
        anyhow::ensure!(self.wo.len() == cfg.dim * q_dim, "wo shape mismatch");
        anyhow::ensure!(self.ffn_norm.len() == cfg.dim, "ffn_norm shape mismatch");
        anyhow::ensure!(
            self.w1.len() == cfg.hidden_dim * cfg.dim,
            "w1 shape mismatch"
        );
        anyhow::ensure!(
            self.w2.len() == cfg.dim * cfg.hidden_dim,
            "w2 shape mismatch"
        );
        anyhow::ensure!(
            self.w3.len() == cfg.hidden_dim * cfg.dim,
            "w3 shape mismatch"
        );
        Ok(())
    }
}

/// Run one decoder layer step for one token (`hidden` at position `pos`).
pub fn decoder_layer_forward_step(
    cfg: DecoderConfig,
    layer_idx: usize,
    weights: &DecoderLayerWeights,
    cache: &mut RollingKvCache,
    hidden: &[f32], // [dim]
    pos: usize,
) -> Result<Vec<f32>> {
    cfg.validate()?;
    weights.validate(cfg)?;
    anyhow::ensure!(hidden.len() == cfg.dim, "hidden size mismatch");

    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // 1) Attention branch.
    let mut x_norm = vec![0.0f32; cfg.dim];
    rms_norm_rows(
        &mut x_norm,
        hidden,
        &weights.attention_norm,
        cfg.dim,
        cfg.norm_eps,
    );

    let mut q = linear(&x_norm, 1, cfg.dim, &weights.wq, q_dim, None);
    let mut k = linear(&x_norm, 1, cfg.dim, &weights.wk, kv_dim, None);
    let v = linear(&x_norm, 1, cfg.dim, &weights.wv, kv_dim, None);

    rope_interleaved_inplace(&mut q, 1, cfg.n_heads, cfg.head_dim, pos, cfg.rope_theta);
    rope_interleaved_inplace(&mut k, 1, cfg.n_kv_heads, cfg.head_dim, pos, cfg.rope_theta);

    cache.append_layer(layer_idx, &k, &v, 1);
    let (k_cache, v_cache) = cache.layer_tensors(layer_idx);
    let seq_len = cache.layer_len_tokens(layer_idx);

    let attn = attention_gqa_step(
        &q,
        k_cache,
        v_cache,
        AttentionShape {
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
            seq_len,
            n_kv_heads: cfg.n_kv_heads,
            sliding_window: cfg.sliding_window,
        },
    );
    let attn_proj = linear(&attn, 1, q_dim, &weights.wo, cfg.dim, None);

    let mut h = hidden.to_vec();
    add_inplace(&mut h, &attn_proj);

    // 2) FFN branch.
    let mut h_norm = vec![0.0f32; cfg.dim];
    rms_norm_rows(&mut h_norm, &h, &weights.ffn_norm, cfg.dim, cfg.norm_eps);

    let mut gate = linear(&h_norm, 1, cfg.dim, &weights.w1, cfg.hidden_dim, None);
    let up = linear(&h_norm, 1, cfg.dim, &weights.w3, cfg.hidden_dim, None);
    silu_inplace(&mut gate);
    for (g, u) in gate.iter_mut().zip(up.iter().copied()) {
        *g *= u;
    }
    let down = linear(&gate, 1, cfg.hidden_dim, &weights.w2, cfg.dim, None);
    add_inplace(&mut h, &down);

    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::{DecoderConfig, DecoderLayerWeights, decoder_layer_forward_step};
    use crate::kv::RollingKvCache;

    fn fake_weights(cfg: DecoderConfig) -> DecoderLayerWeights {
        let mut seed = 7u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (((seed >> 8) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * 0.05
        };

        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        DecoderLayerWeights {
            attention_norm: (0..cfg.dim).map(|_| next()).collect(),
            wq: (0..(q_dim * cfg.dim)).map(|_| next()).collect(),
            wk: (0..(kv_dim * cfg.dim)).map(|_| next()).collect(),
            wv: (0..(kv_dim * cfg.dim)).map(|_| next()).collect(),
            wo: (0..(cfg.dim * q_dim)).map(|_| next()).collect(),
            ffn_norm: (0..cfg.dim).map(|_| next()).collect(),
            w1: (0..(cfg.hidden_dim * cfg.dim)).map(|_| next()).collect(),
            w2: (0..(cfg.dim * cfg.hidden_dim)).map(|_| next()).collect(),
            w3: (0..(cfg.hidden_dim * cfg.dim)).map(|_| next()).collect(),
        }
    }

    #[test]
    fn decoder_step_grows_and_compacts_cache() {
        let cfg = DecoderConfig {
            dim: 8,
            hidden_dim: 16,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 4,
            sliding_window: 4,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };
        let w = fake_weights(cfg);
        let mut cache = RollingKvCache::new(1, cfg.n_kv_heads, cfg.head_dim, cfg.sliding_window);

        let mut hidden = vec![0.0f32; cfg.dim];
        for pos in 0..6 {
            for (i, h) in hidden.iter_mut().enumerate() {
                *h = (pos as f32) * 0.1 + (i as f32) * 0.01;
            }
            let out =
                decoder_layer_forward_step(cfg, 0, &w, &mut cache, &hidden, pos).expect("step");
            assert_eq!(out.len(), cfg.dim);
            assert!(out.iter().all(|v| v.is_finite()));
            hidden = out;
        }

        // Window is 4 and we appended 6 tokens => 2 dropped.
        assert_eq!(cache.layer_len_tokens(0), 4);
        assert_eq!(cache.layer_pos_offset(0), 2);
    }
}
