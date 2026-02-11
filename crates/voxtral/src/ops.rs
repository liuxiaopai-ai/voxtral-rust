//! Inference ops shared by encoder/decoder implementations.

use crate::math::softmax_inplace;

/// Linear layer: `y = x * W^T + b`.
///
/// Shapes:
/// - `input`: `[n_rows, in_dim]`
/// - `weight`: `[out_dim, in_dim]`
/// - output: `[n_rows, out_dim]`
pub fn linear(
    input: &[f32],
    n_rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), n_rows * in_dim);
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_dim);
    }

    let mut out = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        let x = &input[r * in_dim..(r + 1) * in_dim];
        let y = &mut out[r * out_dim..(r + 1) * out_dim];
        for o in 0..out_dim {
            let w = &weight[o * in_dim..(o + 1) * in_dim];
            let mut sum = bias.map_or(0.0, |b| b[o]);
            for i in 0..in_dim {
                sum += x[i] * w[i];
            }
            y[o] = sum;
        }
    }
    out
}

pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter().copied()) {
        *d += s;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    pub n_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub n_kv_heads: usize,
    /// `0` means "no window limit".
    pub sliding_window: usize,
}

/// Grouped-query attention for one decode step.
///
/// Shapes:
/// - `query`: `[n_heads, head_dim]`
/// - `keys`: `[seq_len, n_kv_heads, head_dim]`
/// - `values`: `[seq_len, n_kv_heads, head_dim]`
///
/// Returns: `[n_heads, head_dim]`.
pub fn attention_gqa_step(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: AttentionShape,
) -> Vec<f32> {
    let AttentionShape {
        n_heads,
        head_dim,
        seq_len,
        n_kv_heads,
        sliding_window,
    } = shape;

    debug_assert_eq!(query.len(), n_heads * head_dim);
    debug_assert_eq!(keys.len(), seq_len * n_kv_heads * head_dim);
    debug_assert_eq!(values.len(), seq_len * n_kv_heads * head_dim);
    debug_assert!(n_heads > 0 && n_kv_heads > 0);
    debug_assert_eq!(n_heads % n_kv_heads, 0);

    let kv_group = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_heads * head_dim];
    let mut scores = vec![0.0f32; seq_len];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let start = if sliding_window == 0 {
        0
    } else {
        seq_len.saturating_sub(sliding_window)
    };

    for h in 0..n_heads {
        let q = &query[h * head_dim..(h + 1) * head_dim];
        let kv_h = h / kv_group;

        for (t, score) in scores.iter_mut().enumerate().take(seq_len) {
            if t < start {
                *score = f32::NEG_INFINITY;
                continue;
            }
            let k_base = (t * n_kv_heads + kv_h) * head_dim;
            let k = &keys[k_base..k_base + head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q[i] * k[i];
            }
            *score = dot * scale;
        }

        softmax_inplace(&mut scores);

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        out_h.fill(0.0);
        for (t, &a) in scores.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let v_base = (t * n_kv_heads + kv_h) * head_dim;
            let v = &values[v_base..v_base + head_dim];
            for i in 0..head_dim {
                out_h[i] += a * v[i];
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{AttentionShape, attention_gqa_step, linear};

    #[test]
    fn linear_smoke() {
        // x: [2,3], W: [2,3]
        let x = [1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0];
        let w = [1.0f32, 0.0, -1.0, 2.0, 1.0, 0.0];
        let b = [0.5f32, -1.0];
        let y = linear(&x, 2, 3, &w, 2, Some(&b));
        // row0: [1-3+0.5, 2+2-1] = [-1.5, 3.0]
        // row1: [-1-1+0.5, -2+0-1] = [-1.5, -3.0]
        assert!((y[0] + 1.5).abs() < 1e-6);
        assert!((y[1] - 3.0).abs() < 1e-6);
        assert!((y[2] + 1.5).abs() < 1e-6);
        assert!((y[3] + 3.0).abs() < 1e-6);
    }

    #[test]
    fn gqa_matches_expanded_kv_reference() {
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 3usize;
        let seq_len = 5usize;

        let mut seed = 123u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0
        };

        let mut q = vec![0.0f32; n_heads * head_dim];
        let mut k = vec![0.0f32; seq_len * n_kv_heads * head_dim];
        let mut v = vec![0.0f32; seq_len * n_kv_heads * head_dim];
        for x in &mut q {
            *x = rand();
        }
        for x in &mut k {
            *x = rand();
        }
        for x in &mut v {
            *x = rand();
        }

        let got = attention_gqa_step(
            &q,
            &k,
            &v,
            AttentionShape {
                n_heads,
                head_dim,
                seq_len,
                n_kv_heads,
                sliding_window: 0,
            },
        );

        // Reference: expand KV heads to full MHA heads explicitly.
        let repeat = n_heads / n_kv_heads;
        let mut ref_out = vec![0.0f32; n_heads * head_dim];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for h in 0..n_heads {
            let kv_h = h / repeat;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0.0f32; seq_len];
            for (t, score) in scores.iter_mut().enumerate().take(seq_len) {
                let kb = (t * n_kv_heads + kv_h) * head_dim;
                let kh = &k[kb..kb + head_dim];
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += qh[i] * kh[i];
                }
                *score = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }

            let out_h = &mut ref_out[h * head_dim..(h + 1) * head_dim];
            for (t, score) in scores.iter().copied().enumerate().take(seq_len) {
                let vb = (t * n_kv_heads + kv_h) * head_dim;
                let vh = &v[vb..vb + head_dim];
                for i in 0..head_dim {
                    out_h[i] += score * vh[i];
                }
            }
        }

        let mut max_diff = 0.0f32;
        for (a, b) in got.iter().zip(ref_out.iter()) {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(max_diff < 1e-5, "max diff {max_diff}");
    }
}
