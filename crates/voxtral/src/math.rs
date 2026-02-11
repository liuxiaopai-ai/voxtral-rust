//! Core math kernels for inference.

#[inline]
pub fn silu_inplace(x: &mut [f32]) {
    for v in x {
        let t = *v;
        *v = t / (1.0 + (-t).exp());
    }
}

pub fn rms_norm_rows(output: &mut [f32], input: &[f32], weight: &[f32], dim: usize, eps: f32) {
    debug_assert!(dim > 0);
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(weight.len(), dim);
    debug_assert_eq!(input.len() % dim, 0);

    for (in_row, out_row) in input.chunks_exact(dim).zip(output.chunks_exact_mut(dim)) {
        let mut sq_sum = 0.0f32;
        for &v in in_row {
            sq_sum += v * v;
        }
        let inv_rms = 1.0 / ((sq_sum / (dim as f32) + eps).sqrt());
        for i in 0..dim {
            out_row[i] = in_row[i] * inv_rms * weight[i];
        }
    }
}

/// Apply interleaved RoPE (GPT-J style) in-place on `[seq_len, n_heads, head_dim]` row-major data.
pub fn rope_interleaved_inplace(
    data: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    theta: f32,
) {
    debug_assert_eq!(data.len(), seq_len * n_heads * head_dim);
    debug_assert_eq!(head_dim % 2, 0);

    let row_stride = n_heads * head_dim;
    for t in 0..seq_len {
        let pos = (pos_offset + t) as f32;
        let row = &mut data[t * row_stride..(t + 1) * row_stride];

        for h in 0..n_heads {
            let head = &mut row[h * head_dim..(h + 1) * head_dim];
            for pair in 0..(head_dim / 2) {
                let i0 = 2 * pair;
                let i1 = i0 + 1;

                // same inv_freq formula used by transformer RoPE:
                // inv_freq[pair] = theta^(-(2*pair)/head_dim)
                let inv_freq = theta.powf(-(i0 as f32) / (head_dim as f32));
                let angle = pos * inv_freq;
                let (sin, cos) = angle.sin_cos();

                let x0 = head[i0];
                let x1 = head[i1];
                head[i0] = x0 * cos - x1 * sin;
                head[i1] = x0 * sin + x1 * cos;
            }
        }
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_v = x[0];
    for &v in &x[1..] {
        if v > max_v {
            max_v = v;
        }
    }

    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{rms_norm_rows, rope_interleaved_inplace, silu_inplace, softmax_inplace};

    #[test]
    fn rms_norm_rows_smoke() {
        let input = [3.0f32, 4.0, 0.0, 1.0];
        let weight = [1.0f32, 1.0];
        let mut out = [0.0f32; 4];
        rms_norm_rows(&mut out, &input, &weight, 2, 1e-5);

        // [3,4] norm -> [0.8485, 1.1314] approximately
        assert!((out[0] - 0.8485).abs() < 1e-3);
        assert!((out[1] - 1.1314).abs() < 1e-3);
    }

    #[test]
    fn rope_preserves_pairwise_norm() {
        let seq = 3usize;
        let heads = 2usize;
        let head_dim = 4usize;
        let mut x = vec![0.0f32; seq * heads * head_dim];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 + 0.1;
        }
        let before = x.clone();
        rope_interleaved_inplace(&mut x, seq, heads, head_dim, 7, 10_000.0);

        for t in 0..seq {
            for h in 0..heads {
                let base = (t * heads + h) * head_dim;
                for p in 0..(head_dim / 2) {
                    let i0 = base + 2 * p;
                    let i1 = i0 + 1;
                    let n0 = before[i0] * before[i0] + before[i1] * before[i1];
                    let n1 = x[i0] * x[i0] + x[i1] * x[i1];
                    assert!(
                        (n0 - n1).abs() < 1e-4,
                        "pair norm drift at t={t} h={h} p={p}"
                    );
                }
            }
        }
    }

    #[test]
    fn silu_and_softmax_smoke() {
        let mut v = [0.0f32, 1.0, -1.0];
        silu_inplace(&mut v);
        assert!(v[1] > v[0]);
        assert!(v[2] < v[0]);

        let mut s = [1.0f32, 2.0, 3.0];
        softmax_inplace(&mut s);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }
}
