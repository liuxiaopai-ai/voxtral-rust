//! Convolution stem utilities for the Voxtral encoder.
//!
//! The reference implementation uses a causal conv stem:
//! - Conv0: (mel_bins -> dim), kernel=3, stride=1
//! - Conv1: (dim -> dim), kernel=3, stride=2
//!
//! This module provides a literal implementation plus an incremental streaming
//! wrapper that preserves boundary correctness across chunks.

#[inline]
pub fn gelu_inplace(x: &mut [f32]) {
    // Matches the approximation used in the reference C implementation.
    for v in x {
        let val = *v;
        let x3 = val * val * val;
        let inner = 0.797_884_6_f32 * (val + 0.044_715_f32 * x3);
        *v = 0.5_f32 * val * (1.0_f32 + inner.tanh());
    }
}

/// Causal 1D convolution that matches the padding scheme used in vLLM's WhisperCausalConv1d.
///
/// Input and output are **column-major**:
/// - `input`: `[channels_in, length]` stored as `input[ic * length + t]`
/// - `output`: `[channels_out, out_length]` stored as `output[oc * out_length + t]`
///
/// Weights are stored as `[channels_out, channels_in, kernel]` contiguous.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_colmajor(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    channels_in: usize,
    channels_out: usize,
    length: usize,
    kernel: usize,
    stride: usize,
) {
    let padding_total = kernel - stride;
    let n_frames =
        ((length as f32) - (kernel as f32) + (padding_total as f32)) / (stride as f32) + 1.0;
    let out_length = n_frames.ceil().max(0.0) as usize;
    if out_length == 0 {
        return;
    }

    debug_assert_eq!(output.len(), channels_out * out_length);
    debug_assert_eq!(input.len(), channels_in * length);
    debug_assert_eq!(weight.len(), channels_out * channels_in * kernel);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), channels_out);
    }

    let left_pad = padding_total as isize;

    for oc in 0..channels_out {
        let b = bias.map_or(0.0, |bb| bb[oc]);
        let out_row = &mut output[oc * out_length..(oc + 1) * out_length];

        for (ol, out_elem) in out_row.iter_mut().enumerate() {
            let mut sum = b;
            let base = (ol * stride) as isize - left_pad;
            for ic in 0..channels_in {
                let in_row = &input[ic * length..(ic + 1) * length];
                let w_base = (oc * channels_in * kernel) + (ic * kernel);
                for k in 0..kernel {
                    let il = base + (k as isize);
                    if il >= 0 && (il as usize) < length {
                        sum += in_row[il as usize] * weight[w_base + k];
                    }
                }
            }
            *out_elem = sum;
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvStemWeights {
    pub conv0_weight: Vec<f32>, // [dim, mel_bins, 3]
    pub conv0_bias: Vec<f32>,   // [dim]
    pub conv1_weight: Vec<f32>, // [dim, dim, 3]
    pub conv1_bias: Vec<f32>,   // [dim]
    pub mel_bins: usize,
    pub dim: usize,
}

impl ConvStemWeights {
    /// Create random-ish weights for tests (deterministic).
    #[cfg(test)]
    fn fake(mel_bins: usize, dim: usize) -> Self {
        fn lcg(seed: &mut u32) -> f32 {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            // [-1, 1]
            let v = (*seed >> 8) as f32 / ((1u32 << 24) as f32);
            (v * 2.0) - 1.0
        }

        let mut seed = 1u32;
        let mut conv0_weight = vec![0.0f32; dim * mel_bins * 3];
        for v in &mut conv0_weight {
            *v = lcg(&mut seed) * 0.01;
        }
        let mut conv0_bias = vec![0.0f32; dim];
        for v in &mut conv0_bias {
            *v = lcg(&mut seed) * 0.01;
        }

        let mut conv1_weight = vec![0.0f32; dim * dim * 3];
        for v in &mut conv1_weight {
            *v = lcg(&mut seed) * 0.01;
        }
        let mut conv1_bias = vec![0.0f32; dim];
        for v in &mut conv1_bias {
            *v = lcg(&mut seed) * 0.01;
        }

        Self {
            conv0_weight,
            conv0_bias,
            conv1_weight,
            conv1_bias,
            mel_bins,
            dim,
        }
    }
}

/// Incremental conv stem state. Input is mel frames row-major `[n_frames, mel_bins]`.
///
/// Output is post-conv row-major `[n_pos, dim]`.
#[derive(Debug)]
pub struct ConvStemState {
    mel_bins: usize,
    dim: usize,

    // Tail buffers (kernel=3 => 2 frames).
    mel_tail: Vec<f32>,   // [mel_bins, 2] column-major
    conv0_tail: Vec<f32>, // [dim, 2] column-major (last 2 conv0 outputs)

    conv0_residual: Vec<f32>,    // [dim]
    conv0_residual_count: usize, // 0 or 1

    initialized: bool,
}

impl ConvStemState {
    #[must_use]
    pub fn new(mel_bins: usize, dim: usize) -> Self {
        Self {
            mel_bins,
            dim,
            mel_tail: vec![0.0f32; mel_bins * 2],
            conv0_tail: vec![0.0f32; dim * 2],
            conv0_residual: vec![0.0f32; dim],
            conv0_residual_count: 0,
            initialized: false,
        }
    }

    /// Process new mel frames and return new post-conv positions.
    pub fn process(&mut self, w: &ConvStemWeights, mel_new: &[f32], n_new_mel: usize) -> Vec<f32> {
        if n_new_mel == 0 {
            return Vec::new();
        }
        debug_assert_eq!(w.mel_bins, self.mel_bins);
        debug_assert_eq!(w.dim, self.dim);
        debug_assert_eq!(mel_new.len(), n_new_mel * self.mel_bins);

        let is_first = !self.initialized;

        // === Phase 1: Conv0 (mel -> dim), stride=1 ===
        let (conv0_new_colmajor, conv0_new_len) = if is_first {
            let conv_in_len = n_new_mel;
            let conv_in = transpose_row_to_col(mel_new, self.mel_bins, conv_in_len);
            let mut conv0_out = vec![0.0f32; self.dim * conv_in_len];
            causal_conv1d_colmajor(
                &mut conv0_out,
                &conv_in,
                &w.conv0_weight,
                Some(&w.conv0_bias),
                self.mel_bins,
                self.dim,
                conv_in_len,
                3,
                1,
            );
            gelu_inplace(&mut conv0_out);

            self.update_mel_tail(mel_new, n_new_mel);
            self.initialized = true;

            (conv0_out, conv_in_len)
        } else {
            let padded_len = 2 + n_new_mel;
            let mut conv_in = vec![0.0f32; self.mel_bins * padded_len];
            // prepend mel_tail (col-major) then new mel (row-major)
            for m in 0..self.mel_bins {
                conv_in[m * padded_len] = self.mel_tail[m * 2];
                conv_in[m * padded_len + 1] = self.mel_tail[m * 2 + 1];
                for f in 0..n_new_mel {
                    conv_in[m * padded_len + 2 + f] = mel_new[f * self.mel_bins + m];
                }
            }

            let mut conv0_full = vec![0.0f32; self.dim * padded_len];
            causal_conv1d_colmajor(
                &mut conv0_full,
                &conv_in,
                &w.conv0_weight,
                Some(&w.conv0_bias),
                self.mel_bins,
                self.dim,
                padded_len,
                3,
                1,
            );
            gelu_inplace(&mut conv0_full);

            // Discard first 2 columns.
            let mut conv0_new = vec![0.0f32; self.dim * n_new_mel];
            for d in 0..self.dim {
                let src = &conv0_full[d * padded_len + 2..d * padded_len + 2 + n_new_mel];
                let dst = &mut conv0_new[d * n_new_mel..(d + 1) * n_new_mel];
                dst.copy_from_slice(src);
            }

            self.update_mel_tail(mel_new, n_new_mel);

            (conv0_new, n_new_mel)
        };

        // === Phase 2: stride alignment for Conv1 (needs even number of conv0 outputs) ===
        let prev_res = self.conv0_residual_count;
        let total_avail = prev_res + conv0_new_len;
        let new_res = total_avail & 1;
        let feed_from_new = conv0_new_len - new_res;
        let feed_total = prev_res + feed_from_new; // always even

        if feed_total == 0 {
            if new_res == 1 {
                // save last conv0 column
                for d in 0..self.dim {
                    self.conv0_residual[d] =
                        conv0_new_colmajor[d * conv0_new_len + (conv0_new_len - 1)];
                }
            }
            self.conv0_residual_count = new_res;
            return Vec::new();
        }

        let mut feed = vec![0.0f32; self.dim * feed_total];
        let mut fpos = 0usize;
        if prev_res == 1 {
            for d in 0..self.dim {
                feed[d * feed_total] = self.conv0_residual[d];
            }
            fpos = 1;
        }

        for d in 0..self.dim {
            let src = &conv0_new_colmajor[d * conv0_new_len..d * conv0_new_len + feed_from_new];
            let dst = &mut feed[d * feed_total + fpos..d * feed_total + fpos + feed_from_new];
            dst.copy_from_slice(src);
        }

        if new_res == 1 {
            for d in 0..self.dim {
                self.conv0_residual[d] =
                    conv0_new_colmajor[d * conv0_new_len + (conv0_new_len - 1)];
            }
        }
        self.conv0_residual_count = new_res;

        // === Phase 3: Conv1 (dim -> dim), stride=2 ===
        let (conv1_in, conv1_in_len, conv1_discard) = if is_first {
            // First chunk: conv1 uses zero left-pad (implicit in causal conv).
            // Save tail for the next chunk before moving `feed`.
            for d in 0..self.dim {
                self.conv0_tail[d * 2] = feed[d * feed_total + (feed_total - 2)];
                self.conv0_tail[d * 2 + 1] = feed[d * feed_total + (feed_total - 1)];
            }
            (feed, feed_total, 0usize)
        } else {
            let conv1_in_len = 2 + feed_total;
            let mut conv1_in = vec![0.0f32; self.dim * conv1_in_len];
            for d in 0..self.dim {
                conv1_in[d * conv1_in_len] = self.conv0_tail[d * 2];
                conv1_in[d * conv1_in_len + 1] = self.conv0_tail[d * 2 + 1];
                let src = &feed[d * feed_total..(d + 1) * feed_total];
                let dst = &mut conv1_in[d * conv1_in_len + 2..d * conv1_in_len + 2 + feed_total];
                dst.copy_from_slice(src);
            }
            // Update conv0_tail from last 2 of the (even-aligned) feed, for the next chunk.
            // Important: this must happen *after* we have used the previous tail as context.
            for d in 0..self.dim {
                self.conv0_tail[d * 2] = feed[d * feed_total + (feed_total - 2)];
                self.conv0_tail[d * 2 + 1] = feed[d * feed_total + (feed_total - 1)];
            }
            (conv1_in, conv1_in_len, 1usize)
        };

        debug_assert!(conv1_in_len % 2 == 0, "conv1 input length should be even");
        let conv1_out_len = conv1_in_len / 2;
        let mut conv1_out = vec![0.0f32; self.dim * conv1_out_len];
        causal_conv1d_colmajor(
            &mut conv1_out,
            &conv1_in,
            &w.conv1_weight,
            Some(&w.conv1_bias),
            self.dim,
            self.dim,
            conv1_in_len,
            3,
            2,
        );
        gelu_inplace(&mut conv1_out);

        let result_len = conv1_out_len.saturating_sub(conv1_discard);
        if result_len == 0 {
            return Vec::new();
        }

        // Transpose [dim, result_len] col-major -> row-major [result_len, dim].
        let mut result = vec![0.0f32; result_len * self.dim];
        for si in 0..result_len {
            let src_col = conv1_discard + si;
            for d in 0..self.dim {
                result[si * self.dim + d] = conv1_out[d * conv1_out_len + src_col];
            }
        }
        result
    }

    fn update_mel_tail(&mut self, mel_new: &[f32], n_new_mel: usize) {
        // Store last 2 mel frames as col-major [mel_bins,2].
        let ts = n_new_mel.saturating_sub(2);
        let tc = n_new_mel.min(2);
        self.mel_tail.fill(0.0);
        for f in 0..tc {
            for m in 0..self.mel_bins {
                let dst_col = 2 - tc + f;
                self.mel_tail[m * 2 + dst_col] = mel_new[(ts + f) * self.mel_bins + m];
            }
        }
    }
}

fn transpose_row_to_col(input_row: &[f32], channels: usize, length: usize) -> Vec<f32> {
    debug_assert_eq!(input_row.len(), channels * length);
    let mut out = vec![0.0f32; channels * length];
    for t in 0..length {
        for c in 0..channels {
            out[c * length + t] = input_row[t * channels + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv0_incremental_collect(
        w: &ConvStemWeights,
        mel: &[f32],
        mel_bins: usize,
        dim: usize,
        chunk_sizes: &[usize],
    ) -> Vec<f32> {
        let total_frames = mel.len() / mel_bins;
        let mut out = vec![0.0f32; dim * total_frames]; // col-major [dim, frames]

        let mut mel_tail = vec![0.0f32; mel_bins * 2];
        let mut initialized = false;

        let mut cursor = 0usize;
        for &cs in chunk_sizes {
            let mel_new = &mel[cursor * mel_bins..(cursor + cs) * mel_bins];

            let conv0_new_col = if !initialized {
                let conv_in = transpose_row_to_col(mel_new, mel_bins, cs);
                let mut conv0_out = vec![0.0f32; dim * cs];
                causal_conv1d_colmajor(
                    &mut conv0_out,
                    &conv_in,
                    &w.conv0_weight,
                    Some(&w.conv0_bias),
                    mel_bins,
                    dim,
                    cs,
                    3,
                    1,
                );
                gelu_inplace(&mut conv0_out);

                // update mel_tail
                let ts = cs.saturating_sub(2);
                let tc = cs.min(2);
                mel_tail.fill(0.0);
                for f in 0..tc {
                    for m in 0..mel_bins {
                        let dst_col = 2 - tc + f;
                        mel_tail[m * 2 + dst_col] = mel_new[(ts + f) * mel_bins + m];
                    }
                }
                initialized = true;

                conv0_out
            } else {
                let padded_len = 2 + cs;
                let mut conv_in = vec![0.0f32; mel_bins * padded_len];
                for m in 0..mel_bins {
                    conv_in[m * padded_len] = mel_tail[m * 2];
                    conv_in[m * padded_len + 1] = mel_tail[m * 2 + 1];
                    for f in 0..cs {
                        conv_in[m * padded_len + 2 + f] = mel_new[f * mel_bins + m];
                    }
                }

                let mut conv0_full = vec![0.0f32; dim * padded_len];
                causal_conv1d_colmajor(
                    &mut conv0_full,
                    &conv_in,
                    &w.conv0_weight,
                    Some(&w.conv0_bias),
                    mel_bins,
                    dim,
                    padded_len,
                    3,
                    1,
                );
                gelu_inplace(&mut conv0_full);

                // Discard first 2 columns.
                let mut conv0_new = vec![0.0f32; dim * cs];
                for d in 0..dim {
                    let src = &conv0_full[d * padded_len + 2..d * padded_len + 2 + cs];
                    let dst = &mut conv0_new[d * cs..(d + 1) * cs];
                    dst.copy_from_slice(src);
                }

                // update mel_tail
                let ts = cs.saturating_sub(2);
                let tc = cs.min(2);
                mel_tail.fill(0.0);
                for f in 0..tc {
                    for m in 0..mel_bins {
                        let dst_col = 2 - tc + f;
                        mel_tail[m * 2 + dst_col] = mel_new[(ts + f) * mel_bins + m];
                    }
                }

                conv0_new
            };

            // Copy into final output at [cursor..cursor+cs].
            for d in 0..dim {
                let src = &conv0_new_col[d * cs..(d + 1) * cs];
                let dst = &mut out[d * total_frames + cursor..d * total_frames + cursor + cs];
                dst.copy_from_slice(src);
            }

            cursor += cs;
        }

        assert_eq!(cursor, total_frames);
        out
    }

    #[test]
    fn incremental_conv_matches_offline_even_length() {
        // Use tiny dims so the naive conv is fast.
        let mel_bins = 8usize;
        let dim = 16usize;
        let total_frames = 40usize; // even => no residual at end

        let w = ConvStemWeights::fake(mel_bins, dim);
        let mut mel = vec![0.0f32; total_frames * mel_bins];
        // Deterministic pseudo-random mel.
        let mut seed = 123u32;
        for v in &mut mel {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = (seed >> 8) as f32 / ((1u32 << 24) as f32);
            *v = (r * 2.0) - 1.0;
        }

        // Offline conv stem on the whole sequence.
        let mel_col = transpose_row_to_col(&mel, mel_bins, total_frames);
        let mut conv0 = vec![0.0f32; dim * total_frames];
        causal_conv1d_colmajor(
            &mut conv0,
            &mel_col,
            &w.conv0_weight,
            Some(&w.conv0_bias),
            mel_bins,
            dim,
            total_frames,
            3,
            1,
        );
        gelu_inplace(&mut conv0);

        // Conv0 incremental should match offline exactly.
        let chunk_sizes = [7usize, 3, 11, 5, 14]; // sums to 40
        let conv0_inc = conv0_incremental_collect(&w, &mel, mel_bins, dim, &chunk_sizes);
        assert_eq!(conv0_inc.len(), conv0.len());
        for (i, (a, b)) in conv0_inc.iter().zip(conv0.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "conv0 mismatch at {i}");
        }

        let conv1_in_len = total_frames; // conv0 length == mel length
        let conv1_out_len = conv1_in_len / 2;
        let mut conv1 = vec![0.0f32; dim * conv1_out_len];
        causal_conv1d_colmajor(
            &mut conv1,
            &conv0,
            &w.conv1_weight,
            Some(&w.conv1_bias),
            dim,
            dim,
            conv1_in_len,
            3,
            2,
        );
        gelu_inplace(&mut conv1);

        // Transpose offline output to row-major.
        let mut offline = vec![0.0f32; conv1_out_len * dim];
        for t in 0..conv1_out_len {
            for d in 0..dim {
                offline[t * dim + d] = conv1[d * conv1_out_len + t];
            }
        }

        // Incremental conv stem in chunks.
        let mut st = ConvStemState::new(mel_bins, dim);
        let mut got = Vec::<f32>::new();
        let mut cursor = 0usize;
        for &cs in &chunk_sizes {
            let part = &mel[cursor * mel_bins..(cursor + cs) * mel_bins];
            let out = st.process(&w, part, cs);
            got.extend_from_slice(&out);
            cursor += cs;
        }
        assert_eq!(cursor, total_frames);

        assert_eq!(got.len(), offline.len());
        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        for (i, (a, b)) in got.iter().zip(offline.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }

        // NOTE: we expect very close results, but allow a small tolerance since this is
        // a floating-point pipeline and we may change low-level implementations later.
        assert!(
            max_diff < 1e-4,
            "max diff too large: idx={max_idx} got={} offline={} diff={max_diff}",
            got[max_idx],
            offline[max_idx],
        );
    }

    #[test]
    fn default_mel_bins_constant_matches_model() {
        // Sanity: keep MEL_BINS constant aligned with model.
        assert_eq!(crate::constants::MEL_BINS, 128);
    }
}
