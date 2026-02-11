//! Incremental log-mel spectrogram (Voxtral/Whisper-style).
//!
//! This is intentionally implemented in a very literal way to match the
//! reference behavior:
//! - Periodic Hann window
//! - Direct DFT (no FFT) for exactness
//! - Slaney-style mel filter bank
//! - log10 clamp and scaling

use crate::constants::{
    HOP_LENGTH, LOG_MEL_MAX, MEL_BINS, N_FFT, N_FREQ, SAMPLE_RATE_HZ, WINDOW_SIZE,
};

#[inline]
fn hertz_to_mel(freq: f32) -> f32 {
    // Slaney-style mel scale (matches mistral_common).
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    // Precomputed: ln(6.4)/27.
    const LOGSTEP: f32 = 0.068_751_78;

    let mut mels = 3.0 * freq / 200.0;
    if freq >= MIN_LOG_HZ {
        mels = MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() * LOGSTEP;
    }
    mels
}

#[inline]
fn mel_to_hertz(mels: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOGSTEP: f32 = 0.068_751_78;

    let mut freq = 200.0 * mels / 3.0;
    if mels >= MIN_LOG_MEL {
        freq = MIN_LOG_HZ * (LOGSTEP * (mels - MIN_LOG_MEL)).exp();
    }
    freq
}

fn build_mel_filters() -> Vec<f32> {
    let mut fft_freqs = vec![0.0f32; N_FREQ];
    for (f, v) in fft_freqs.iter_mut().enumerate() {
        *v = (f as f32) * (SAMPLE_RATE_HZ as f32) / (N_FFT as f32);
    }

    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel((SAMPLE_RATE_HZ as f32) / 2.0);

    let mut filter_freqs = vec![0.0f32; MEL_BINS + 2];
    for (i, v) in filter_freqs.iter_mut().enumerate() {
        let mel = mel_min + (mel_max - mel_min) * (i as f32) / ((MEL_BINS + 1) as f32);
        *v = mel_to_hertz(mel);
    }

    let mut filter_diff = vec![0.0f32; MEL_BINS + 1];
    for (i, v) in filter_diff.iter_mut().enumerate() {
        *v = filter_freqs[i + 1] - filter_freqs[i];
        if *v == 0.0 {
            *v = 1e-6;
        }
    }

    let mut filters = vec![0.0f32; MEL_BINS * N_FREQ];
    for m in 0..MEL_BINS {
        let denom = filter_freqs[m + 2] - filter_freqs[m];
        let enorm = 2.0 / denom;
        for f in 0..N_FREQ {
            let down = (fft_freqs[f] - filter_freqs[m]) / filter_diff[m];
            let up = (filter_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            let mut val = down.min(up);
            if val < 0.0 {
                val = 0.0;
            }
            filters[m * N_FREQ + f] = val * enorm;
        }
    }

    filters
}

fn build_dft_tables() -> (Vec<f32>, Vec<f32>) {
    let mut cos_t = vec![0.0f32; N_FREQ * N_FFT];
    let mut sin_t = vec![0.0f32; N_FREQ * N_FFT];

    for k in 0..N_FREQ {
        for n in 0..N_FFT {
            let angle = 2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / (N_FFT as f32);
            cos_t[k * N_FFT + n] = angle.cos();
            sin_t[k * N_FFT + n] = angle.sin();
        }
    }

    (cos_t, sin_t)
}

fn build_hann_window() -> [f32; WINDOW_SIZE] {
    let mut w = [0.0f32; WINDOW_SIZE];
    for (i, wi) in w.iter_mut().enumerate() {
        // Periodic Hann: 0.5*(1-cos(2*pi*i/N))
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (WINDOW_SIZE as f32);
        *wi = 0.5 * (1.0 - angle.cos());
    }
    w
}

/// Incremental mel spectrogram state.
///
/// The internal sample buffer starts with `left_pad = 200 + left_pad_samples` zeros:
/// - 200 corresponds to STFT center padding with `n_fft/2` samples
/// - `left_pad_samples` is the upstream audio left padding used by Voxtral streaming
#[derive(Debug)]
pub struct MelCtx {
    mel_filters: Vec<f32>, // [MEL_BINS * N_FREQ]
    dft_cos: Vec<f32>,     // [N_FREQ * N_FFT]
    dft_sin: Vec<f32>,     // [N_FREQ * N_FFT]
    window: [f32; WINDOW_SIZE],

    samples: Vec<f32>,
    mel: Vec<f32>,
    n_mel_frames: usize,

    left_pad: usize,
    finished: bool,
}

impl MelCtx {
    #[must_use]
    pub fn new(left_pad_samples: usize) -> Self {
        let mel_filters = build_mel_filters();
        let (dft_cos, dft_sin) = build_dft_tables();
        let window = build_hann_window();

        let left_pad = (N_FFT / 2) + left_pad_samples;
        let mut samples = vec![0.0f32; left_pad + SAMPLE_RATE_HZ as usize]; // ~1s slack
        samples.truncate(left_pad); // logical length

        Self {
            mel_filters,
            dft_cos,
            dft_sin,
            window,
            samples,
            mel: Vec::new(),
            n_mel_frames: 0,
            left_pad,
            finished: false,
        }
    }

    /// Total left padding in samples.
    #[must_use]
    pub fn left_pad_samples(&self) -> usize {
        self.left_pad
    }

    /// Append new samples and compute all mel frames that fit.
    pub fn feed(&mut self, input: &[f32]) -> usize {
        if input.is_empty() {
            return 0;
        }

        self.samples.extend_from_slice(input);
        self.compute_available()
    }

    /// Finish the stream by appending `right_pad_samples` zeros, then applying a 200-sample
    /// right reflect pad, then computing remaining frames and dropping the final frame.
    pub fn finish(&mut self, right_pad_samples: usize) {
        if self.finished {
            return;
        }

        if right_pad_samples > 0 {
            self.samples
                .extend(std::iter::repeat_n(0.0f32, right_pad_samples));
        }

        let reflect_len = N_FFT / 2;
        let real_end = self.samples.len().saturating_sub(right_pad_samples);

        for i in 0..reflect_len {
            let src = real_end as isize - 2 - (i as isize);
            let v = if src >= 0 {
                self.samples.get(src as usize).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            self.samples.push(v);
        }

        self.compute_available();

        // vLLM convention: drop last frame (magnitudes = stft[..., :-1]).
        if self.n_mel_frames > 0 {
            self.n_mel_frames -= 1;
            self.mel.truncate(self.n_mel_frames * MEL_BINS);
        }

        self.finished = true;
    }

    /// Return mel data as a flat slice `[frames * MEL_BINS]` and the frame count.
    #[must_use]
    pub fn data(&self) -> (&[f32], usize) {
        (&self.mel, self.n_mel_frames)
    }

    fn compute_available(&mut self) -> usize {
        let mut new_frames = 0usize;

        let mut windowed = [0.0f32; N_FFT];
        let mut power = [0.0f32; N_FREQ];

        loop {
            let t = self.n_mel_frames;
            let start = t * HOP_LENGTH;
            let end = start + WINDOW_SIZE;
            if end > self.samples.len() {
                break;
            }

            // Ensure mel capacity.
            if self.mel.len() < (t + 1) * MEL_BINS {
                self.mel.resize((t + 1) * MEL_BINS, 0.0);
            }

            for (i, out) in windowed.iter_mut().enumerate() {
                *out = self.samples[start + i] * self.window[i];
            }

            // Direct DFT -> power spectrum.
            for (k, pk) in power.iter_mut().enumerate() {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let cos_row = &self.dft_cos[k * N_FFT..(k + 1) * N_FFT];
                let sin_row = &self.dft_sin[k * N_FFT..(k + 1) * N_FFT];
                for n in 0..N_FFT {
                    re += windowed[n] * cos_row[n];
                    im += windowed[n] * sin_row[n];
                }
                *pk = re * re + im * im;
            }

            // Apply mel filters + log10 clamp.
            let mel_row = &mut self.mel[t * MEL_BINS..(t + 1) * MEL_BINS];
            for (m, out_m) in mel_row.iter_mut().enumerate() {
                let filt = &self.mel_filters[m * N_FREQ..(m + 1) * N_FREQ];
                let mut sum = 0.0f32;
                for k in 0..N_FREQ {
                    sum += filt[k] * power[k];
                }
                if sum < 1e-10 {
                    sum = 1e-10;
                }
                let mut val = sum.log10();
                let min_val = LOG_MEL_MAX - 8.0;
                if val < min_val {
                    val = min_val;
                }
                *out_m = (val + 4.0) / 4.0;
            }

            self.n_mel_frames += 1;
            new_frames += 1;
        }

        new_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_has_constant_value() {
        let mut ctx = MelCtx::new(0);
        assert_eq!(ctx.left_pad_samples(), 200);

        let n_new = ctx.feed(&vec![0.0f32; 400]);
        assert_eq!(n_new, 2);

        let (mel, frames) = ctx.data();
        assert_eq!(frames, 2);
        assert_eq!(mel.len(), frames * MEL_BINS);

        // For silence: log10(1e-10)=-10, clamped to (1.5-8)=-6.5, then (val+4)/4 = -0.625.
        let expect = -0.625f32;
        for &v in mel {
            assert!((v - expect).abs() < 1e-6, "got {v}, expected {expect}");
        }
    }

    #[test]
    fn finish_drops_last_frame() {
        let mut ctx = MelCtx::new(0);
        let fed = 1600usize;
        ctx.feed(&vec![0.0f32; fed]);

        // After finish(): add 200-sample right reflect, then drop the last frame.
        let total_samples = ctx.left_pad_samples() + fed + (N_FFT / 2);
        let frames_total = ((total_samples - WINDOW_SIZE) / HOP_LENGTH) + 1;
        let expect_after = frames_total.saturating_sub(1);

        ctx.finish(0);
        let (_, after) = ctx.data();
        assert_eq!(after, expect_after);
    }
}
