//! Audio utilities.
//!
//! v0.1 scope:
//! - mono f32 samples
//! - linear resampling (for matching the reference implementation)
//! - minimal WAV parser (16-bit PCM)

use std::collections::VecDeque;

/// Linearly resample `input` from `src_hz` to `dst_hz`.
///
/// This matches the simple linear interpolation approach used by the C reference.
#[must_use]
pub fn resample_linear_mono_f32(input: &[f32], src_hz: u32, dst_hz: u32) -> Vec<f32> {
    if src_hz == dst_hz || input.is_empty() {
        return input.to_vec();
    }

    let new_n_u64 = (input.len() as u64) * u64::from(dst_hz) / u64::from(src_hz);
    let new_n = usize::try_from(new_n_u64).expect("resampled length overflowed usize");
    let mut out = vec![0.0f32; new_n];

    for (i, y) in out.iter_mut().enumerate() {
        let src_pos = (i as f32) * (src_hz as f32) / (dst_hz as f32);
        let idx = src_pos.floor() as usize;
        let frac = src_pos - (idx as f32);

        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        *y = a * (1.0 - frac) + b * frac;
    }

    out
}

/// A simple streaming linear resampler that preserves phase across chunks.
///
/// This is used for microphone capture where audio arrives in small buffers.
#[derive(Debug, Clone)]
pub struct StreamingResampler {
    src_hz: u32,
    dst_hz: u32,
    step: f64,
    pos: f64,
    input_index: u64,
    prev: f32,
    has_prev: bool,
}

impl StreamingResampler {
    #[must_use]
    pub fn new(src_hz: u32, dst_hz: u32) -> Self {
        let step = (src_hz as f64) / (dst_hz as f64);
        Self {
            src_hz,
            dst_hz,
            step,
            pos: 0.0,
            input_index: 0,
            prev: 0.0,
            has_prev: false,
        }
    }

    #[must_use]
    pub fn src_hz(&self) -> u32 {
        self.src_hz
    }

    #[must_use]
    pub fn dst_hz(&self) -> u32 {
        self.dst_hz
    }

    /// Process a chunk of mono `f32` samples and return resampled output at `dst_hz`.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() || self.src_hz == self.dst_hz {
            self.input_index += input.len() as u64;
            if let Some(&last) = input.last() {
                self.prev = last;
                self.has_prev = true;
            }
            return input.to_vec();
        }

        let start_index = self.input_index;
        let end_index = start_index + (input.len() as u64);
        let mut out = Vec::new();

        // Ensure we have a previous sample for interpolation at the beginning.
        if !self.has_prev {
            self.prev = input[0];
            self.has_prev = true;
        }

        // Need idx+1 to exist in this chunk to interpolate.
        while self.pos + 1.0 < (end_index as f64) {
            let idx_f = self.pos.floor();
            let idx = idx_f as i64;
            let frac = (self.pos - idx_f) as f32;

            let a = if idx < (start_index as i64) {
                self.prev
            } else {
                input[(idx as u64 - start_index) as usize]
            };

            let b_idx = idx + 1;
            let b = if b_idx < (start_index as i64) {
                self.prev
            } else if (b_idx as u64) < end_index {
                input[(b_idx as u64 - start_index) as usize]
            } else {
                break;
            };

            let y = a * (1.0 - frac) + b * frac;
            out.push(y);
            self.pos += self.step;
        }

        self.input_index = end_index;
        self.prev = *input.last().unwrap_or(&self.prev);
        out
    }

    /// Inform the resampler that `n` input samples were dropped (not processed).
    ///
    /// This is useful for real-time backpressure where we discard old audio to
    /// keep latency bounded. We advance the internal timebase and reset the
    /// interpolation state to avoid smearing across the discontinuity.
    pub fn skip_input_samples(&mut self, n: u64) {
        if n == 0 {
            return;
        }
        self.input_index = self.input_index.saturating_add(n);
        self.pos += n as f64;
        self.has_prev = false;
        self.prev = 0.0;
    }
}

#[derive(Debug, Clone)]
pub struct WavData {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples_mono: Vec<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum WavError {
    #[error("not a valid WAV file")]
    InvalidHeader,
    #[error("unsupported WAV format (need 16-bit PCM)")]
    UnsupportedFormat,
    #[error("malformed WAV chunks")]
    MalformedChunks,
}

fn read_u16_le(p: &[u8]) -> u16 {
    u16::from_le_bytes([p[0], p[1]])
}

fn read_u32_le(p: &[u8]) -> u32 {
    u32::from_le_bytes([p[0], p[1], p[2], p[3]])
}

/// Parse WAV bytes and return mono `f32` samples at the file's sample rate.
///
/// Supports: PCM (`audio_format=1`), 16-bit, >=1 channels.
pub fn parse_wav_bytes(data: &[u8]) -> Result<WavData, WavError> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(WavError::InvalidHeader);
    }

    let mut channels: u16 = 0;
    let mut sample_rate_hz: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut audio_format: u16 = 0;

    let mut pcm_data: Option<&[u8]> = None;

    let mut p = 12usize;
    while p + 8 <= data.len() {
        let chunk_id = &data[p..p + 4];
        let chunk_size = read_u32_le(&data[p + 4..p + 8]) as usize;
        let chunk_data_start = p + 8;
        let chunk_data_end = chunk_data_start.saturating_add(chunk_size);
        if chunk_data_end > data.len() {
            break;
        }

        if chunk_id == b"fmt " && chunk_size >= 16 {
            audio_format = read_u16_le(&data[chunk_data_start..chunk_data_start + 2]);
            channels = read_u16_le(&data[chunk_data_start + 2..chunk_data_start + 4]);
            sample_rate_hz = read_u32_le(&data[chunk_data_start + 4..chunk_data_start + 8]);
            bits_per_sample = read_u16_le(&data[chunk_data_start + 14..chunk_data_start + 16]);
        } else if chunk_id == b"data" {
            pcm_data = Some(&data[chunk_data_start..chunk_data_end]);
        }

        p = chunk_data_end;
        if chunk_size & 1 == 1 {
            p = p.saturating_add(1);
        }
    }

    let Some(pcm_data) = pcm_data else {
        return Err(WavError::MalformedChunks);
    };

    if audio_format != 1 || bits_per_sample != 16 || channels < 1 {
        return Err(WavError::UnsupportedFormat);
    }

    let frame_bytes = usize::from(channels) * 2;
    let n_frames = pcm_data.len() / frame_bytes;

    let mut samples_mono = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let frame = &pcm_data[i * frame_bytes..(i + 1) * frame_bytes];
        if channels == 1 {
            let s = i16::from_le_bytes([frame[0], frame[1]]);
            samples_mono.push((s as f32) / 32768.0);
        } else {
            let mut sum = 0.0f32;
            for c in 0..channels {
                let off = (c as usize) * 2;
                let s = i16::from_le_bytes([frame[off], frame[off + 1]]);
                sum += (s as f32) / 32768.0;
            }
            samples_mono.push(sum / (channels as f32));
        }
    }

    Ok(WavData {
        sample_rate_hz,
        channels,
        samples_mono,
    })
}

/// A fixed-capacity audio ring buffer that drops the oldest samples on overflow.
#[derive(Debug)]
pub struct DropOldestRing {
    cap: usize,
    buf: VecDeque<f32>,
    dropped: u64,
}

impl DropOldestRing {
    #[must_use]
    pub fn new(cap_samples: usize) -> Self {
        Self {
            cap: cap_samples,
            buf: VecDeque::with_capacity(cap_samples.min(16_384)),
            dropped: 0,
        }
    }

    #[must_use]
    pub fn dropped_samples(&self) -> u64 {
        self.dropped
    }

    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            if self.buf.len() == self.cap {
                self.buf.pop_front();
                self.dropped += 1;
            }
            self.buf.push_back(s);
        }
    }

    /// Drain up to `max` samples into `out`.
    pub fn drain_into(&mut self, out: &mut Vec<f32>, max: usize) {
        let n = self.buf.len().min(max);
        out.clear();
        out.reserve(n);
        for _ in 0..n {
            if let Some(v) = self.buf.pop_front() {
                out.push(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_linear_identity() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = resample_linear_mono_f32(&x, 16_000, 16_000);
        assert_eq!(x, y);
    }

    #[test]
    fn resample_linear_length() {
        let x = vec![0.0f32; 16_000];
        let y = resample_linear_mono_f32(&x, 48_000, 16_000);
        assert_eq!(y.len(), 5333); // floor(16000*16000/48000)
    }

    #[test]
    fn wav_parse_smoke() {
        // Minimal 16-bit PCM mono WAV with a single zero sample (44-byte header + 2 bytes).
        let mut wav = Vec::<u8>::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36u32 + 2).to_le_bytes()); // chunk size
        wav.extend_from_slice(b"WAVE");

        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&(16u32).to_le_bytes()); // fmt chunk size
        wav.extend_from_slice(&(1u16).to_le_bytes()); // PCM
        wav.extend_from_slice(&(1u16).to_le_bytes()); // mono
        wav.extend_from_slice(&(16_000u32).to_le_bytes()); // sample rate
        wav.extend_from_slice(&(32_000u32).to_le_bytes()); // byte rate
        wav.extend_from_slice(&(2u16).to_le_bytes()); // block align
        wav.extend_from_slice(&(16u16).to_le_bytes()); // bits

        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(2u32).to_le_bytes());
        wav.extend_from_slice(&(0i16).to_le_bytes());

        let parsed = parse_wav_bytes(&wav).expect("parse wav");
        assert_eq!(parsed.sample_rate_hz, 16_000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.samples_mono.len(), 1);
        assert!((parsed.samples_mono[0]).abs() < 1e-6);
    }

    #[test]
    fn ring_drops_oldest() {
        let mut r = DropOldestRing::new(3);
        r.push(&[1.0, 2.0, 3.0]);
        r.push(&[4.0]);
        assert_eq!(r.dropped_samples(), 1);
        let mut out = Vec::new();
        r.drain_into(&mut out, 10);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }
}
