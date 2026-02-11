//! Model and signal-processing constants.

// Audio preprocessing (matches the reference implementation).
pub const SAMPLE_RATE_HZ: u32 = 16_000;
pub const MEL_BINS: usize = 128;
pub const HOP_LENGTH: usize = 160; // 10ms @ 16kHz
pub const WINDOW_SIZE: usize = 400; // 25ms @ 16kHz
pub const N_FFT: usize = 400;
pub const N_FREQ: usize = N_FFT / 2 + 1; // 201
pub const LOG_MEL_MAX: f32 = 1.5;

// Token-time mapping for Voxtral (12.5 Hz => 80ms/token).
pub const FRAME_RATE_HZ: f32 = 12.5;
pub const RAW_AUDIO_SAMPLES_PER_TOKEN: usize = 1280; // 80ms @ 16kHz
