use std::io::Read;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use voxtral::audio::{
    DropOldestRing, StreamingResampler, parse_wav_bytes, resample_linear_mono_f32,
};
use voxtral::constants::SAMPLE_RATE_HZ;
use voxtral::mel::MelCtx;
use voxtral::model::{ModelBundle, ModelMetadata};

#[derive(Debug, Parser)]
#[command(name = "voxtral")]
#[command(about = "Voxtral Realtime (Rust) - WIP", long_about = None)]
struct Args {
    /// Path to a WAV file.
    #[arg(long)]
    audio: Option<PathBuf>,

    /// Read audio from stdin (WAV or raw s16le 16kHz mono).
    #[arg(long, default_value_t = false)]
    stdin: bool,

    /// Capture audio from microphone (cross-platform).
    #[arg(long, default_value_t = false)]
    from_mic: bool,

    /// Model directory with params.json / tekken.json / consolidated.safetensors.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Validate model metadata (and optionally weights) without running transcription.
    #[arg(long, default_value_t = false)]
    inspect_model: bool,

    /// When used with --inspect-model, also validate consolidated.safetensors header.
    #[arg(long, default_value_t = false)]
    inspect_weights: bool,

    /// Seconds between frontend processing reports (debug).
    #[arg(long, default_value_t = 1.0)]
    report_every: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.inspect_model {
        let model_dir = args
            .model_dir
            .as_ref()
            .context("--inspect-model requires --model-dir")?;
        return inspect_model(model_dir, args.inspect_weights);
    }

    let modes = u32::from(args.audio.is_some()) + u32::from(args.stdin) + u32::from(args.from_mic);
    if modes != 1 {
        anyhow::bail!("choose exactly one input mode: --audio, --stdin, or --from-mic");
    }

    if let Some(path) = args.audio {
        return run_file(&path);
    }

    if args.stdin {
        return run_stdin();
    }

    run_mic(Duration::from_secs_f32(args.report_every.max(0.1)))
}

fn inspect_model(model_dir: &PathBuf, inspect_weights: bool) -> Result<()> {
    let meta = ModelMetadata::load_from_dir(model_dir).context("load model metadata")?;
    let prompt = meta.tokenizer.default_prompt_ids().unwrap_or_default();
    eprintln!(
        "model ok: decoder_dim={} decoder_layers={} encoder_dim={} encoder_layers={} vocab_size={} prompt_len={}",
        meta.params.dim,
        meta.params.n_layers,
        meta.params.multimodal.whisper_model_args.encoder_args.dim,
        meta.params
            .multimodal
            .whisper_model_args
            .encoder_args
            .n_layers,
        meta.params.vocab_size,
        prompt.len()
    );

    if inspect_weights {
        let bundle =
            ModelBundle::load_from_dir(model_dir).context("load model bundle with weights")?;
        let names = bundle.weights.names().context("list tensor names")?;
        eprintln!("weights ok: tensor_count={}", names.len());
    }
    Ok(())
}

fn run_file(path: &PathBuf) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read file {path:?}"))?;
    let wav = parse_wav_bytes(&bytes).context("parse wav")?;
    let samples = if wav.sample_rate_hz == SAMPLE_RATE_HZ {
        wav.samples_mono
    } else {
        resample_linear_mono_f32(&wav.samples_mono, wav.sample_rate_hz, SAMPLE_RATE_HZ)
    };

    let mut mel = MelCtx::new(0);
    mel.feed(&samples);
    mel.finish(0);
    let (_, frames) = mel.data();
    eprintln!("mel frames: {frames}");
    Ok(())
}

fn run_stdin() -> Result<()> {
    let mut buf = Vec::new();
    std::io::stdin()
        .read_to_end(&mut buf)
        .context("read stdin")?;

    let samples = if buf.len() >= 12 && &buf[0..4] == b"RIFF" && &buf[8..12] == b"WAVE" {
        let wav = parse_wav_bytes(&buf).context("parse wav")?;
        if wav.sample_rate_hz == SAMPLE_RATE_HZ {
            wav.samples_mono
        } else {
            resample_linear_mono_f32(&wav.samples_mono, wav.sample_rate_hz, SAMPLE_RATE_HZ)
        }
    } else {
        // raw s16le 16kHz mono
        if buf.len() % 2 != 0 {
            buf.pop();
        }
        let mut out = Vec::with_capacity(buf.len() / 2);
        for b in buf.chunks_exact(2) {
            let s = i16::from_le_bytes([b[0], b[1]]);
            out.push((s as f32) / 32768.0);
        }
        out
    };

    let mut mel = MelCtx::new(0);
    mel.feed(&samples);
    mel.finish(0);
    let (_, frames) = mel.data();
    eprintln!("mel frames: {frames}");
    Ok(())
}

fn run_mic(report_every: Duration) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default input device")?;

    let cfg = device.default_input_config().context("default config")?;
    let channels = cfg.channels();
    let src_hz = cfg.sample_rate().0;
    let stream_config: cpal::StreamConfig = cfg.clone().into();

    eprintln!(
        "mic: device={:?} sample_rate={} channels={} format={:?}",
        device.name().ok(),
        src_hz,
        channels,
        cfg.sample_format()
    );

    // Drop-oldest ring buffer in *source* sample rate domain.
    // Keep ~5 seconds of audio to bound latency.
    let cap_samples = (src_hz as usize).saturating_mul(5);
    let ring = Arc::new(Mutex::new(DropOldestRing::new(cap_samples)));
    let dropped = Arc::new(AtomicU64::new(0));

    let running = Arc::new(AtomicBool::new(true));
    {
        let running = Arc::clone(&running);
        ctrlc::set_handler(move || {
            running.store(false, Ordering::SeqCst);
        })
        .context("install ctrl-c handler")?;
    }

    let err_fn = |e| eprintln!("mic stream error: {e}");

    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            let ring_cb = Arc::clone(&ring);
            let dropped_cb = Arc::clone(&dropped);
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    let mut mono = Vec::with_capacity(data.len() / channels as usize);
                    for frame in data.chunks_exact(channels as usize) {
                        let mut sum = 0.0f32;
                        for &s in frame {
                            sum += s;
                        }
                        mono.push(sum / (channels as f32));
                    }

                    let mut r = ring_cb.lock().expect("ring lock");
                    let before = r.dropped_samples();
                    r.push(&mono);
                    let after = r.dropped_samples();
                    dropped_cb.fetch_add(after - before, Ordering::Relaxed);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let ring_cb = Arc::clone(&ring);
            let dropped_cb = Arc::clone(&dropped);
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let mut mono = Vec::with_capacity(data.len() / channels as usize);
                    for frame in data.chunks_exact(channels as usize) {
                        let mut sum = 0.0f32;
                        for &s in frame {
                            sum += (s as f32) / 32768.0;
                        }
                        mono.push(sum / (channels as f32));
                    }

                    let mut r = ring_cb.lock().expect("ring lock");
                    let before = r.dropped_samples();
                    r.push(&mono);
                    let after = r.dropped_samples();
                    dropped_cb.fetch_add(after - before, Ordering::Relaxed);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let ring_cb = Arc::clone(&ring);
            let dropped_cb = Arc::clone(&dropped);
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| {
                    let mut mono = Vec::with_capacity(data.len() / channels as usize);
                    for frame in data.chunks_exact(channels as usize) {
                        let mut sum = 0.0f32;
                        for &s in frame {
                            let s = (s as i32) - 32768;
                            sum += (s as f32) / 32768.0;
                        }
                        mono.push(sum / (channels as f32));
                    }

                    let mut r = ring_cb.lock().expect("ring lock");
                    let before = r.dropped_samples();
                    r.push(&mono);
                    let after = r.dropped_samples();
                    dropped_cb.fetch_add(after - before, Ordering::Relaxed);
                },
                err_fn,
                None,
            )?
        }
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    };

    stream.play().context("start mic stream")?;

    // Frontend consumer loop.
    let mut resampler = StreamingResampler::new(src_hz, SAMPLE_RATE_HZ);
    let mut mel = MelCtx::new(32 * voxtral::constants::RAW_AUDIO_SAMPLES_PER_TOKEN);
    let mut tmp = Vec::new();
    let mut last_dropped_total = 0u64;

    let mut last_report = Instant::now();
    let start = Instant::now();
    while running.load(Ordering::SeqCst) {
        {
            let mut r = ring.lock().expect("ring lock");
            let dropped_total = r.dropped_samples();
            let delta = dropped_total.saturating_sub(last_dropped_total);
            last_dropped_total = dropped_total;
            resampler.skip_input_samples(delta);
            r.drain_into(&mut tmp, src_hz as usize / 10); // up to 100ms
        }

        if !tmp.is_empty() {
            let samples_16k = resampler.process(&tmp);
            mel.feed(&samples_16k);
        } else {
            std::thread::sleep(Duration::from_millis(5));
        }

        if last_report.elapsed() >= report_every {
            let (_, frames) = mel.data();
            let dropped = dropped.load(Ordering::Relaxed);
            let secs = start.elapsed().as_secs_f32();
            eprintln!("t={secs:.1}s mel_frames={frames} dropped_samples={dropped}");
            last_report = Instant::now();
        }
    }

    mel.finish(0);
    let (_, frames) = mel.data();
    eprintln!(
        "stopped. mel_frames={frames} dropped_samples={}",
        dropped.load(Ordering::Relaxed)
    );
    Ok(())
}
