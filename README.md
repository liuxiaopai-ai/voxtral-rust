# Voxtral Realtime in Rust

A from-scratch, streaming-first Rust implementation of **Mistral Voxtral Mini 4B Realtime**.

This project aims to be a **library-grade** speech-to-text engine:
- cross-platform audio input (Linux/macOS/Windows)
- deterministic streaming behavior
- strong automated tests and CI
- clean architecture for long-term maintainability

[![CI](https://github.com/liuxiaopai-ai/voxtral-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/liuxiaopai-ai/voxtral-rust/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE-MIT)

## Why This Exists

The original pure-C Voxtral implementation proved the model can be reproduced with a transparent, low-dependency stack.

This Rust project takes that spirit and adds:
- safer systems programming
- reusable APIs for app integration
- stronger cross-platform ergonomics
- stricter correctness guardrails in CI

The goal is not a wrapper. The goal is a robust open source inference core.

## Status

Active development.

Implemented today:
- [x] Audio frontend: WAV parsing, linear resampling, incremental log-mel
- [x] Streaming conv stem with boundary correctness tests
- [x] Tekken tokenizer loading and decode path (`tekken.json`)
- [x] Model params loading/validation (`params.json`)
- [x] Safetensors mmap loading with BF16/F32 conversion helpers
- [x] Rolling KV cache with sliding-window compaction
- [x] Core kernels: RMSNorm, RoPE, linear, GQA attention step
- [x] Decoder single-layer incremental step skeleton (attn + SwiGLU + KV)
- [x] Cross-platform microphone capture with drop-oldest backpressure
- [x] CI matrix on Linux/macOS/Windows

In progress:
- [ ] Full end-to-end encoder + adapter + decoder forward path
- [ ] Real transcription output from full model path in CLI
- [ ] Performance backends and benchmark suite

## Quick Start

### Build and quality checks

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo build --all-targets --release
```

### CLI frontend smoke

```bash
# WAV input
cargo run -p voxtral-cli -- --audio path/to/audio.wav

# stdin input (WAV or raw s16le mono 16k)
cat path/to/audio.wav | cargo run -p voxtral-cli -- --stdin

# microphone input (Ctrl+C to stop)
cargo run -p voxtral-cli -- --from-mic
```

### Model asset checks

```bash
# Validate params + tokenizer
cargo run -p voxtral-cli -- --inspect-model --model-dir /path/to/model

# Validate params + tokenizer + safetensors header/index
cargo run -p voxtral-cli -- --inspect-model --inspect-weights --model-dir /path/to/model
```

### Optional local E2E smoke

```bash
VOXTRAL_MODEL_DIR=/path/to/model \
  cargo test -p voxtral --test e2e_model_env -- --nocapture
```

## Streaming Architecture

```text
Audio (file/stdin/mic)
  -> resample to 16k mono
  -> incremental log-mel
  -> incremental conv stem
  -> encoder (incremental)
  -> adapter
  -> decoder (autoregressive + rolling KV cache)
  -> tokens
```

Prompt/flush scheduling follows the reference strategy:
- prompt: `[BOS] + [STREAMING_PAD] * (left_pad + delay_tokens)`
- decode starts once adapter positions reach prompt length
- finish flush aligns to token boundary and appends right-pad token budget

## Repo Layout

```text
crates/voxtral
  audio.rs      mel.rs        conv.rs       stream.rs
  params.rs     tokenizer.rs  weights.rs    model.rs
  math.rs       ops.rs        kv.rs         pipeline.rs
  decoder.rs

crates/voxtral-cli
  main.rs
```

## CI and Quality Bar

GitHub Actions runs on every push/PR:
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --all-targets`
- `cargo build --all-targets --release`

Matrix:
- `ubuntu-latest`
- `macos-latest`
- `windows-latest`

## Roadmap

1. Complete full-model inference path (encoder + adapter + decoder loop)
2. Add deterministic parity tests for critical tensor paths
3. Add benchmark suite (latency / throughput / memory)
4. Package binary releases for all major platforms

## Third-Party and Attribution

- Third-party notices: `THIRD_PARTY_NOTICES.md`
- Acknowledgement: original reference implementation `antirez/voxtral.c` (MIT)
- Model weights (`mistralai/Voxtral-Mini-4B-Realtime-2602`) are distributed separately under their own terms

## License

`MIT OR Apache-2.0`
