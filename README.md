# Voxtral Realtime (Rust)

A from-scratch Rust implementation of Mistral's Voxtral Mini 4B Realtime speech-to-text
model, focused on a library-quality streaming API and cross-platform CLI.

Status: active implementation (frontend + model asset loading + streaming schedule).

## Goals

- Correctness-first implementation with strong tests
- Cross-platform CLI: file, stdin piping, microphone
- CI on Linux/macOS/Windows

## Non-goals (initially)

- Metal fused-kernel performance parity with optimized native implementations
- Quantization / WASM

## Docs

- Design: `docs/plans/2026-02-11-voxtral-rust-design.md`
- Research notes: `docs/research/2026-02-11-voxtral-c-research.md`

## What's Implemented

- Audio frontend:
  - WAV parsing (`s16` PCM), linear resampler, incremental log-mel
  - Streaming microphone input with bounded drop-oldest ring buffer
- Streaming alignment core:
  - Incremental conv stem with boundary-correct overlap logic
  - Prompt/flush scheduling helpers (`[BOS] + [STREAMING_PAD]...`, right-pad flush)
  - `PipelineState` for adapter-token/decoding-budget tracking
- Model assets:
  - `params.json` parser and validation
  - `tekken.json` decode-only tokenizer loader
  - `safetensors` mmap loader with BF16/F32 conversion helpers
- Inference primitives:
  - RMSNorm / RoPE / Softmax helpers
  - Linear op and GQA attention step with reference parity tests
- CI:
  - GitHub Actions matrix on Linux/macOS/Windows
  - `fmt`, `clippy -D warnings`, `test`, `build --release`

## CLI (Current)

```bash
# 1) Audio frontend smoke (prints mel frame count)
cargo run -p voxtral-cli -- --audio path/to/audio.wav

# 2) Stdin mode (WAV or raw s16le mono 16kHz)
cat path/to/audio.wav | cargo run -p voxtral-cli -- --stdin

# 3) Cross-platform microphone capture (Ctrl+C to stop)
cargo run -p voxtral-cli -- --from-mic

# 4) Validate model directory metadata
cargo run -p voxtral-cli -- --inspect-model --model-dir /path/to/model

# 5) Validate model directory + safetensors header
cargo run -p voxtral-cli -- --inspect-model --inspect-weights --model-dir /path/to/model
```

## Optional E2E Smoke

With real model files locally present:

```bash
VOXTRAL_MODEL_DIR=/path/to/model cargo test -p voxtral --test e2e_model_env -- --nocapture
```

## Third-party

- Notices and attributions: `THIRD_PARTY_NOTICES.md`

## License

MIT OR Apache-2.0
