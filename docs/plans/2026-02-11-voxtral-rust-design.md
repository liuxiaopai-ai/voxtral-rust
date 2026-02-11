# Design: Voxtral Realtime (Rust, CPU-first) (2026-02-11)

## Goals

- Provide a **library-grade** Rust implementation of Voxtral Mini 4B Realtime inference,
  focusing on **correctness**, **streaming**, and **cross-platform** microphone input.
- Provide a CLI that matches the reference project's UX:
  - file transcription
  - stdin piping (WAV or raw s16le 16k mono)
  - microphone transcription (`--from-mic`)
- Be CI-friendly:
  - Always run `fmt`, `clippy`, unit tests on Linux/macOS/Windows
  - Avoid requiring the 9GB model weights in CI
  - Keep a gated E2E mode for local runs

## Non-goals (v0.1)

- Matching the reference project's Apple Metal fused-kernel performance.
- Quantization (GGUF/Q4) and Web/WASM support.

## High-Level Architecture

We split the project into a small workspace:

- `voxtral` (library)
  - Model loading: `safetensors` via memory-mapped file
  - Tokenizer: Tekken JSON decoder
  - Audio frontend: resample -> incremental log-mel
  - Streaming pipeline: conv stem tail-state -> encoder (incremental) -> adapter -> decoder
  - Rolling KV cache with compaction

- `voxtral-cli` (binary)
  - CLI parsing
  - WAV file input + stdin decoding
  - Cross-platform microphone capture (cpal)
  - Backpressure policy: **drop-oldest** audio when behind real-time

## Streaming Behavior

We intentionally mirror the reference C implementation's streaming schedule:
- Prefix prompt ids: `[BOS] + [STREAMING_PAD] * (32 + delay_tokens)`
- Start the decoder once there are at least `L` adapter positions
- Prefill `L-1` positions, then generate 1 token at `L-1`
- For each subsequent adapter position `pos`, feed:
  - `adapter[pos] + tok_embed(prev_token)`

End-of-stream flushing:
- Align to 1280-sample token boundary (80ms)
- Append `(delay_tokens + 1 + 10)` additional token steps of zero audio

## Microphone Backpressure

When inference can't keep up with real-time:
- Maintain a bounded audio ring buffer
- If the buffer is full, **discard the oldest samples** before pushing new ones
- Periodically log a warning and the number of dropped samples

This keeps latency bounded and avoids the "slowly increasing delay" failure mode.

## Testing Strategy

### CI-default tests (no model weights)

- Audio frontend unit tests:
  - window function shape and basic properties
  - mel filter bank construction sanity checks
  - log scaling clamp behavior
- Streaming engine unit tests:
  - conv tail alignment correctness (pure shape + simple numeric cases)
  - 4x downsample residual alignment
  - KV cache compaction logic (synthetic tensors)

### Deterministic small-model tests

Add a "mini" configuration:
- small dims/layers/heads
- random weights with a fixed seed
- compare against a naive reference implementation (same codebase)

This validates attention/RMSNorm/RoPE/FFN correctness without huge weights.

### Optional E2E tests (local)

- Enabled when `VOXTRAL_MODEL_DIR` is set
- Run a short audio clip and assert key substrings appear

## CI

- GitHub Actions matrix: ubuntu-latest, macos-latest, windows-latest
- Steps: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`

## Packaging (later)

- Add `cargo-dist` for release artifacts when the CLI is stable.
