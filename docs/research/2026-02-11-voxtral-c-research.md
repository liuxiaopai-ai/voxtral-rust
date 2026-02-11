# Research Notes: voxtral.c (2026-02-11)

This note summarizes what we learned from Salvatore Sanfilippo's `antirez/voxtral.c` project
(a pure C implementation of Mistral's Voxtral Mini 4B Realtime inference pipeline), and
highlights implications for a from-scratch Rust rewrite.

References:
- `antirez/voxtral.c` (MIT licensed) repository
- Model: `mistralai/Voxtral-Mini-4B-Realtime-2602` (Apache-2.0 licensed weights)

## What voxtral.c Implements

Voxtral Realtime is a streaming speech-to-text model with two main parts:
- Audio encoder: causal Transformer, sliding-window attention (window=750)
- Text decoder: autoregressive Transformer, GQA (32Q / 8KV), sliding-window attention (window=8192)

`voxtral.c` implements:
- A streaming inference pipeline (feed audio incrementally, receive tokens incrementally)
- Audio preprocessing: resampling to 16kHz, log-mel spectrogram
- Chunked/incremental encoder execution with bounded memory:
  - conv stem tail-state handling (stride=2 conv) to preserve correct alignment
  - encoder KV cache for incremental attention
  - residual handling for 4x downsample alignment
- Decoder with a rolling KV cache (compaction when past the window)
- A Tekken tokenizer decoder (`tekken.json`) for token-id -> bytes -> UTF-8

The repo also includes a self-contained Python reference implementation and a detailed
architecture doc (`MODEL.md`) describing tensor names, shapes, and the offline decoding
schedule.

## Key Algorithms / Conventions To Match

### Log-mel
The mel front-end must match the model's expectations closely. In `voxtral.c`:
- Hann window, size=400
- STFT framing with center padding semantics
- Slaney-style mel filter bank (0..8000 Hz, 128 bins)
- log10 clamp to `[global_log_mel_max-8, global_log_mel_max]`, then `(val+4)/4`

### Token-time mapping
- Frame rate: 12.5 Hz
- 1 text token corresponds to 80ms of audio (`1280` samples at 16kHz)

### Streaming schedule implemented in C
The C code effectively performs the *offline schedule incrementally*:
- Build prompt ids = `[BOS] + [STREAMING_PAD] * (32 + delay_tokens)`
- Prefill for prompt positions `0..L-2`
- Generate first token at position `L-1`
- For each new audio embedding position `pos >= L`, feed `audio_embed[pos] + tok_embed(prev_token)`

At stream end, it feeds a right-padding budget so the decoder can emit tokens that
are behind the delay window:
- Align to 1280-sample token boundary
- Append `(delay_tokens + 1 + 10)` additional token steps of zero audio

## Engineering Implications for a Rust Rewrite

### Feasibility
A full Rust rewrite is feasible because `voxtral.c` provides:
- A readable, complete end-to-end pipeline
- A detailed model reference (`MODEL.md`) with exact tensor names/shapes
- A regression test strategy based on substring matching (not bit-exact)

### Hard parts (risk)
- Correctness-sensitive audio frontend (mel + framing + padding)
- Streaming alignment (conv stride=2 tail-state, 4x downsample residual)
- Rolling KV cache compaction correctness (long-form audio)
- Maintaining stable behavior across platforms (floating point variance)

### Suggested test strategy (CI-friendly)
Since GitHub Actions cannot realistically download/run 9GB weights + GPU backends:
- Unit tests for audio frontend primitives (windowing, mel filters, log scaling)
- Unit tests for cache compaction logic using synthetic tensors
- Small-shape "mini-model" tests (random weights, deterministic seed) against a naive reference
- Optional E2E tests gated by env var (requires local model directory)

## Related implementations
There are already other Voxtral implementations in the wild (Rust + ggml). A new
project should differentiate on "library-quality" engineering: stable API, strong
CI, packaging, and carefully designed streaming interfaces.
