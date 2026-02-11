# Voxtral Realtime (Rust)

A from-scratch Rust implementation of Mistral's Voxtral Mini 4B Realtime speech-to-text
model, focused on a library-quality streaming API and cross-platform CLI.

Status: early design / scaffolding.

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

## Third-party

- Notices and attributions: `THIRD_PARTY_NOTICES.md`

## License

MIT OR Apache-2.0
