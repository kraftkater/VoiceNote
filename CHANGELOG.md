# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may contain breaking changes).

## [Unreleased]

## [0.9.0] — first public beta

### Added
- Global-hotkey dictation into the active window (`F9`, `F10` with LLM clean-up) and into numbered note files (`F7`, `F8`),
  cancel key, "Continue" mode with `Alt` + hotkey toggles and distinct beeps.
- Long-recording pipeline: audio streamed to disk, cut at pauses, transcribed incrementally, crash recovery to WAV.
- faster-whisper models on CPU or NVIDIA GPU, own downloader (real progress, pause / resume / stop, SHA-256, delete),
  add-by-link from Hugging Face, per-model language lists read from the model files, choice of the model file
  (quantization) when a repository has several.
- Optional engines in `packs/`: GPU support (CUDA 12 libraries), SenseVoice (sherpa-onnx), PyTorch (Qwen3-ASR, CUDA build
  selected by GPU / driver and verified after installation), llama.cpp (`llama-server`, GGUF audio models, Vulkan or CPU build).
- Audio / video file → text with `.srt` / `.vtt` subtitles, drag & drop, folders, batch, cancel.
- Optional LLM clean-up through Ollama, LM Studio or any OpenAI-compatible server; selectable processor for Ollama.
- Fixed top bar (status, recording timer, last result), window size remembered, light/dark theme, tray icon.
- Portable installation; all external addresses in `sources.json`.
- Model folders copied into `models/` by hand are detected automatically (Whisper CTranslate2, SenseVoice, Qwen3-ASR, GGUF) and appear in the list.
- 8 UI languages.

[Unreleased]: https://github.com/kraftkater/VoiceNote/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/kraftkater/VoiceNote/releases/tag/v0.9.0
