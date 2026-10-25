# Security policy

## Supported versions

Only the latest release receives fixes while the project is in beta.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Use GitHub's private reporting: **Security → Report a vulnerability** on this repository
(<https://github.com/kraftkater/VoiceNote/security/advisories/new>). Include the affected version, steps to reproduce and the impact you expect.

You can expect an acknowledgement within a few days. Fixes are published in a new release together with a note in
[CHANGELOG.md](CHANGELOG.md); reporters are credited unless they prefer otherwise.

## Scope and design notes

VoiceNote is a local desktop program. Things worth knowing when assessing its security:

- **Global keyboard hook.** Hotkeys use the `keyboard` library (a low-level hook). VoiceNote reacts only to the configured
  keys and does not record keystrokes.
- **Downloads.** The program downloads Python and pip packages (python.org, bootstrap.pypa.io, PyPI), models (Hugging Face),
  PyTorch builds (download.pytorch.org) and `llama-server` (GitHub releases). Model files are checked with the SHA-256 sums
  published by Hugging Face; wheels with the hashes reported by pip; `llama-server` archives with the digest reported by the
  GitHub API when it is present. All addresses are configurable in `sources.json`, so a local mirror can be used.
- **Local network.** The optional `llama-server` listens on `127.0.0.1` on a random port only while a model is loaded.
  The optional LLM clean-up sends your dictated text to the server address you configure — choose a server you trust.
- **No telemetry.** The program sends nothing anywhere by itself.
- **Pickled / executable model formats** are not loaded: supported model formats are CTranslate2, ONNX, safetensors and GGUF.
