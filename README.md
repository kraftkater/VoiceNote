<div align="center">

# VoiceNote

**Portable, fully local voice dictation for Windows.**
Press a hotkey, speak, and the text appears in the window you are working in — or in a numbered note file.
Everything runs on your PC: no cloud, no account, no telemetry.

[![CI](https://github.com/kraftkater/VoiceNote/actions/workflows/ci.yml/badge.svg)](https://github.com/kraftkater/VoiceNote/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform: Windows 10/11](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D6)
![Python 3.11](https://img.shields.io/badge/python-3.11%20(bundled)-3776AB)
![Status: beta](https://img.shields.io/badge/status-beta-orange)

[Features](#features) · [Quick start](#quick-start) · [Hotkeys](#hotkeys) · [Speech engines](#speech-engines) · [Configuration](#configuration) · [Troubleshooting](#troubleshooting) · [Contributing](CONTRIBUTING.md) · [Русская версия](README.ru.md)

</div>

> **Screenshot:** add `docs/images/main-window.png` and reference it here
> (`![VoiceNote main window](docs/images/main-window.png)`).

---

## Features

**Dictation**
- Global hotkeys: dictate into the **active window** (`F9`), the same with **LLM clean-up** (`F10`), or into a
  **numbered note file** (`F7` / `F8`, e.g. `note_01.txt`, `note_02.txt`, …). Every hotkey and the cancel key can be changed.
- **"Continue" mode** per hotkey (checkbox or `Alt` + hotkey): keep appending to the last note, or keep sending text to the
  window you started in. Distinct beeps tell you which mode you toggled.
- **Long recordings without glitches**: audio is streamed to disk, cut at natural pauses into 15–40 s chunks and transcribed
  while you speak; notes grow incrementally. After a crash the raw audio is converted to WAV on the next start.
- Optional **LLM clean-up** (punctuation, filler words, paragraphs) through [Ollama](https://ollama.com),
  [LM Studio](https://lmstudio.ai) or any OpenAI-compatible local server. The processor (CPU / GPU) for Ollama is selectable.

**Speech recognition**
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) models from *tiny* to *large-v3-turbo*, on **CPU or NVIDIA GPU**.
- Models are downloaded with **real byte progress**, pause / resume / stop, SHA-256 verification, and can be deleted again.
- **Add a model by Hugging Face link** (CTranslate2 Whisper, SenseVoice, Qwen3-ASR, GGUF audio models). If a repository offers
  several files (e.g. quantizations) **you choose which one to download**.
- **Already have a model?** Copy its folder into `models/` — it is detected automatically (CTranslate2 Whisper, SenseVoice, Qwen3-ASR, GGUF; the folder name must contain `sense-voice` / `qwen3-asr` for those two) and appears in the list within seconds.
- The language list shows **only the languages the selected model supports** (read from the model's own files where possible).
- Optional extra engines, installed only on demand into a removable `packs/` folder — see [Speech engines](#speech-engines).

**Audio file → text**
- Drop audio/video files (mp3, wav, m4a, mp4, mkv, ogg, flac, …) or a whole folder onto the window: plain text plus optional
  **`.srt` / `.vtt` subtitles**, batch processing with progress and cancel.

**Everything else**
- **Portable**: an embedded Python is downloaded into the program folder; nothing is installed system-wide. Delete the folder to uninstall.
- Native **PySide6** window, light/dark theme, system tray, autosave of every setting, 8 UI languages
  (English, Русский, Українська, Español, Deutsch, Français, Português, 简体中文).
- **No hard-coded download links**: every external address and package name lives in [`sources.json`](sources.json).

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11, x64 |
| Microphone | any input device |
| Internet | first run (Python + libraries) and for downloading models |
| Disk | a few hundred MB for Python + libraries, plus models (75 MB … several GB) |
| GPU (optional) | NVIDIA with a recent driver for CUDA acceleration (driver 527+ for the CUDA 12 libraries) |

Linux and macOS are **not** supported (global hotkeys, window handling and the portable setup are Windows-specific).

## Quick start

**From a release** (recommended)

1. Download `VoiceNote-<version>.zip` from the [Releases](../../releases) page and unzip it anywhere (a path without
   special characters is the safest).
2. Run **`run.bat`**. On the first start it downloads the portable Python and the libraries into the folder (needs internet).
3. In *Speech recognition* pick a model — it is downloaded on first use — and press **F9**.

**From source**

```bat
git clone https://github.com/kraftkater/VoiceNote.git
cd VoiceNote
run.bat
```

`run.bat` repairs the installation automatically if something is missing. The app writes `crash.log` next to itself if it fails to start.

## Hotkeys

| Key | Action |
|---|---|
| `F9` | Dictate into the active window (press again to stop; the text is pasted at the end) |
| `F10` | Same, with LLM clean-up |
| `F7` / `F8` | Dictate into the next numbered note file / with LLM clean-up |
| `Pause` | Cancel the current recording (on long recordings press twice within 3 s) |
| `Alt` + hotkey | Toggle **Continue** mode for that hotkey |

All keys are configurable in the UI. Windows running **as administrator** only receive pasted text if VoiceNote is also started as administrator.

## Speech engines

| Engine | Runs | Install size | Status |
|---|---|---|---|
| **faster-whisper** (built in) | CPU / NVIDIA GPU | in the base install | main path |
| **GPU support** (CUDA 12 libraries) | NVIDIA GPU | ~1 GB, on demand | working |
| **SenseVoice** (sherpa-onnx) | CPU | small, on demand | working — zh / en / ja / ko / yue only |
| **PyTorch** (Qwen3-ASR) | CPU / CUDA | several GB, on demand | **experimental** |
| **llama.cpp** (GGUF audio models, e.g. Qwen3-ASR, Voxtral) | CPU / GPU (Vulkan) | small, on demand | **experimental** |

- Extra engines are installed from the *Extra engines* section into `packs/<name>` — **separately from the main libraries**,
  so they cannot break them — and can be removed there at any time. They run in a separate process; unloading a model kills it
  and frees its memory.
- The PyTorch installer detects your NVIDIA GPU and driver and downloads the matching CUDA build (RTX 50xx needs CUDA 12.8+),
  then verifies that the installed build can really run on your GPU. Without an NVIDIA GPU the CPU build is installed.
- The llama.cpp engine downloads a ready-made `llama-server` from the
  [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (Vulkan build when a GPU is found, CPU build otherwise).
  GGUF repositories made for *other* programs (for example CrispASR) will not run in it.
- Not supported: NVIDIA NeMo (`.nemo`) models and plain ONNX / Transformers weights without a matching engine.

## Configuration

User data stays in the program folder:

| Path | Contents |
|---|---|
| `settings.json` | all settings (autosaved) |
| `sources.json` | download addresses and package names — edit if a link ever changes |
| `models/` | downloaded speech models |
| `packs/` | optional engines (`gpu`, `sensevoice`, `pytorch`, `llamacpp`) |
| `notes/` (default) | note files from `F7` / `F8` and results of *Audio file → text* |
| `recordings/` | temporary raw audio of the running recording (optionally kept as WAV) |
| `crash.log`, `worker.log`, `packs/install.log` | diagnostics |

`sources.json` keys: `python_embed_urls`, `get_pip_urls`, `pip_index_urls` (mirrors, tried in order), `pip_extra_index_urls`,
`torch_cuda_indexes`, `hf_base` (set a mirror here), `packs` (pin versions, e.g. `"torch==2.5.1"`), `llamacpp_release_api`,
`llamacpp_assets`, `whisper_repos`, `extra_models`, `keep_wheels`. A broken file is ignored and the built-in defaults are used.

## Privacy

- Audio and text never leave your machine. There is no telemetry and no account.
- Network access happens only to download things you ask for: Python / pip (python.org, bootstrap.pypa.io, PyPI),
  models (Hugging Face), PyTorch builds (download.pytorch.org), `llama-server` (GitHub), and to the **local LLM server
  address you configure** (Ollama / LM Studio / custom).

## Troubleshooting

| Symptom | What to check |
|---|---|
| Window does not open | `crash.log` next to `run.bat` |
| Hotkey does nothing | Another program may use the key; change it. Antivirus software sometimes flags global keyboard hooks — see below |
| Text is not pasted into an app | The app runs elevated: start VoiceNote as administrator too |
| GPU not used | *Speech recognition → Processor* shows what runs where; install *GPU support* (NVIDIA only, driver 527+) |
| Extra engine fails | `worker.log` (engine process) and `packs/install.log` (installation) |
| A download link stopped working | Edit `sources.json` |

> **Antivirus note.** Global hotkeys are implemented with the [`keyboard`](https://github.com/boppreh/keyboard) library,
> which installs a low-level keyboard hook. Some security products flag *any* program that does this. VoiceNote does not log keys;
> the source is in this repository.

## Project status

VoiceNote is in **beta**. Automated tests (`pytest`, run in CI) cover the pure logic — settings validation, chunking, subtitles,
model-file selection, translations. Hardware- and OS-specific paths (CUDA, Vulkan, global hotkeys, the Qt UI, the extra engines)
cannot run on CI and are verified manually on a limited set of machines. The **experimental** engines in particular may behave
differently on your PC. Please [open an issue](../../issues/new/choose) with your logs if something fails.

## Contributing

Bug reports, translations and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the short
[architecture overview](docs/ARCHITECTURE.md). Please follow the [Code of Conduct](CODE_OF_CONDUCT.md);
report security issues as described in [SECURITY.md](SECURITY.md). Release history: [CHANGELOG.md](CHANGELOG.md).

## License

VoiceNote is released under the [MIT License](LICENSE). Third-party components keep their own licenses — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). **Speech models are separate works with their own licenses**
(check each model's page on Hugging Face before redistributing or using it commercially).

## Acknowledgements

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2),
[OpenAI Whisper](https://github.com/openai/whisper), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx),
[SenseVoice / FunASR](https://github.com/FunAudioLLM/SenseVoice), [Qwen3-ASR](https://github.com/QwenLM),
[llama.cpp](https://github.com/ggml-org/llama.cpp), [PyTorch](https://pytorch.org), [Qt for Python](https://www.qt.io/qt-for-python),
[sounddevice](https://github.com/spatialaudio/python-sounddevice), [keyboard](https://github.com/boppreh/keyboard),
[pyperclip](https://github.com/asweigart/pyperclip).
