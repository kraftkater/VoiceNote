# Architecture overview

A short map for contributors. Everything lives in the repository root next to `run.bat`; there is no package layout yet.

## Modules

| File | Role |
|---|---|
| `voicenote.py` | PySide6 window: reads the engine's state, calls its methods. Holds the English UI table `EN` and the translation loader. No recognition logic. |
| `engine.py` | Everything else: settings, hotkeys, recording, model download, speech-to-text, LLM clean-up, pasting, engine packs. **No UI code.** |
| `engine_worker.py` | Separate process that hosts the optional engines (SenseVoice via sherpa-onnx, Qwen3-ASR via PyTorch). Talks to the engine over stdin/stdout: a JSON command line followed by raw float32 audio, JSON lines back. Also has a `probe` mode used after installing PyTorch. |
| `audio_pipeline.py` | `Chunker`: cuts a live audio stream into 15–40 s chunks at pauses (pure numpy, unit-tested). |
| `filejob.py` | Pure helpers for *Audio file → text*: file collection, subtitle cues, `.srt` / `.vtt` writers, paragraphs, collision-free names. |
| `sources.json` | Every external address and package name (user-editable); defaults are duplicated in `engine.py` (`_SOURCE_DEFAULTS`). |
| `setup.ps1`, `run.bat` | Portable bootstrap: download embedded Python and `get-pip`, install `requirements.txt`, start the app. |
| `lang/*.json` | UI translations; the `EN` dictionary is the fallback and source of truth. |

## Dictation flow

```
hotkey (keyboard hook) ─▶ Engine.toggle(mode) ─▶ sounddevice InputStream ─▶ raw file in recordings/
                                                       │
                                                       └▶ Chunker (pause-aware) ─▶ transcribe chunk ─▶ text queue
                                                                                                        │
stop ─▶ flush Chunker ─▶ wait for queue ─▶ (optional LLM clean-up) ─▶ paste into window / append to note file
```

Audio is written to disk while recording, so a crash loses nothing: on the next start `recordings/` is scanned and
unfinished raw files become `recovered_*.wav`.

## Models and engines

`Engine` knows four *kinds* of model:

| Kind | Where it runs | Needs |
|---|---|---|
| `whisper` | in-process, faster-whisper / CTranslate2 | base install |
| `sensevoice` | worker process, sherpa-onnx | pack `sensevoice` |
| `qwen3asr` | worker process, PyTorch + `qwen-asr` | pack `pytorch` |
| `gguf` | `llama-server` child process, HTTP on `127.0.0.1` | pack `llamacpp` |

Out-of-process kinds (`RemoteModel`, `LlamaModel`) expose the same small interface as the in-process model (`recognize(audio, lang)`, `close()`), so the dictation and file
pipelines do not care which one is loaded. Child processes are attached to a Windows *job object* (`kill_with_parent`) so they
can never outlive the program.

### Packs

`packs/<name>/` holds one optional engine with two marker files, `.installed` and `.variant` (e.g. `cuda` / `arch` / `cpu` for PyTorch, `gpu` / `cpu` for llama.cpp).
Install runs in a thread and publishes a state dict (`busy`, `stage`, `done`, `total`, `detail`, …) that the UI polls for the
live progress bar. Files are downloaded by VoiceNote itself (real progress); `pip` only resolves the wheel list, and sdists are
built with `pip wheel`. PyTorch tries each index from `torch_cuda_indexes` (filtered by GPU compute capability and driver) until a
`+cu` wheel is found, then `engine_worker.py probe` verifies that CUDA actually works.

### Model download

`Engine._download_model` talks to the Hugging Face API/CDN: HTTP `Range` resume, `.part` files, SHA-256 from the LFS metadata, a
`.downloading` marker so interrupted downloads are recognised. For GGUF repositories `pick_gguf` chooses a default quantization;
when several exist the engine sets `Engine.choose` and the UI asks the user first (choice stored in `settings.json → model_files`).

## Settings

`DEFAULTS` defines the schema. `load_settings` = `deep_merge(DEFAULTS, migrate(raw))` — unknown keys are dropped, missing ones
filled — and `validate` returns `(clean, errors)`. Bump `SETTINGS_VERSION` and extend `migrate` for incompatible changes.

## Translations

`voicenote.py: EN` → `lang/<code>.json`. `tests/test_translations.py` enforces identical key sets and `{placeholders}`.

## Testing

`tests/` covers pure logic only; `tests/conftest.py` stubs `sounddevice`, `keyboard` and `pyperclip`. GPU, Vulkan, hotkeys, Qt and the
optional engines need a real Windows machine and are tested manually — say how you tested in your pull request.
