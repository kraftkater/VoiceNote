# Third-party notices

VoiceNote itself is MIT-licensed (see [LICENSE](LICENSE)). It uses, and on demand downloads, the components below.
They are **not** part of this repository: they are installed on the user's machine from their original sources
(PyPI, python.org, Hugging Face, GitHub, download.pytorch.org). Each keeps its own license.

> **Verify before relying on this table.** The license names were compiled from the projects' public pages and may be
> out of date or incomplete. The authoritative text is the license file shipped with each package (`*.dist-info/` in the
> installed `python/Lib/site-packages` and `packs/` folders) and the project pages linked below.

## Always installed (base installation)

| Component | Use | License | Link |
|---|---|---|---|
| Python (embeddable package), pip, get-pip.py | runtime and installer | PSF License; pip: MIT | https://www.python.org/ · https://github.com/pypa/pip |
| faster-whisper | speech recognition (Whisper) | MIT | https://github.com/SYSTRAN/faster-whisper |
| CTranslate2 | inference engine of faster-whisper | MIT | https://github.com/OpenNMT/CTranslate2 |
| PyAV (with bundled FFmpeg libraries) | audio/video decoding | BSD-3-Clause (PyAV); FFmpeg libraries: LGPL/GPL, see the package | https://github.com/PyAV-Org/PyAV |
| ONNX Runtime, tokenizers, huggingface-hub | dependencies of faster-whisper | MIT / Apache-2.0 | see PyPI |
| NumPy | arrays | BSD-3-Clause | https://numpy.org |
| sounddevice (+ PortAudio) | microphone input | MIT (PortAudio: MIT-style) | https://github.com/spatialaudio/python-sounddevice |
| keyboard | global hotkeys, key simulation | MIT | https://github.com/boppreh/keyboard |
| pyperclip | clipboard | BSD-3-Clause | https://github.com/asweigart/pyperclip |
| PySide6 / Qt for Python (PySide6-Essentials) | user interface | LGPL-3.0 (also GPL-3.0 / commercial) | https://www.qt.io/qt-for-python |

Qt libraries are used as unmodified shared libraries installed separately by pip, so they can be replaced by the user
(LGPL-3.0 requirement).

## Installed on demand (optional engines in `packs/`)

| Component | Use | License | Link |
|---|---|---|---|
| NVIDIA CUDA libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) | GPU acceleration of Whisper | NVIDIA proprietary license (downloaded by the user from PyPI; not redistributed by this project) | https://developer.nvidia.com/cuda-toolkit |
| sherpa-onnx | SenseVoice engine | Apache-2.0 | https://github.com/k2-fsa/sherpa-onnx |
| PyTorch | Qwen3-ASR engine | BSD-style license | https://github.com/pytorch/pytorch |
| qwen-asr (and its dependencies, e.g. Transformers) | Qwen3-ASR engine | see the package metadata (Transformers: Apache-2.0) | https://github.com/QwenLM |
| llama.cpp (`llama-server`) | GGUF audio models | MIT | https://github.com/ggml-org/llama.cpp |

## Optional external tools

| Tool | Use | License |
|---|---|---|
| Ollama | local LLM clean-up | MIT |
| LM Studio | local LLM clean-up | proprietary freeware |

VoiceNote only talks to them over HTTP at an address the user configures; it does not bundle or install them.

## Models

Speech models are downloaded from Hugging Face on the user's request and are **separate works with their own licenses**.
Examples (verify on each model page): OpenAI Whisper weights (MIT) and their CTranslate2 conversions by Systran /
mobiuslabsgmbh; SenseVoice (code MIT, model under the FunASR model license); Qwen3-ASR and its GGUF conversions
(see the respective model cards). Check the license of a model before redistributing it or using it commercially.
