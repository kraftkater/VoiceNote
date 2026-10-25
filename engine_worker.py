"""Speech engine in a separate process (SenseVoice via sherpa-onnx, Qwen3-ASR via PyTorch).

Why a separate process: the extra engines bring their own libraries (installed into packs/<name>), which may need
other versions than the main program. Here they never touch the main program's libraries, and killing this process
frees all of their memory at once.

Protocol (stdin/stdout, one JSON object per line):
  worker -> {"ready": true, "device": "cpu"}   or   {"error": "..."}
  main   -> {"cmd": "rec", "n": <bytes>, "lang": "en"}  followed by n bytes of float32 mono 16 kHz audio
  worker -> {"text": "..."}   or   {"error": "..."}
  main   -> {"cmd": "quit"}
"""
import json
import os
import re
import sys
import tempfile
import traceback
import wave
from pathlib import Path


# Qwen3-ASR takes the English language name (same table as in engine.py)
QWEN_LANGS = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French",
              "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ko": "Korean",
              "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish", "hi": "Hindi",
              "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish",
              "cs": "Czech", "tl": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
              "mk": "Macedonian", "ro": "Romanian"}


def prepare(pack):
    """Make the libraries of the engine pack importable (and their DLL folders loadable)."""
    sys.path.insert(0, pack)
    if os.name == "nt":  # DLLs shipped inside the pack (torch\\lib, sherpa_onnx\\lib, ...)
        for d in Path(pack).glob("*/lib"):
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass


def cuda_state(torch):
    """('ok' | 'no_cuda' | 'arch', info): is there a GPU this torch build can really run on?
    A CUDA build only supports the GPU architectures it was compiled for (plus PTX for newer ones)."""
    if not torch.cuda.is_available():
        return "no_cuda", ""
    try:
        major, minor = torch.cuda.get_device_capability(0)
        archs = list(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001 - cannot tell: assume it works
        return "ok", ""
    cc = major * 10 + minor
    if f"sm_{cc}" in archs or any(a.startswith("compute_") and a[8:].isdigit() and int(a[8:]) <= cc for a in archs):
        return "ok", ""
    return "arch", f"sm_{cc} not in {archs}"


def probe(pack):
    """`engine_worker.py probe <pack>`: report what the installed torch can do (used right after the install)."""
    prepare(pack)
    try:
        import torch
        state, info = cuda_state(torch)
        print(json.dumps({"torch": torch.__version__, "cuda_build": bool(torch.version.cuda), "state": state, "info": info}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))


def main():
    if sys.argv[1] == "probe":
        return probe(sys.argv[2])
    kind, model_dir, device, pack = sys.argv[1:5]
    proto = sys.stdout.buffer
    sys.stdout = sys.stderr  # library chatter must never corrupt the protocol stream
    prepare(pack)

    def send(obj):
        proto.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        proto.flush()

    try:
        import numpy as np
        md = Path(model_dir)
        if kind == "sensevoice":
            import sherpa_onnx
            model = md / "model.int8.onnx"
            if not model.exists():
                model = md / "model.onnx"
            cache = {}

            def get(lang):
                if lang not in cache:
                    cache[lang] = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                        model=str(model), tokens=str(md / "tokens.txt"), num_threads=max(2, (os.cpu_count() or 4) // 2),
                        use_itn=True, language=lang, debug=False)
                return cache[lang]

            def run(audio, lang):
                rec = get(lang if lang in ("zh", "en", "ja", "ko", "yue") else "auto")
                s = rec.create_stream()
                s.accept_waveform(16000, audio)
                rec.decode_stream(s)
                return s.result.text

            get("auto")
            used = "cpu"
        elif kind == "qwen3asr":
            import torch
            from qwen_asr import Qwen3ASRModel
            state, info = cuda_state(torch)
            if device == "cuda" and state != "ok":
                raise RuntimeError("torch_cpu" if state == "no_cuda" else "torch_arch")  # the UI explains both
            cuda = state == "ok" and device in ("cuda", "auto")   # Auto never picks a GPU torch cannot use
            used = "cuda" if cuda else "cpu"
            model = Qwen3ASRModel.from_pretrained(str(md), dtype=torch.bfloat16 if cuda else torch.float32,
                                                  device_map="cuda:0" if cuda else "cpu")

            def run(audio, lang):
                with tempfile.TemporaryDirectory() as t:  # a wav path is the input form every version accepts
                    p = os.path.join(t, "chunk.wav")
                    with wave.open(p, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(16000)
                        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
                    res = model.transcribe(audio=p, language=QWEN_LANGS.get(lang))  # None = auto-detect
                return res[0].text
        else:
            raise RuntimeError(f"unknown engine kind '{kind}'")
    except Exception as e:  # noqa: BLE001 - reported to the main program
        traceback.print_exc()  # into worker.log
        send({"error": str(e) if str(e) in ("torch_cpu", "torch_arch") else f"{type(e).__name__}: {e}"})
        return
    send({"ready": True, "device": used})

    inp = sys.stdin.buffer
    while True:
        line = inp.readline()
        if not line:
            return  # the main program is gone
        try:
            msg = json.loads(line)
            if msg.get("cmd") == "quit":
                return
            raw = inp.read(int(msg["n"]))
            audio = np.frombuffer(raw, dtype="float32")
            text = re.sub(r"<\|[^|>]*\|>", "", str(run(audio, msg.get("lang") or "auto"))).strip()
            send({"text": text})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
