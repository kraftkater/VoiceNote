"""Test setup: the engine imports Windows/audio-only modules at import time; replace them with stubs so the pure
logic can be tested on any OS (CI runs on Linux without a sound card)."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _stub(name, **attrs):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod


_stub("sounddevice", default=types.SimpleNamespace(hostapi=0), query_hostapis=lambda *a: {"devices": []},
      query_devices=lambda *a: [])
_stub("keyboard", parse_hotkey=lambda s: s)
_stub("pyperclip", copy=lambda s: None, paste=lambda: "")
