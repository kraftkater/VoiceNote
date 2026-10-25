"""VoiceNote - native window (PySide6) around engine.py.

Press a global hotkey, speak, and the text is pasted into the active window (or saved as a note file).
The window is optional: closing it hides it to the system tray and the hotkeys keep working.
Every setting is saved automatically as soon as it changes.
"""
import copy
import html
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))  # the embeddable Python does not add the script folder to sys.path

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, QSize, QTimer, QUrl, Qt, Signal, Slot
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QGuiApplication, QIcon, QLinearGradient, QPainter,
                           QPen, QPixmap, QTextOption)
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QApplication, QButtonGroup, QComboBox, QFileDialog,
                               QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QInputDialog, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSystemTrayIcon, QToolButton, QVBoxLayout, QWidget)

import engine as E

SINGLE_INSTANCE_NAME = "VoiceNote-single-instance"
LANG_DIR = BASE / "lang"

# --------------------------------------------------------------------------- translations
# English is built in and used as the fallback. Other languages live in lang/<code>.json
# (add your own file with a "_name" entry to get a new language in the dropdown).
EN = {
    "st_ready": "Ready", "st_recording": "Recording", "st_processing": "Processing…",
    "st_loading": "Loading model", "st_downloading": "Downloading model", "st_error": "Model error",
    "sec_hotkeys": "Hotkeys", "hk_dictate": "Dictate", "hk_dictate_llm": "Dictate + AI cleanup",
    "hk_note": "Dictate to a file", "hk_note_llm": "Dictate to a file + AI cleanup", "hk_cancel": "Cancel recording",
    "hk_hint": "Click a button, then press the new key or combination. Press once to start recording, again to "
               "stop. The first two paste the text into the active window, the file keys save it as a note.",
    "hk_press": "Press keys…", "hk_unsupported": "Unsupported key",
    "sec_speech": "Speech recognition", "sp_language": "Language", "sp_auto": "Auto-detect", "sp_model": "Model",
    "sp_model_hint": "Bigger models are more accurate but slower. A model is downloaded once, the first time you use it.",
    "sp_mic": "Microphone", "sp_mic_default": "System default", "sp_mic_missing": "{name} (not found)",
    "sp_search_ph": "Search…", "sp_no_match": "No matches",
    "model_err": "Could not load the model: {err}",
    "sec_files": "Audio file → text",
    "files_hint": "Pick audio or video files (or a folder) or drop them onto this window. The text is saved to the notes folder next to the other notes. The model and language from “Speech recognition” are used. Subtitle timing is exact with Whisper models and approximate with the other engines.",
    "files_pick": "Choose files…", "files_pick_dir": "Choose folder…", "files_cancel": "Cancel", "files_open": "Open folder",
    "files_filter": "Audio and video", "files_srt": "Subtitles (.srt)", "files_vtt": "Subtitles (.vtt)",
    "files_progress": "File {i}/{n}: {name} — {stage}",
    "files_st_decode": "reading", "files_st_transcribe": "recognising", "files_st_write": "saving",
    "files_done": "Done: {n} file(s) saved to {folder}", "files_cancelled": "Cancelled.",
    "files_err_busy": "Wait until the current recording, loading or file job is finished.",
    "files_err_no_model": "Choose a speech model first.", "files_err_not_ready": "The speech model is still loading.",
    "files_err_no_files": "No audio or video files found.",
    "pack_installed_cpu": "Installed (CPU build — no GPU)",
    "pack_installed_arch": "Installed (this build does not support your GPU — CPU only)",
    "torch_arch": "The installed PyTorch build does not support your GPU model, so the GPU cannot be used. Pick Auto / CPU, or change the order of “torch_cuda_indexes” in sources.json and reinstall PyTorch.",
    "torch_cpu": "The installed PyTorch is the CPU-only build, so the GPU cannot be used. In “Extra engines” remove PyTorch and install it again (with an NVIDIA GPU the CUDA build is downloaded), or pick Auto / CPU.", "retry": "Retry",
    "dl_model": "Downloading {name}: {done} / {total}",
    "sec_notes": "Notes (text files)", "notes_folder": "Folder",
    "notes_folder_ph": "Default: the “notes” folder next to the program", "notes_browse": "Browse…",
    "notes_open": "Open", "notes_pick": "Choose a folder for notes", "notes_prefix": "File name start",
    "notes_hint": "Each recording creates a new file: {a}, {b}, …",
    "sec_llm": "AI text cleanup", "llm_provider": "Provider", "llm_custom": "Custom", "llm_url": "Base URL",
    "llm_key": "API key", "llm_key_ph": "optional", "llm_model": "Model", "llm_none": "(none: use raw text)",
    "llm_refresh": "Refresh model list", "llm_test": "Test cleanup",
    "llm_test_hint": "Pick a model and press Test to check the connection.",
    "llm_looking": "Looking for models…", "llm_connected": "Connected · {n} model(s) available",
    "llm_connected_empty": "Connected, but no models installed yet.", "llm_unreachable": "Not reachable: {err}.",
    "llm_help_ollama": "Is Ollama installed and running? <a href=\"https://ollama.com/download\">Get Ollama</a>",
    "llm_help_lmstudio": "Start the LM Studio local server (Developer tab).",
    "llm_testing": "Testing…", "llm_test_failed": "Test failed: {err}",
    "dl_title": "Download a model (via Ollama)", "dl_ph": "e.g. qwen2.5:3b", "dl_btn": "Download",
    "dl_progress": "{name}: {status}", "dl_progress_pct": "{name}: {status} · {pct}% ({done} / {total})",
    "dl_failed": "Download failed: {err}", "dl_done": "{name} downloaded and selected.",
    "dl_busy": "A download is already running.",
    "prompt_title": "Cleanup prompt", "prompt_reset": "Reset to default",
    "sec_options": "Options", "opt_ui_lang": "Interface language", "opt_theme": "Theme", "theme_system": "System",
    "theme_light": "Light", "theme_dark": "Dark", "opt_beep": "Sound signals",
    "opt_restore": "Restore clipboard after pasting", "opt_space": "Add a space after the text",
    "sec_last": "Last result", "last_empty": "Nothing yet. Press your hotkey and speak.", "sec_log": "Log",
    "footer": "Settings are saved automatically. Closing this window keeps VoiceNote running in the system tray; "
              "use Quit to exit completely.",
    "btn_quit": "Quit", "saved": "Saved ✓", "save_error": "Not saved: {err}", "quit_title": "Quit VoiceNote",
    "quit_confirm": "Quit VoiceNote? Hotkeys will stop working and the speech model will be unloaded from memory.",
    "tray_open": "Open", "tray_quit": "Quit", "tray_running": "VoiceNote is still running in the tray.",
    "sp_model_choose": "— Choose a model —",
    "sp_model_none_hint": "No model selected yet. Pick one above: it is downloaded once and then works offline.",
    "sp_model_link_tip": "Open the model page on Hugging Face",
    "sp_device": "Processor", "dev_auto": "Auto", "dev_cpu": "CPU", "dev_gpu": "GPU (NVIDIA)",
    "dev_hint": "Auto uses an NVIDIA GPU when there is one and falls back to the CPU. The GPU makes recognition "
                "several times faster, but it needs “GPU support”, which is installed in the “Extra engines” section "
                "(~1 GB). AMD and Intel GPUs are not supported by this engine.",
    "dev_need_pack": "To use the GPU, install “GPU support” in the “Extra engines” section.",
    "dev_running": "Speech model runs on: {dev}", "dev_note_no_nvidia": "No NVIDIA GPU found, so the CPU is used.",
    "dev_note_unavail": "The GPU cannot be used yet ({err}). The CPU is used instead.",
    "add_title": "Add a model from Hugging Face", "add_ph": "https://huggingface.co/owner/model", "add_btn": "Add",
    "add_hint": "Works with Whisper models in the CTranslate2 (faster-whisper) format, SenseVoice (sherpa-onnx) and "
                "Qwen3-ASR; the last two need their engine from the “Extra engines” section. Other formats are "
                "recognised and reported as unsupported.",
    "add_busy": "Checking the link…", "add_ok": "Added {name}. It is selected and will be downloaded.",
    "add_exists": "{name} is already in the list. Selected.",
    "add_bad": "That does not look like a Hugging Face model link.",
    "add_gguf_single": "Note: there is no separate audio encoder (mmproj) in this repository. It works only if the encoder is built into the GGUF and llama.cpp supports this model; files made for other programs (for example CrispASR) will not start.",
    "sp_variant": "Model file (quantization)",
    "choose_title": "Choose a model file",
    "choose_label": "This repository has several model files (quantizations). Smaller files download faster and use less memory; larger ones are more accurate. Which one should be downloaded? You can change it later in “Model file”.",
    "choose_recommended": "recommended", "choose_pending": "Choose which model file to download…",
    "add_nf": "{name} was not found (or it is private).", "add_unsupported": "Not supported: {detail}",
    "add_net": "Could not check the link: {err}",
    "hk_continue": "Continue",
    "hk_continue_tip": "On: the file hotkeys keep writing into the last note file. The dictation hotkeys remember the "
                       "window where you first used them and keep sending text there, so you do not have to click "
                       "into it again. Switch it with Alt + the hotkey.",
    "hk_alt_hint": "Alt + a hotkey switches its “Continue” tick on or off (a short double beep confirms it).",
    "llm_device": "Processor", "llm_dev_auto": "Auto", "llm_dev_cpu": "CPU only",
    "llm_dev_hint": "Ollama only. Auto: Ollama decides (it uses the GPU when it can). LM Studio chooses in its own "
                    "settings.",
    "llm_where_gpu": "the model runs on the GPU", "llm_where_cpu": "the model runs on the CPU",
    "llm_where_mixed": "the model is split between GPU and CPU",
    "opt_max_len": "Maximum recording length", "len_unlimited": "No limit", "len_hours": "{n} h",
    "opt_keep_audio": "Keep a WAV copy of every recording", "st_processing_n": "Processing… {done}/{total}",
    "need_model_msg": "Choose a speech model first (Speech recognition section).",
    "dl_paused": "Paused · {name}: {done} / {total}", "dl_pause": "Pause", "dl_resume": "Resume",
    "dl_stop": "Stop and discard the download", "st_paused": "Download paused",
    "model_remove_tip": "Delete this model from the disk (a model added from a link also leaves the list)",
    "model_delete_confirm": "Delete the downloaded files of {name}? It stays in the list and can be downloaded again.",
    "pack_gpu": "GPU support: makes speech recognition several times faster (NVIDIA CUDA libraries, ~1 GB)",
    "pack_resolving": "Looking up packages…",
    "pack_downloading": "Downloading {pct}% ({done} / {total})", "pack_unpacking": "Installing…", "pack_building": "Preparing package {name} ({n}/{m})…",
    "pack_sources": "Open the download sources file (sources.json)",
    "llm_dev_gpu": "GPU (all layers)", "dev_gpu_info": "NVIDIA GPU: {name} (driver {driver})",
    "dev_driver_old": "The NVIDIA driver is too old for CUDA 12: please update it.",
    "model_remove_confirm": "Remove {name} from the list and delete its files?",
    "model_remove_busy": "Stop recording first.", "dev_restart": "Restart VoiceNote to try the GPU again.",
    "sec_engines": "Extra engines (optional)",
    "pack_hint": "Only needed for the matching models in the list. Each engine is installed into the “packs” folder next "
                 "to the program, separately from the main libraries, so it cannot break them. You can remove it here "
                 "at any time.",
    "pack_group_stable": "Ready to use",
    "pack_group_stable_hint": "Tested and working. GPU support makes Whisper several times faster (NVIDIA only). "
                              "SenseVoice is a small, fast engine for Chinese, English, Japanese, Korean and Cantonese.",
    "pack_group_exp": "Experimental",
    "pack_group_exp_hint": "Less tested: they may fail or behave differently on your PC. Only the models listed here "
                           "are supported for now. PyTorch (large) runs Qwen3-ASR. llama.cpp (small, no PyTorch needed) "
                           "runs audio models in GGUF format, e.g. Qwen3-ASR or Voxtral: add the model with a Hugging "
                           "Face link in “Speech recognition”.",
    "pack_llamacpp": "llama.cpp engine for GGUF audio models (small; GPU build is picked automatically)",
    "pack_sensevoice": "SenseVoice engine (sherpa-onnx, small)",
    "pack_pytorch": "PyTorch engine for Qwen3-ASR (large, several GB; CUDA build picked for your GPU)",
    "pack_installed": "Installed", "pack_not_installed": "Not installed", "pack_install": "Install",
    "pack_remove": "Remove", "pack_installing": "Installing… this can take several minutes",
    "pack_failed": "Installation failed: {err}",
    "pack_needed": "This model needs “{name}”. Install it in the “Extra engines” section.",
    "pack_remove_confirm": "Remove “{name}” from this folder?",
    "lang_warn": "This model supports only: {langs}. Other languages will not be recognised well.",
}


class Translator:
    def __init__(self):
        self.code, self.table = "en", {}

    @staticmethod
    def available():
        """[(code, native name)] - English first, then every lang/<code>.json (files starting with _ are data)."""
        out = [("en", "English")]
        for f in sorted(LANG_DIR.glob("*.json")):
            if f.stem.startswith("_"):
                continue
            try:
                out.append((f.stem, json.loads(f.read_text(encoding="utf-8")).get("_name", f.stem)))
            except (OSError, ValueError):
                pass
        return out

    def set(self, code):
        self.code, self.table = "en", {}
        if code != "en":
            try:
                self.table = json.loads((LANG_DIR / f"{code}.json").read_text(encoding="utf-8"))
                self.code = code
            except (OSError, ValueError):
                pass

    def __call__(self, key, **kw):
        text = self.table.get(key) or EN.get(key, key)
        try:
            return text.format(**kw) if kw else text
        except (KeyError, IndexError):
            return text


tr = Translator()


def load_native_names():
    try:
        return json.loads((LANG_DIR / "_native.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


NATIVE = load_native_names()  # language code -> name in that language ("ru" -> "Русский")


def filter_items(items, query):
    """items: [(code, display, native, search_keys)]. Match the beginning of any word first, then any substring."""
    q = query.strip().lower()
    if not q:
        return list(items)
    starts, contains = [], []
    for it in items:
        keys = it[3]
        if q == it[0] or any(k.startswith(q) or any(w.startswith(q) for w in re.split(r"[\s\-_/(),]+", k)) for k in keys):
            starts.append(it)
        elif any(q in k for k in keys):
            contains.append(it)
    return starts + contains


def fmt_size(n):
    return f"{n / 1073741824:.1f} GB" if n >= 1073741824 else f"{n / 1048576:.0f} MB"


# --------------------------------------------------------------------------- theme
LIGHT = dict(bg="#f4f5f9", card="#ffffff", text="#1b1d27", muted="#6b7084", line="#e3e5ee", field="#f7f8fc",
             accent="#6d5efc", accent2="#8f84ff", ok="#1fa971", warn="#e8a317", bad="#e5484d", bad_h="#cf3a3f",
             hover="#eceefa")
DARK = dict(bg="#101118", card="#181a24", text="#eceef7", muted="#8b90a7", line="#272a38", field="#1f2230",
            accent="#7a6cff", accent2="#9a90ff", ok="#2fc58a", warn="#f0b429", bad="#f0585d", bad_h="#d94a4f",
            hover="#282c3e")
P = dict(LIGHT)  # current palette


def build_qss(p):
    return f"""
    QWidget {{ color:{p['text']}; font-family:"Segoe UI","Noto Sans",sans-serif; font-size:13px; }}
    QMainWindow, #root, #body, QScrollArea {{ background:{p['bg']}; border:0; }}
    #card {{ background:{p['card']}; border:1px solid {p['line']}; border-radius:14px; }}
    #topbar {{ background:{p['bg']}; border-bottom:1px solid {p['line']}; }}
    #card QLabel, #card QCheckBox {{ background:transparent; }}
    QLabel#h2 {{ color:{p['muted']}; font-size:11px; font-weight:700; }}
    QLabel#hint {{ color:{p['muted']}; font-size:12px; }}
    QLabel#title {{ font-size:19px; font-weight:700; }}
    QLabel[kind="ok"] {{ color:{p['ok']}; }}
    QLabel[kind="bad"] {{ color:{p['bad']}; }}
    QLabel[kind="muted"] {{ color:{p['muted']}; }}
    QComboBox, QLineEdit, QPlainTextEdit, QPushButton#picker {{ background:{p['field']};
        border:1px solid {p['line']}; border-radius:9px; padding:6px 10px;
        selection-background-color:{p['accent']}; selection-color:white; }}
    QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus, QPushButton#picker:hover {{ border-color:{p['accent']}; }}
    QComboBox::drop-down {{ border:0; width:26px; }}
    QComboBox QAbstractItemView {{ background:{p['card']}; border:1px solid {p['line']};
        selection-background-color:{p['accent']}; selection-color:white; outline:0; }}
    QPushButton {{ background:{p['field']}; border:1px solid {p['line']}; border-radius:9px; padding:7px 14px; }}
    QComboBox:hover, QLineEdit:hover, QPlainTextEdit:hover {{ border-color:{p['accent']}; }}
    QComboBox[warn="true"] {{ border:1px solid {p['bad']}; }}
    QPushButton:hover {{ border-color:{p['accent']}; background:{p['hover']}; }}
    QPushButton:pressed {{ background:{p['line']}; }}
    QPushButton:disabled {{ color:{p['muted']}; }}
    QPushButton:disabled:hover {{ border-color:{p['line']}; background:{p['field']}; }}
    QPushButton#ghost {{ background:transparent; }}
    QPushButton#ghost:hover {{ background:{p['hover']}; }}
    QPushButton#danger {{ background:{p['bad']}; color:white; border:0; font-weight:600; padding:8px 22px; }}
    QPushButton#danger:hover {{ background:{p['bad_h']}; border:0; }}
    QPushButton#cardhead {{ background:transparent; border:0; text-align:left; padding:4px 0; color:{p['muted']};
        font-size:11px; font-weight:700; }}
    QPushButton#cardhead:hover {{ color:{p['text']}; background:transparent; }}
    QPushButton#key {{ font-family:"Cascadia Mono",Consolas,monospace; font-weight:700; min-width:150px; }}
    QPushButton#key[cap="true"] {{ border:1px solid {p['accent']}; color:{p['accent']}; }}
    QPushButton#seg {{ border:0; background:transparent; color:{p['muted']}; padding:6px 14px; border-radius:8px; }}
    QPushButton#seg:hover {{ background:{p['hover']}; color:{p['text']}; border:0; }}
    QPushButton#seg:checked, QPushButton#seg:checked:hover {{ background:{p['card']}; color:{p['text']}; font-weight:600; }}
    QPushButton#refresh {{ padding:6px 10px; }}
    #segwrap {{ background:{p['field']}; border:1px solid {p['line']}; border-radius:11px; }}
    QProgressBar {{ background:{p['line']}; border:0; border-radius:2px; max-height:4px; min-height:4px; }}
    QProgressBar::chunk {{ background:{p['accent']}; border-radius:2px; }}
    #pill {{ background:{p['card']}; border:1px solid {p['line']}; border-radius:14px; padding:4px 12px;
        font-weight:600; }}
    #bottom {{ background:{p['card']}; border-top:1px solid {p['line']}; }}
    #sep {{ background:{p['line']}; max-height:1px; min-height:1px; }}
    #popup {{ background:{p['card']}; border:1px solid {p['line']}; border-radius:10px; }}
    QListWidget {{ background:transparent; border:0; outline:0; }}
    QListWidget::item {{ padding:6px 8px; border-radius:6px; }}
    QListWidget::item:selected {{ background:{p['accent']}; color:white; }}
    QToolButton#disc {{ border:0; background:transparent; color:{p['text']}; font-weight:600; padding:2px 0; }}
    QToolButton#disc:hover {{ color:{p['accent']}; }}
    QScrollBar:vertical {{ width:10px; background:transparent; margin:0; }}
    QScrollBar::handle:vertical {{ background:{p['line']}; border-radius:5px; min-height:30px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
    QToolTip {{ background:{p['card']}; color:{p['text']}; border:1px solid {p['line']}; }}
    QMessageBox, QDialog {{ background:{p['card']}; }}
    """


def set_prop(w, name, value):
    w.setProperty(name, value)
    w.style().unpolish(w)
    w.style().polish(w)


def make_icon(c1, c2):
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    g = QLinearGradient(0, 0, 64, 64)
    g.setColorAt(0, QColor(c1))
    g.setColorAt(1, QColor(c2))
    p.setBrush(g)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 16, 16)
    p.setBrush(QColor("white"))
    p.drawRoundedRect(25, 12, 14, 25, 7, 7)
    pen = QPen(QColor("white"), 3.5)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawArc(18, 21, 28, 24, 180 * 16, 180 * 16)
    p.drawLine(32, 45, 32, 51)
    p.drawLine(26, 51, 38, 51)
    p.end()
    return QIcon(pm)


# --------------------------------------------------------------------------- small widgets
class Toggle(QAbstractButton):
    """iOS-style on/off switch."""

    def __init__(self):
        super().__init__()
        self.setCheckable(True)
        self.setFixedSize(40, 22)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover, True)  # repaint on mouse enter/leave

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        on, hov = self.isChecked(), self.underMouse()
        p.setBrush(QColor((P["accent2"] if hov else P["accent"]) if on else (P["muted"] if hov else P["line"])))
        p.drawRoundedRect(0, 0, 40, 22, 11, 11)
        p.setBrush(QColor("white"))
        p.drawEllipse(21 if self.isChecked() else 3, 3, 16, 16)


class Tick(QAbstractButton):
    """A check box with a drawn tick (looks the same in both themes) and a text label to its right."""

    def __init__(self):
        super().__init__()
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover, True)

    def sizeHint(self):
        return QSize(26 + self.fontMetrics().horizontalAdvance(self.text()), 22)

    def setText(self, t):
        super().setText(t)
        self.updateGeometry()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        on, hov = self.isChecked(), self.underMouse()
        box = QRectF(1, 3, 16, 16)
        p.setBrush(QColor(P["accent"]) if on else QColor(P["field"]))
        p.setPen(QPen(QColor(P["accent"] if (on or hov) else P["muted"]), 1.4))
        p.drawRoundedRect(box, 4, 4)
        if on:
            pen = QPen(QColor("white"), 2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline([QPointF(4.5, 11.5), QPointF(7.5, 14.5), QPointF(13.5, 7.5)])
        p.setPen(QColor(P["text"] if hov else P["muted"]))
        p.drawText(QRectF(24, 0, self.width() - 24, self.height()), self.text(), QTextOption(Qt.AlignVCenter | Qt.AlignLeft))


class NoWheelCombo(QComboBox):
    """Combo box that ignores the mouse wheel, so scrolling the settings page never changes a value by accident."""

    def wheelEvent(self, ev):
        ev.ignore()  # the event propagates to the scroll area


class PickerPopup(QFrame):
    """Drop-down with a search box. Typing the first letters of any word of a language name filters the list."""
    chosen = Signal(str)

    def __init__(self, items, current, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setObjectName("popup")
        self.items, self.current = items, current
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(tr("sp_search_ph"))
        self.edit.setClearButtonEnabled(True)
        self.list = QListWidget()
        lay.addWidget(self.edit)
        lay.addWidget(self.list, 1)
        self.edit.textChanged.connect(self.refill)
        self.list.itemClicked.connect(self.pick)
        self.edit.installEventFilter(self)
        self.refill("")

    def refill(self, query):
        self.list.clear()
        found = filter_items(self.items, query)
        for code, disp, native, _keys in found:
            it = QListWidgetItem(disp + (f"   ·   {native}" if native and native != disp else ""))
            it.setData(Qt.UserRole, code)
            self.list.addItem(it)
            if code == self.current and not query.strip():
                self.list.setCurrentItem(it)
        if not found:
            it = QListWidgetItem(tr("sp_no_match"))
            it.setFlags(Qt.NoItemFlags)
            self.list.addItem(it)
        elif self.list.currentItem() is None:
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QEvent.KeyPress:
            k = ev.key()
            if k in (Qt.Key_Down, Qt.Key_Up) and self.list.count():
                row = self.list.currentRow() + (1 if k == Qt.Key_Down else -1)
                self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
                return True
            if k in (Qt.Key_Return, Qt.Key_Enter):
                self.pick(self.list.currentItem())
                return True
        return super().eventFilter(obj, ev)

    def pick(self, item):
        code = item.data(Qt.UserRole) if item is not None else None
        if code:
            self.chosen.emit(code)
            self.close()

    def show_under(self, widget):
        self.setFixedSize(max(widget.width(), 320), 340)
        self.move(widget.mapToGlobal(QPoint(0, widget.height() + 2)))
        self.show()
        self.edit.setFocus()
        cur = self.list.currentItem()
        if cur is not None:
            self.list.scrollToItem(cur, QAbstractItemView.PositionAtCenter)


class SearchPicker(QPushButton):
    """Looks like a combo box; opens a popup with a search field."""
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("picker")
        self.setMinimumHeight(34)
        self.items, self.code, self._popup = [], "", None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        self.text_lb, arrow = QLabel(), QLabel("▾")
        for lb in (self.text_lb, arrow):
            lb.setAttribute(Qt.WA_TransparentForMouseEvents)
            lb.setStyleSheet("background:transparent;")
        lay.addWidget(self.text_lb, 1)
        lay.addWidget(arrow)
        self.clicked.connect(self.open_popup)

    def set_items(self, items):
        self.items = items
        self._show_current()

    def set_code(self, code):
        self.code = code
        self._show_current()

    def _show_current(self):
        for code, disp, *_ in self.items:
            if code == self.code:
                self.text_lb.setText(disp)
                return
        self.text_lb.setText(self.code)

    def open_popup(self):
        self._popup = PickerPopup(self.items, self.code, self)
        self._popup.chosen.connect(self._on_chosen)
        self._popup.show_under(self)

    def _on_chosen(self, code):
        self.code = code
        self._show_current()
        self.changed.emit()


# Windows virtual-key codes -> names understood by the `keyboard` library. Using VK codes keeps
# hotkeys independent of the keyboard layout (e.g. the Russian layout).
VK = {0x20: "space", 0x0D: "enter", 0x1B: "esc", 0x09: "tab", 0x08: "backspace", 0x2E: "delete", 0x2D: "insert",
      0x24: "home", 0x23: "end", 0x21: "page up", 0x22: "page down", 0x26: "up", 0x28: "down", 0x25: "left",
      0x27: "right", 0xBD: "-", 0xBB: "=", 0xBC: ",", 0xBE: ".", 0xBF: "/", 0xDC: "\\", 0xBA: ";", 0xDE: "'",
      0xDB: "[", 0xDD: "]", 0xC0: "`", 0x13: "pause", 0x91: "scroll lock"}
MOD_KEYS = (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta, Qt.Key_AltGr)


def key_name(ev):
    vk = ev.nativeVirtualKey()
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return chr(vk).lower()
    if 0x70 <= vk <= 0x87:
        return f"f{vk - 0x6F}"
    if vk in VK:
        return VK[vk]
    k = ev.key()  # fallback (non-Windows / unknown VK)
    if Qt.Key_A <= k <= Qt.Key_Z or Qt.Key_0 <= k <= Qt.Key_9:
        return chr(k).lower()
    if Qt.Key_F1 <= k <= Qt.Key_F24:
        return f"f{k - Qt.Key_F1 + 1}"
    return {Qt.Key_Space: "space", Qt.Key_Return: "enter", Qt.Key_Enter: "enter", Qt.Key_Escape: "esc",
            Qt.Key_Tab: "tab", Qt.Key_Backspace: "backspace", Qt.Key_Delete: "delete", Qt.Key_Insert: "insert",
            Qt.Key_Home: "home", Qt.Key_End: "end", Qt.Key_PageUp: "page up", Qt.Key_PageDown: "page down",
            Qt.Key_Up: "up", Qt.Key_Down: "down", Qt.Key_Left: "left", Qt.Key_Right: "right"}.get(k)


def pretty(hk):
    return " + ".join(p.upper() if len(p) == 1 else p.title() for p in hk.split("+")) if hk else ""


def mods_of(ev):
    m, out = ev.modifiers(), []
    for flag, name in ((Qt.ControlModifier, "ctrl"), (Qt.AltModifier, "alt"), (Qt.ShiftModifier, "shift"),
                       (Qt.MetaModifier, "windows")):
        if m & flag:
            out.append(name)
    return out


class HotkeyButton(QPushButton):
    changed = Signal()
    capturing_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("key")
        self.value, self.capturing = "", False
        self.clicked.connect(self.begin)

    def set_value(self, v):
        self.value = v
        self.setText(pretty(v))

    def begin(self):
        if self.capturing:
            return
        self.capturing = True
        set_prop(self, "cap", True)
        self.setText(tr("hk_press"))
        self.capturing_changed.emit(True)
        self.setFocus()

    def end(self):
        self.capturing = False
        set_prop(self, "cap", False)
        self.setText(pretty(self.value))
        self.capturing_changed.emit(False)

    def keyPressEvent(self, ev):
        if not self.capturing:
            return super().keyPressEvent(ev)
        if ev.key() in MOD_KEYS:  # live preview while modifiers are held
            self.setText(" + ".join([pretty(m) for m in mods_of(ev)] + ["…"]))
            return
        name = key_name(ev)
        if not name:
            self.setText(tr("hk_unsupported"))
            return
        self.value = "+".join(mods_of(ev) + [name])
        self.end()
        self.changed.emit()
        self.clearFocus()

    def focusOutEvent(self, ev):
        if self.capturing:
            self.end()
        super().focusOutEvent(ev)


class Bridge(QObject):
    """Runs a function in a worker thread and delivers its result on the GUI thread."""
    done = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.done.connect(self._deliver)

    @Slot(object, object)
    def _deliver(self, cb, result):
        cb(result)

    def run(self, fn, cb):
        threading.Thread(target=lambda: self.done.emit(cb, fn()), daemon=True).start()


# --------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, eng, app):
        super().__init__()
        self.eng, self.app, self.bridge = eng, app, Bridge()
        self.S = copy.deepcopy(eng.settings)
        self.filling = True
        self._tr_items, self._req, self._pull_active, self._last_icon, self._flash = [], 0, None, None, 0
        self._add_seen, self._need_seen, self._rev_seen, self._sel_seen, self._pack_seen = None, 0.0, 0, 0, ""
        self._cards = {}  # card id -> {head, body, collapsed}
        self.setWindowTitle("VoiceNote")
        self.resize(700, 880)   # the real size is set after the form is built (see fit_window)
        self.setMinimumWidth(540)
        self.icon_idle, self.icon_rec = make_icon("#6d5efc", "#8f84ff"), make_icon("#e5484d", "#ff7a7f")
        self.setWindowIcon(self.icon_idle)
        self.save_timer = QTimer(self)  # autosave: runs shortly after the last change
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(500)
        self.save_timer.timeout.connect(self.autosave)
        self._build()
        self._build_tray()
        self.apply_theme()
        self.fill()
        self.retranslate()
        self.filling = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(400)
        self.refresh_models()
        self.fit_window()
        self.poll()

    # ---- construction helpers
    def fit_window(self):
        """Window size: never narrower than the form needs (it used to cut blocks off); the last size is restored."""
        need = self._body.minimumSizeHint().width() + 30   # + scroll bar and frame
        scr = self.screen().availableGeometry() if self.screen() else None
        if scr:
            need = min(need, int(scr.width() * 0.95))
        self.setMinimumWidth(max(540, need))
        ui = self.S["ui"]
        w = int(ui.get("width") or 0)
        h = int(ui.get("height") or 0)
        w = max(w, need) if w else max(720, need)
        h = h if h >= 480 else 880
        if scr:
            w, h = min(w, scr.width()), min(h, scr.height())
        self.resize(w, h)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not self.filling and self.isVisible() and not self.isMaximized():
            self.mark()   # remember the size (autosave)

    def reg(self, widget, key, setter="setText"):
        self._tr_items.append((widget, key, setter))
        return widget

    def label(self, key=None, name=None):
        lb = QLabel()
        if name:
            lb.setObjectName(name)
        if key:
            self.reg(lb, key)
        return lb

    def card(self, key, into=None):
        """A collapsible section: click the title to fold it away (the cards below move up). Returns the body layout."""
        fr = QFrame()
        fr.setObjectName("card")
        outer = QVBoxLayout(fr)
        outer.setContentsMargins(16, 10, 16, 14)
        outer.setSpacing(0)
        head = QPushButton()
        head.setObjectName("cardhead")
        head.setFlat(True)
        head.setCursor(Qt.PointingHandCursor)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)
        outer.addWidget(head)
        outer.addWidget(body)
        self._cards[key] = {"head": head, "body": body, "collapsed": False}
        head.clicked.connect(lambda _=False, k=key: self.toggle_card(k))
        (into or self.body_l).addWidget(fr)
        return lay

    def update_head(self, key):
        c = self._cards[key]
        c["head"].setText(("▸  " if c["collapsed"] else "▾  ") + tr(key).upper())

    def set_collapsed(self, key, on):
        c = self._cards[key]
        c["collapsed"] = on
        c["body"].setVisible(not on)
        self.update_head(key)

    def toggle_card(self, key):
        self.set_collapsed(key, not self._cards[key]["collapsed"])
        self.mark()  # the folded/unfolded state is remembered (autosave)

    def row(self, lay, key, widget, constrain=True):
        r = QHBoxLayout()
        r.setSpacing(12)
        r.addWidget(self.label(key))
        r.addStretch(1)
        if constrain:
            widget.setMinimumWidth(260)
            widget.setMaximumWidth(300)
        r.addWidget(widget)
        lay.addLayout(r)
        return r

    def hint(self, key=None):
        lb = self.label(key, "hint")
        lb.setWordWrap(True)
        return lb

    def status_label(self):
        lb = QLabel()
        lb.setWordWrap(True)
        lb.setTextFormat(Qt.RichText)
        lb.setOpenExternalLinks(True)
        lb.setProperty("kind", "muted")
        lb.setVisible(False)
        return lb

    def set_status(self, lb, text, kind="muted", rich=False):
        new = text if rich else html.escape(text)
        if lb.text() == new and lb.property("kind") == kind:
            return  # unchanged: avoid needless restyling on every poll
        lb.setText(new)
        set_prop(lb, "kind", kind)
        lb.setVisible(bool(text))

    def mark(self, *_):
        """Something changed: save automatically in a moment."""
        if not self.filling:
            self.save_timer.start()

    # ---- UI construction
    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        # fixed top bar (always visible): logo, recording timer / model status, progress and the last result
        top = QFrame()
        top.setObjectName("topbar")
        tl = QVBoxLayout(top)
        tl.setContentsMargins(18, 12, 18, 10)
        tl.setSpacing(8)
        v.addWidget(top)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("body")
        self.body_l = QVBoxLayout(body)
        self.body_l.setContentsMargins(18, 12, 18, 16)
        self.body_l.setSpacing(12)
        scroll.setWidget(body)
        self._body = body
        v.addWidget(scroll, 1)

        h = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(self.icon_idle.pixmap(30, 30))
        t = QLabel("VoiceNote")
        t.setObjectName("title")
        self.pill = QLabel()
        self.pill.setObjectName("pill")
        self.pill.setTextFormat(Qt.RichText)
        h.addWidget(logo)
        h.addWidget(t)
        h.addStretch(1)
        h.addWidget(self.pill)
        tl.addLayout(h)
        self.model_bar = QProgressBar()
        self.model_bar.setTextVisible(False)
        self.model_bar.setRange(0, 100)
        self.model_bar.setVisible(False)
        tl.addWidget(self.model_bar)
        lay = self.card("sec_last", into=tl)
        self.last = QLabel()
        self.last.setWordWrap(True)
        self.last.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.last.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.last_scroll = QScrollArea()          # a long text must not push the rest of the window away
        self.last_scroll.setWidgetResizable(True)
        self.last_scroll.setFrameShape(QFrame.NoFrame)
        self.last_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.last_scroll.setMaximumHeight(84)
        self.last_scroll.setWidget(self.last)
        lay.addWidget(self.last_scroll)

        # hotkeys
        lay = self.card("sec_hotkeys")
        self.hk, self.ticks = {}, {}
        for name, key in (("dictate", "hk_dictate"), ("dictate_llm", "hk_dictate_llm"), ("note", "hk_note"),
                          ("note_llm", "hk_note_llm"), ("cancel", "hk_cancel")):
            b = HotkeyButton()
            b.changed.connect(self.mark)
            b.capturing_changed.connect(self.on_capture)
            self.hk[name] = b
            r = self.row(lay, key, b, False)
            if name in self.S["append"]:
                t = Tick()
                self.reg(t, "hk_continue")
                self.reg(t, "hk_continue_tip", "setToolTip")
                t.toggled.connect(self.mark)
                self.ticks[name] = t
                r.insertWidget(2, t)  # label, stretch, tick, key
                r.insertSpacing(3, 6)
        lay.addWidget(self.hint("hk_hint"))
        lay.addWidget(self.hint("hk_alt_hint"))
        self.hk_err = self.status_label()
        lay.addWidget(self.hk_err)

        # speech
        lay = self.card("sec_speech")
        self.pick_lang, self.cb_model, self.cb_mic = SearchPicker(), NoWheelCombo(), NoWheelCombo()
        self.row(lay, "sp_language", self.pick_lang)
        mw = QWidget()
        ml = QHBoxLayout(mw)
        ml.setContentsMargins(0, 0, 0, 0)
        self.btn_modellink = QPushButton("↗")
        self.btn_modellink.setObjectName("refresh")
        self.reg(self.btn_modellink, "sp_model_link_tip", "setToolTip")
        self.btn_modellink.clicked.connect(self.open_model_page)
        self.btn_modeldel = QPushButton("✕")
        self.btn_modeldel.setObjectName("refresh")
        self.reg(self.btn_modeldel, "model_remove_tip", "setToolTip")
        self.btn_modeldel.clicked.connect(self.remove_model)
        ml.addWidget(self.cb_model, 1)
        ml.addWidget(self.btn_modellink)
        ml.addWidget(self.btn_modeldel)
        self.row(lay, "sp_model", mw)
        lay.addWidget(self.hint("sp_model_hint"))
        self.model_none = self.status_label()
        lay.addWidget(self.model_none)
        self.variant_box = QWidget()      # model file (quantization) of GGUF repositories that have several
        vb = QHBoxLayout(self.variant_box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(12)
        vb.addWidget(self.label("sp_variant"))
        vb.addStretch(1)
        self.cb_variant = NoWheelCombo()
        self.cb_variant.setMinimumWidth(260)
        self.cb_variant.setMaximumWidth(300)
        self.cb_variant.currentIndexChanged.connect(self.on_variant)
        vb.addWidget(self.cb_variant)
        self.variant_box.hide()
        lay.addWidget(self.variant_box)
        self.dl_row = QWidget()
        dr = QHBoxLayout(self.dl_row)
        dr.setContentsMargins(0, 0, 0, 0)
        self.model_dl = self.status_label()
        self.btn_dlpause, self.btn_dlstop = QPushButton("⏸"), QPushButton("■")
        for b in (self.btn_dlpause, self.btn_dlstop):
            b.setObjectName("refresh")
        self.reg(self.btn_dlstop, "dl_stop", "setToolTip")
        self.btn_dlpause.clicked.connect(self.toggle_dl_pause)
        self.btn_dlstop.clicked.connect(lambda *_: self.eng.stop_download())
        dr.addWidget(self.model_dl, 1)
        dr.addWidget(self.btn_dlpause)
        dr.addWidget(self.btn_dlstop)
        self.dl_row.setVisible(False)
        lay.addWidget(self.dl_row)
        self.model_err = self.status_label()
        self.model_err.linkActivated.connect(lambda _: self.eng.ensure_model())
        lay.addWidget(self.model_err)

        # add a model by Hugging Face link
        t = QLabel()
        t.setStyleSheet("font-weight:600;")
        self.reg(t, "add_title")
        lay.addWidget(t)
        ar = QHBoxLayout()
        self.ed_add = QLineEdit()
        self.reg(self.ed_add, "add_ph", "setPlaceholderText")
        self.btn_add = QPushButton()
        self.reg(self.btn_add, "add_btn")
        self.btn_add.clicked.connect(self.add_model)
        self.ed_add.returnPressed.connect(self.add_model)
        ar.addWidget(self.ed_add, 1)
        ar.addWidget(self.btn_add)
        lay.addLayout(ar)
        lay.addWidget(self.hint("add_hint"))
        self.add_status = self.status_label()
        lay.addWidget(self.add_status)

        # processor
        self.cb_dev = NoWheelCombo()
        for code, key in (("auto", "dev_auto"), ("cpu", "dev_cpu"), ("cuda", "dev_gpu")):
            self.cb_dev.addItem("", code)
        self.row(lay, "sp_device", self.cb_dev)
        lay.addWidget(self.hint("dev_hint"))
        self.dev_status = self.status_label()
        lay.addWidget(self.dev_status)
        self.row(lay, "sp_mic", self.cb_mic)
        self.pick_lang.changed.connect(self.mark)
        for c in (self.cb_model, self.cb_mic, self.cb_dev):
            c.currentIndexChanged.connect(self.mark)
        self.cb_model.currentIndexChanged.connect(self.on_model_changed)

        # LLM
        lay = self.card("sec_llm")
        wrap = QFrame()
        wrap.setObjectName("segwrap")
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(3, 3, 3, 3)
        wl.setSpacing(2)
        self.seg = QButtonGroup(self)
        self.seg_btn = {}
        for pid, text, key in (("ollama", "Ollama", None), ("lmstudio", "LM Studio", None), ("custom", "", "llm_custom")):
            b = QPushButton(text)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            if key:
                self.reg(b, key)
            self.seg.addButton(b)
            self.seg_btn[pid] = b
            wl.addWidget(b)
            b.clicked.connect(lambda _=False, p=pid: self.set_provider(p, True))
        self.row(lay, "llm_provider", wrap, False)
        self.custom_box = QWidget()
        cl = QVBoxLayout(self.custom_box)
        cl.setContentsMargins(0, 0, 0, 0)
        self.ed_url, self.ed_key = QLineEdit(), QLineEdit()
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.reg(self.ed_key, "llm_key_ph", "setPlaceholderText")
        self.ed_url.setPlaceholderText("https://host/v1")
        self.row(cl, "llm_url", self.ed_url)
        self.row(cl, "llm_key", self.ed_key)
        lay.addWidget(self.custom_box)
        for e in (self.ed_url, self.ed_key):
            e.textEdited.connect(self.mark)
        self.cb_llm = NoWheelCombo()
        self.cb_llm.currentIndexChanged.connect(self.mark)
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setObjectName("refresh")
        self.reg(self.btn_refresh, "llm_refresh", "setToolTip")
        self.btn_refresh.clicked.connect(lambda: self.refresh_models())
        mw = QWidget()
        ml = QHBoxLayout(mw)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.cb_llm, 1)
        ml.addWidget(self.btn_refresh)
        self.row(lay, "llm_model", mw)
        self.llm_dev_box = QWidget()
        dl = QVBoxLayout(self.llm_dev_box)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(4)
        self.cb_llm_dev = NoWheelCombo()
        for code in ("auto", "cpu", "gpu"):
            self.cb_llm_dev.addItem("", code)
        self.cb_llm_dev.currentIndexChanged.connect(self.mark)
        self.row(dl, "llm_device", self.cb_llm_dev)
        dl.addWidget(self.hint("llm_dev_hint"))
        lay.addWidget(self.llm_dev_box)
        tr_row = QHBoxLayout()
        self.btn_test = QPushButton()
        self.reg(self.btn_test, "llm_test")
        self.btn_test.clicked.connect(self.test_llm)
        tr_row.addWidget(self.btn_test)
        tr_row.addWidget(self.hint("llm_test_hint"), 1)
        lay.addLayout(tr_row)
        self.llm_status = self.status_label()
        lay.addWidget(self.llm_status)

        self.pull_box = QWidget()
        pl = QVBoxLayout(self.pull_box)
        pl.setContentsMargins(0, 4, 0, 0)
        sep = QFrame()
        sep.setObjectName("sep")
        pl.addWidget(sep)
        pl.addSpacing(4)
        t = QLabel()
        t.setStyleSheet("font-weight:600;")
        self.reg(t, "dl_title")
        pl.addWidget(t)
        pr = QHBoxLayout()
        self.cb_pull = NoWheelCombo()
        self.cb_pull.setEditable(True)
        self.cb_pull.addItems(["qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:3b", "gemma3:4b", "qwen2.5:7b"])
        self.cb_pull.setCurrentIndex(-1)
        self.reg(self.cb_pull.lineEdit(), "dl_ph", "setPlaceholderText")
        self.btn_pull = QPushButton()
        self.reg(self.btn_pull, "dl_btn")
        self.btn_pull.clicked.connect(self.start_pull)
        pr.addWidget(self.cb_pull, 1)
        pr.addWidget(self.btn_pull)
        pl.addLayout(pr)
        self.pull_bar = QProgressBar()
        self.pull_bar.setTextVisible(False)
        self.pull_bar.setRange(0, 100)
        self.pull_bar.setVisible(False)
        pl.addWidget(self.pull_bar)
        self.pull_status = self.status_label()
        pl.addWidget(self.pull_status)
        lay.addWidget(self.pull_box)

        self.prompt_btn = QToolButton()
        self.prompt_btn.setObjectName("disc")
        self.prompt_btn.setCheckable(True)
        self.prompt_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.prompt_btn.toggled.connect(self.toggle_prompt)
        lay.addWidget(self.prompt_btn)
        self.prompt_box = QWidget()
        pbl = QVBoxLayout(self.prompt_box)
        pbl.setContentsMargins(0, 0, 0, 0)
        self.ed_prompt = QPlainTextEdit()
        self.ed_prompt.setMinimumHeight(130)
        self.ed_prompt.textChanged.connect(self.mark)
        self.btn_reset = QPushButton()
        self.btn_reset.setObjectName("ghost")
        self.reg(self.btn_reset, "prompt_reset")
        self.btn_reset.clicked.connect(lambda: self.ed_prompt.setPlainText(E.DEFAULT_PROMPT))
        pbl.addWidget(self.ed_prompt)
        rr = QHBoxLayout()
        rr.addWidget(self.btn_reset)
        rr.addStretch(1)
        pbl.addLayout(rr)
        self.prompt_box.setVisible(False)
        lay.addWidget(self.prompt_box)

        # extra engines
        lay = self.card("sec_engines")
        lay.addWidget(self.hint("pack_hint"))
        self.pack_w = {}
        group_of = {"gpu": "stable", "sensevoice": "stable", "pytorch": "exp", "llamacpp": "exp"}
        shown = set()
        for pack in E.PACKS:
            g = group_of.get(pack, "exp")
            if g not in shown:
                shown.add(g)
                lay.addSpacing(6)
                lay.addWidget(self.label("pack_group_" + g, "h2"))
                lay.addWidget(self.hint("pack_group_" + g + "_hint"))
            r = QHBoxLayout()
            r.setSpacing(10)
            r.addWidget(self.label("pack_" + pack), 1)
            st = QLabel()
            st.setProperty("kind", "muted")
            b = QPushButton()
            b.clicked.connect(lambda _=False, k=pack: self.pack_action(k))
            r.addWidget(st)
            r.addWidget(b)
            lay.addLayout(r)
            self.pack_w[pack] = (st, b)
        self.btn_sources = QPushButton()
        self.btn_sources.setObjectName("ghost")
        self.reg(self.btn_sources, "pack_sources")
        self.btn_sources.clicked.connect(self.open_sources)
        lay.addWidget(self.btn_sources, 0, Qt.AlignLeft)
        self.pack_info = QLabel()          # what pip is doing / downloading right now
        self.pack_info.setProperty("kind", "muted")
        self.pack_info.setWordWrap(True)
        self.pack_info.hide()
        lay.addWidget(self.pack_info)
        self.pack_bar = QProgressBar()
        self.pack_bar.setTextVisible(False)
        self.pack_bar.setRange(0, 0)
        self.pack_bar.setVisible(False)
        lay.addWidget(self.pack_bar)
        self.pack_err = self.status_label()
        lay.addWidget(self.pack_err)

        # audio / video file -> text
        lay = self.card("sec_files")
        lay.addWidget(self.hint("files_hint"))
        r = QHBoxLayout()
        r.setSpacing(8)
        self.btn_fpick, self.btn_fdir, self.btn_fcancel = QPushButton(), QPushButton(), QPushButton()
        self.reg(self.btn_fpick, "files_pick")
        self.reg(self.btn_fdir, "files_pick_dir")
        self.reg(self.btn_fcancel, "files_cancel")
        self.btn_fpick.clicked.connect(self.pick_files)
        self.btn_fdir.clicked.connect(self.pick_files_dir)
        self.btn_fcancel.clicked.connect(self.eng.cancel_file_job)
        self.btn_fcancel.hide()
        for b in (self.btn_fpick, self.btn_fdir, self.btn_fcancel):
            r.addWidget(b)
        r.addStretch(1)
        lay.addLayout(r)
        self.tg_files = {}
        for name, key in (("srt", "files_srt"), ("vtt", "files_vtt")):
            t = Toggle()
            t.toggled.connect(self.mark)
            self.tg_files[name] = t
            self.row(lay, key, t, False)
        self.files_bar = QProgressBar()
        self.files_bar.setTextVisible(False)
        self.files_bar.setFixedHeight(6)
        self.files_bar.hide()
        lay.addWidget(self.files_bar)
        self.files_status = self.status_label()
        lay.addWidget(self.files_status)
        self.btn_fopen = QPushButton()
        self.btn_fopen.setObjectName("ghost")
        self.reg(self.btn_fopen, "files_open")
        self.btn_fopen.clicked.connect(self.open_files_out)
        self.btn_fopen.hide()
        lay.addWidget(self.btn_fopen, 0, Qt.AlignLeft)
        self.setAcceptDrops(True)

        # notes
        lay = self.card("sec_notes")
        fw = QWidget()
        fl = QHBoxLayout(fw)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        self.ed_folder = QLineEdit()
        self.ed_folder.setMinimumWidth(150)
        self.reg(self.ed_folder, "notes_folder_ph", "setPlaceholderText")
        self.btn_browse, self.btn_open = QPushButton(), QPushButton()
        self.reg(self.btn_browse, "notes_browse")
        self.reg(self.btn_open, "notes_open")
        self.btn_browse.clicked.connect(self.browse_folder)
        self.btn_open.clicked.connect(self.open_folder)
        fl.addWidget(self.ed_folder, 1)
        fl.addWidget(self.btn_browse)
        fl.addWidget(self.btn_open)
        r = QHBoxLayout()
        r.setSpacing(12)
        r.addWidget(self.label("notes_folder"))
        r.addWidget(fw, 1)
        lay.addLayout(r)
        self.ed_prefix = QLineEdit()
        self.ed_prefix.setMaxLength(60)
        self.row(lay, "notes_prefix", self.ed_prefix)
        self.notes_hint = self.hint()
        lay.addWidget(self.notes_hint)
        self.ed_folder.textEdited.connect(self.mark)
        self.ed_prefix.textEdited.connect(self.mark)
        self.ed_prefix.textChanged.connect(self.update_notes_hint)

        # options
        lay = self.card("sec_options")
        self.cb_ui_lang, self.cb_theme = NoWheelCombo(), NoWheelCombo()
        self.row(lay, "opt_ui_lang", self.cb_ui_lang)
        self.row(lay, "opt_theme", self.cb_theme)
        self.cb_ui_lang.currentIndexChanged.connect(self.on_ui_lang)
        self.cb_theme.currentIndexChanged.connect(self.on_theme)
        self.tg = {}
        for name, key in (("beep", "opt_beep"), ("restore", "opt_restore"), ("space", "opt_space")):
            t = Toggle()
            t.toggled.connect(self.mark)
            self.tg[name] = t
            self.row(lay, key, t, False)
        self.cb_maxlen = NoWheelCombo()
        self.cb_maxlen.currentIndexChanged.connect(self.mark)
        self.row(lay, "opt_max_len", self.cb_maxlen)
        t = Toggle()
        t.toggled.connect(self.mark)
        self.tg["keep"] = t
        self.row(lay, "opt_keep_audio", t, False)

        # log
        lay = self.card("sec_log")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(180)
        lay.addWidget(self.log_view)
        self.foot = self.label("footer", "hint")
        self.foot.setWordWrap(True)
        self.foot.setAlignment(Qt.AlignCenter)
        self.body_l.addWidget(self.foot)
        self.body_l.addStretch(1)

        # bottom bar: save status + Quit
        bar = QFrame()
        bar.setObjectName("bottom")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(18, 10, 18, 10)
        self.btn_quit = QPushButton()
        self.btn_quit.setObjectName("danger")
        self.reg(self.btn_quit, "btn_quit")
        self.btn_quit.clicked.connect(self.ask_quit)
        self.save_lb = QLabel()
        self.save_lb.setProperty("kind", "muted")
        self.save_lb.setWordWrap(True)
        self.save_lb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bl.addWidget(self.btn_quit)
        bl.addWidget(self.save_lb, 1)
        v.addWidget(bar)
        for cls in (QPushButton, QToolButton, QComboBox):  # findChildren accepts one type at a time
            for w in self.findChildren(cls):
                w.setCursor(Qt.PointingHandCursor)  # every clickable thing shows the hand cursor

    def _build_tray(self):
        self.tray = QSystemTrayIcon(self.icon_idle, self)
        menu = QMenu()
        self.act_open, self.act_quit = QAction(), QAction()
        self.act_open.triggered.connect(self.bring_to_front)
        self.act_quit.triggered.connect(self.quit_app)
        menu.addAction(self.act_open)
        menu.addAction(self.act_quit)
        self.tray_menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.on_tray)
        self.tray.setToolTip("VoiceNote")
        self.tray.show()

    def on_tray(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.bring_to_front()

    # ---- theme / language
    def apply_theme(self):
        mode = self.S["ui"]["theme"]
        if mode == "system":
            try:
                dark = QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
            except Exception:
                dark = QApplication.palette().window().color().lightness() < 128
        else:
            dark = mode == "dark"
        P.clear()
        P.update(DARK if dark else LIGHT)
        self.app.setStyleSheet(build_qss(P))
        for t in self.tg.values():
            t.update()

    def on_theme(self, *_):
        if self.filling:
            return
        self.S["ui"]["theme"] = self.cb_theme.currentData()
        self.apply_theme()
        self.mark()

    def on_ui_lang(self, *_):
        if self.filling:
            return
        self.S["ui"]["language"] = self.cb_ui_lang.currentData()
        tr.set(self.S["ui"]["language"])
        self.retranslate()
        self.mark()

    def set_disc(self, btn, key, on):
        btn.setText(("▼ " if on else "▶ ") + tr(key))

    def toggle_prompt(self, on):
        self.prompt_box.setVisible(on)
        self.set_disc(self.prompt_btn, "prompt_title", on)

    def lang_items(self):
        """[(code, display name, native name, search keys)] - 'auto' first, then A-Z by displayed name."""
        loc = tr.table.get("languages", {})
        items = [("auto", tr("sp_auto"), "", [tr("sp_auto").lower(), "auto", "autodetect"])]
        rest = []
        cur = self.cb_model.currentData() if self.cb_model.count() else ""
        allowed = set(self.eng.model_languages(cur or ""))
        for code, en_name in E.LANGUAGES.items():
            if code not in allowed:
                continue
            disp, native = loc.get(code, en_name), NATIVE.get(code, "")
            rest.append((code, disp, native, sorted({k.lower() for k in (disp, en_name, native) if k})))
        return (items if "auto" in allowed else []) + sorted(rest, key=lambda i: i[1].lower())

    def update_notes_hint(self, *_):
        pre = E.safe_prefix(self.ed_prefix.text())
        self.notes_hint.setText(tr("notes_hint", a=f"{pre}_01.txt", b=f"{pre}_02.txt"))

    def retranslate(self):
        for w, key, setter in self._tr_items:
            getattr(w, setter)(tr(key))
        self.set_disc(self.prompt_btn, "prompt_title", self.prompt_btn.isChecked())
        for k in self._cards:
            self.update_head(k)
        self.act_open.setText(tr("tray_open"))
        self.act_quit.setText(tr("tray_quit"))
        was, self.filling = self.filling, True
        self.pick_lang.set_items(self.lang_items())
        self.cb_llm.setItemText(0, tr("llm_none"))
        self.cb_mic.setItemText(0, tr("sp_mic_default"))
        self.cb_model.setItemText(0, tr("sp_model_choose"))
        for i, k in enumerate(("dev_auto", "dev_cpu", "dev_gpu")):
            self.cb_dev.setItemText(i, tr(k))
        for i, k in enumerate(("llm_dev_auto", "llm_dev_cpu", "llm_dev_gpu")):
            self.cb_llm_dev.setItemText(i, tr(k))
        self.fill_maxlen(self.cb_maxlen.currentData() or 0)
        for i, k in enumerate(("theme_system", "theme_light", "theme_dark")):
            self.cb_theme.setItemText(i, tr(k))
        self.filling = was
        self.update_notes_hint()
        self.last.setText(self.eng.last_text or tr("last_empty"))
        set_prop(self.last, "kind", "" if self.eng.last_text else "muted")

    # ---- fill / collect / autosave
    def fill(self):
        S = self.S
        for k, b in self.hk.items():
            b.set_value(S["hotkeys"][k])
        self.pick_lang.set_items(self.lang_items())
        self.pick_lang.set_code(S["stt"]["language"])
        self.fill_models(S["stt"]["model"])
        self.cb_dev.setCurrentIndex(max(0, self.cb_dev.findData(S["stt"]["device"])))
        self.cb_llm_dev.setCurrentIndex(max(0, self.cb_llm_dev.findData(S["llm"]["device"])))
        for k, t in self.ticks.items():
            t.setChecked(S["append"][k])
        self.fill_maxlen(S["output"]["max_record_seconds"])
        self.tg["keep"].setChecked(S["output"]["keep_audio"])
        for k, t in self.tg_files.items():
            t.setChecked(S["files"][k])
        self.load_mics()
        self.ed_folder.setText(S["notes"]["folder"])
        self.ed_prefix.setText(S["notes"]["prefix"])
        self.set_provider(S["llm"]["provider"], False)
        self.ed_url.setText(S["llm"]["base_url"])
        self.ed_key.setText(S["llm"]["api_key"])
        self.ed_prompt.setPlainText(S["llm"]["prompt"])
        self.set_llm_models([], S["llm"]["model"])
        self.cb_ui_lang.clear()
        for code, name in Translator.available():
            self.cb_ui_lang.addItem(name, code)
        self.cb_ui_lang.setCurrentIndex(max(0, self.cb_ui_lang.findData(S["ui"]["language"])))
        self.cb_theme.clear()
        for k, v in (("theme_system", "system"), ("theme_light", "light"), ("theme_dark", "dark")):
            self.cb_theme.addItem(tr(k), v)
        self.cb_theme.setCurrentIndex(self.cb_theme.findData(S["ui"]["theme"]))
        for k in self._cards:
            self.set_collapsed(k, k in S["ui"]["collapsed"])
        self.tg["beep"].setChecked(S["output"]["beep"])
        self.tg["restore"].setChecked(S["output"]["restore_clipboard"])
        self.tg["space"].setChecked(S["output"]["trailing_space"])

    def fill_models(self, select):
        """Model list: placeholder, built-in models, models added from Hugging Face links."""
        was, self.filling = self.filling, True
        self.cb_model.clear()
        self.cb_model.addItem(tr("sp_model_choose"), "")
        sizes = {m: E.MODEL_SIZES[m] for m in E.MODEL_SIZES}
        for m, x in E.EXTRA_MODELS.items():
            sizes[m] = x["size"]
        for c in self.S["custom_models"]:
            sizes[c["id"]] = int(c.get("size") or 0)
        for n, info in self.eng.scan_local_models().items():     # folders dropped into models/ by hand
            sizes[n] = info["size"]
        self._local_sig = tuple(sorted(self.eng.local_models_cache))
        engines = {"whisper": "Whisper", "sensevoice": "SenseVoice", "qwen3asr": "PyTorch", "gguf": "llama.cpp"}
        rank = {k: i for i, k in enumerate(engines)}
        for m, mb in sorted(sizes.items(), key=lambda kv: rank.get(self.eng.model_kind(kv[0]), 9)):   # grouped by engine
            label = E.EXTRA_MODELS[m]["title"] if m in E.EXTRA_MODELS else m.replace("--", "/")
            size = "" if not mb else (f" · ~{mb / 1000:.1f} GB" if mb >= 1000 else f" · ~{mb} MB")
            self.cb_model.addItem(f"{label} · {engines.get(self.eng.model_kind(m), '?')}{size}", m)
        if select and self.cb_model.findData(select) < 0:
            self.cb_model.addItem(select, select)
        self.cb_model.setCurrentIndex(max(0, self.cb_model.findData(select or "")))
        self.filling = was
        self.on_model_changed()

    def fill_maxlen(self, cur):
        was, self.filling = self.filling, True
        self.cb_maxlen.clear()
        self.cb_maxlen.addItem(tr("len_unlimited"), 0)
        for h in (1, 2, 4, 8, 12):
            self.cb_maxlen.addItem(tr("len_hours", n=h), h * 3600)
        if cur and self.cb_maxlen.findData(cur) < 0:
            self.cb_maxlen.addItem(f"{cur // 60} min", cur)
        self.cb_maxlen.setCurrentIndex(max(0, self.cb_maxlen.findData(cur)))
        self.filling = was

    def on_model_changed(self, *_):
        cur = self.cb_model.currentData() or ""
        self.btn_modellink.setEnabled(bool(self.eng.model_url(cur)))
        self.btn_modeldel.setVisible(any(c["id"] == cur for c in self.S["custom_models"]) or self.eng.model_downloaded(cur))
        self.refresh_langs()
        self.eng.fetch_gguf_options(cur)
        self.refresh_variants()
        set_prop(self.cb_model, "warn", "true" if not cur else "false")
        self.set_status(self.model_none, tr("sp_model_none_hint") if not cur else "", "bad" if self._need_seen else "muted")

    def refresh_variants(self):
        """Show the model-file combo when the selected GGUF repository offers several files."""
        cur = self.cb_model.currentData() or ""
        opts = self.eng.gguf_opts.get(cur)
        shown = bool(opts and len(opts["main"]) > 1)
        if shown:
            chosen = (self.S["model_files"].get(cur) or {}).get("file") or opts["default"]
            sig = (cur, chosen, len(opts["main"]))
            if getattr(self, "_variant_sig", None) != sig:
                self._variant_sig = sig
                was, self.filling = self.filling, True
                self.cb_variant.clear()
                for f, size in opts["main"]:
                    self.cb_variant.addItem(f"{f} · {fmt_size(size)}" + (" ★" if f == opts["default"] else ""), f)
                self.cb_variant.setCurrentIndex(max(0, self.cb_variant.findData(chosen)))
                self.filling = was
        self.variant_box.setVisible(shown)

    def on_variant(self, *_):
        cur, f = self.cb_model.currentData() or "", self.cb_variant.currentData()
        if self.filling or not cur or not f:
            return
        self.S["model_files"][cur] = {"file": f}
        self.mark()      # autosave: the engine downloads (if needed) and loads the chosen file

    def ask_choice(self, c):
        """A repository with several model files: let the user pick before anything is downloaded."""
        items = [f"{f} · {fmt_size(size)}" + (f" ({tr('choose_recommended')})" if f == c["default"] else "")
                 for f, size in c["options"]]
        cur = next((i for i, (f, _s) in enumerate(c["options"]) if f == c["default"]), 0)
        text, ok = QInputDialog.getItem(self, tr("choose_title"), tr("choose_label"), items, cur, False)
        if not ok or text not in items:     # Cancel: nothing is downloaded and the model is deselected again
            if (self.cb_model.currentData() or "") == c["name"]:
                self.cb_model.setCurrentIndex(0)    # "— Choose a model —" (the change handler saves it)
            return
        self.S["model_files"][c["name"]] = {"file": c["options"][items.index(text)][0]}
        self.mark()

    def refresh_langs(self):
        """The language list shows only what the selected model can recognise."""
        was = self.pick_lang.code
        items = self.lang_items()
        self.pick_lang.set_items(items)
        codes = [i[0] for i in items]
        if was and was not in codes:
            self.pick_lang.set_code("auto" if "auto" in codes else codes[0])
            self.mark()

    def pack_action(self, pack):
        if self.eng.pack_installed(pack):
            if QMessageBox.question(self, "VoiceNote", tr("pack_remove_confirm", name=tr("pack_" + pack))) == QMessageBox.Yes:
                if not self.eng.remove_pack(pack):
                    self.set_status(self.pack_err, tr("model_remove_busy"), "bad")
        else:
            self.eng.start_pack_install(pack)

    @staticmethod
    def pack_text(st):
        """Status text of a running pack installation: stage, real percentage and sizes."""
        stage, total, done = st.get("stage"), st.get("total", 0), st.get("done", 0)
        sec = int(max(0, time.time() - st.get("since", time.time())))
        clock = f"{sec // 60}:{sec % 60:02d}"
        if stage == "download":
            if total:
                return tr("pack_downloading", pct=min(99, round(100 * done / total)), done=fmt_size(done), total=fmt_size(total))
            return tr("pack_downloading", pct="…", done=fmt_size(done), total="?")
        if stage == "install":
            return tr("pack_unpacking") + (f" {min(99, round(100 * done / total))}%" if total else "")
        if stage == "build":
            head = tr("pack_building", name=st.get("name", ""), n=st.get("step", 1), m=st.get("steps", 1))
        else:
            head = tr("pack_resolving")
        return f"{head} {clock}"

    def poll_packs(self):
        e, err, bar = self.eng, "", None
        for pack, (lb, btn) in self.pack_w.items():
            st = e.pack_state.get(pack) or {}
            installed = e.pack_installed(pack)
            if st.get("busy"):
                bar = st
                lb.setText(self.pack_text(st))
                btn.setEnabled(False)
                continue
            if st.get("error"):
                err = tr("pack_failed", err=st["error"])
            var = e.pack_variant(pack) if installed and pack == "pytorch" and e.nvidia else ""
            lb.setText(tr("pack_installed_arch") if var == "arch" else tr("pack_installed_cpu") if var == "cpu" else tr("pack_installed") if installed else tr("pack_not_installed"))
            btn.setText(tr("pack_remove") if installed else tr("pack_install"))
            btn.setEnabled(True)
        self.pack_bar.setVisible(bool(bar))
        info = (bar or {}).get("detail", "") if (bar or {}).get("stage") in ("resolve", "build") else ""
        self.pack_info.setText(info)
        self.pack_info.setVisible(bool(info))
        if bar:
            det = bar.get("stage") in ("download", "install") and bar.get("total", 0) > 0
            self.pack_bar.setRange(0, 100 if det else 0)
            if det:
                self.pack_bar.setValue(max(1, min(99, round(100 * bar.get("done", 0) / bar["total"]))))
        self.set_status(self.pack_err, err, "bad")

    def toggle_dl_pause(self, *_):
        if self.eng.paused_name:
            self.eng.resume_download()
        else:
            self.eng.pause_download()

    def remove_model(self, *_):
        mid = self.cb_model.currentData() or ""
        if not mid:
            return
        custom = any(c["id"] == mid for c in self.S["custom_models"])
        nm = self.cb_model.currentText().split(" · ")[0]
        if QMessageBox.question(self, "VoiceNote", tr("model_remove_confirm" if custom else "model_delete_confirm", name=nm)) != QMessageBox.Yes:
            return
        self.flush_save()
        if self.eng.remove_model(mid):
            self.sync_models()
        else:
            self.set_status(self.add_status, tr("model_remove_busy"), "bad")

    def sync_models(self):
        """The engine changed the model list or the selection (remove / stop): mirror it in the form."""
        self.S["custom_models"] = copy.deepcopy(self.eng.settings["custom_models"])
        self.S["model_files"] = copy.deepcopy(self.eng.settings["model_files"])
        self.S["stt"]["model"] = self.eng.settings["stt"]["model"]
        self.fill_models(self.S["stt"]["model"])

    def open_sources(self, *_):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(E.SOURCES_FILE)))

    def open_model_page(self, *_):
        url = self.eng.model_url(self.cb_model.currentData() or "")
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def add_model(self, *_):
        text = self.ed_add.text().strip()
        if text and self.eng.start_add_model(text):
            self.set_status(self.add_status, tr("add_busy"))

    def collect(self):
        S = self.S
        for k, b in self.hk.items():
            S["hotkeys"][k] = b.value
        S["stt"]["language"] = self.pick_lang.code
        S["stt"]["model"] = self.cb_model.currentData() or ""
        S["stt"]["device"] = self.cb_dev.currentData() or "auto"
        S["llm"]["device"] = self.cb_llm_dev.currentData() or "auto"
        for k, t in self.ticks.items():
            S["append"][k] = t.isChecked()
        S["output"]["max_record_seconds"] = self.cb_maxlen.currentData() or 0
        S["output"]["keep_audio"] = self.tg["keep"].isChecked()
        for k, t in self.tg_files.items():
            S["files"][k] = t.isChecked()
        S["stt"]["input_device"] = self.cb_mic.currentData() or ""
        S["notes"]["folder"] = self.ed_folder.text().strip()
        S["notes"]["prefix"] = self.ed_prefix.text()
        S["llm"].update(base_url=self.ed_url.text().strip(), api_key=self.ed_key.text(),
                        model=self.cb_llm.currentData() or "", prompt=self.ed_prompt.toPlainText())
        S["ui"]["language"] = self.cb_ui_lang.currentData()
        S["ui"]["theme"] = self.cb_theme.currentData()
        if not self.isMaximized() and self.width() > 100:
            S["ui"]["width"], S["ui"]["height"] = self.width(), self.height()
        S["ui"]["collapsed"] = [k for k, c in self._cards.items() if c["collapsed"]]
        S["output"].update(beep=self.tg["beep"].isChecked(), restore_clipboard=self.tg["restore"].isChecked(),
                           trailing_space=self.tg["space"].isChecked())
        return S

    def flash_status(self, text, kind):
        self._flash += 1
        n = self._flash
        self.save_lb.setText(text)
        set_prop(self.save_lb, "kind", kind)
        if kind == "ok":  # the "Saved" message disappears by itself, errors stay until fixed
            QTimer.singleShot(2000, lambda: self.save_lb.setText("") if n == self._flash else None)

    def autosave(self):
        res = self.eng.apply(self.collect())
        if res["ok"]:
            self.flash_status(tr("saved"), "ok")
        else:
            self.flash_status(tr("save_error", err=res["errors"][0]), "bad")

    def flush_save(self):
        if self.save_timer.isActive():
            self.save_timer.stop()
            self.autosave()

    def load_mics(self):
        self.cb_mic.clear()
        self.cb_mic.addItem(tr("sp_mic_default"), "")
        for d in E.input_devices():
            self.cb_mic.addItem(d["name"], d["name"])
        cur = self.S["stt"]["input_device"]
        if cur and self.cb_mic.findData(cur) < 0:
            self.cb_mic.addItem(tr("sp_mic_missing", name=cur), cur)
        self.cb_mic.setCurrentIndex(max(0, self.cb_mic.findData(cur)))

    # ---- hotkey capture / notes folder
    def on_capture(self, on):
        self.eng.suspended = on  # global hotkeys are paused while a new one is being recorded

    def browse_folder(self, *_):
        start = self.ed_folder.text().strip() or str(self.eng.notes_dir())
        d = QFileDialog.getExistingDirectory(self, tr("notes_pick"), start)
        if d:
            self.ed_folder.setText(d)
            self.mark()

    def open_folder(self, *_):
        d = Path(self.ed_folder.text().strip()).expanduser() if self.ed_folder.text().strip() else E.BASE / "notes"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    # ---- LLM
    def provider(self):
        return self.S["llm"]["provider"]

    def set_provider(self, p, user):
        self.S["llm"]["provider"] = p
        self.seg_btn[p].setChecked(True)
        self.custom_box.setVisible(p == "custom")
        self.pull_box.setVisible(p == "ollama")
        self.llm_dev_box.setVisible(p == "ollama")
        if user:
            self.mark()
            self.refresh_models()

    def llm_cfg(self):
        return E.deep_merge(E.DEFAULTS["llm"], {
            "provider": self.provider(), "base_url": self.ed_url.text().strip(), "api_key": self.ed_key.text(),
            "model": self.cb_llm.currentData() or "", "prompt": self.ed_prompt.toPlainText(),
            "device": self.cb_llm_dev.currentData() or "auto"})

    def set_llm_models(self, names, selected):
        was, self.filling = self.filling, True
        items = list(names)
        if selected and selected not in items:
            items.insert(0, selected)
        self.cb_llm.clear()
        self.cb_llm.addItem(tr("llm_none"), "")
        for n in items:
            self.cb_llm.addItem(n, n)
        self.cb_llm.setCurrentIndex(max(0, self.cb_llm.findData(selected or "")))
        self.filling = was

    def refresh_models(self, select=None):
        cfg = self.llm_cfg()
        keep = select or cfg["model"]
        self._req += 1
        req = self._req
        self.set_status(self.llm_status, tr("llm_looking"))

        def work():
            try:
                return {"ok": True, "models": E.llm_models(cfg)}
            except Exception as ex:
                return {"ok": False, "error": E.friendly_error(ex)}

        def done(r):
            if req != self._req:
                return  # a newer request superseded this one
            if not r["ok"]:
                help_ = {"ollama": tr("llm_help_ollama"), "lmstudio": tr("llm_help_lmstudio")}.get(cfg["provider"], "")
                self.set_status(self.llm_status, html.escape(tr("llm_unreachable", err=r["error"])) + " " + help_, "bad", True)
                return
            self.set_llm_models(r["models"], keep)
            n = len(r["models"])
            self.set_status(self.llm_status, tr("llm_connected", n=n) if n else tr("llm_connected_empty"),
                            "ok" if n else "muted")

        self.bridge.run(work, done)

    def test_llm(self, *_):
        cfg = self.llm_cfg()
        self.set_status(self.llm_status, tr("llm_testing"))
        self.btn_test.setEnabled(False)
        sample = "um so like i think we should uh meet tomorrow at three maybe in the main office"

        def work():
            try:
                out = E.llm_refine(sample, cfg)
                return {"ok": True, "out": out, "where": E.llm_where(cfg)}
            except Exception as ex:
                return {"ok": False, "error": E.friendly_error(ex)}

        def done(r):
            self.btn_test.setEnabled(True)
            if r["ok"]:
                where = r.get("where")
                extra = f"\n({tr('llm_where_' + where)})" if where in ("gpu", "cpu", "mixed") else ""
                self.set_status(self.llm_status, f"“{sample}”  →  “{r['out']}”{extra}", "ok")
            else:
                self.set_status(self.llm_status, tr("llm_test_failed", err=r["error"]), "bad")

        self.bridge.run(work, done)

    def start_pull(self, *_):
        name = self.cb_pull.currentText().strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,100}", name):
            return
        if self.eng.start_pull(name):
            self._pull_active = name
        else:
            self.set_status(self.pull_status, tr("dl_busy"), "bad")

    # ---- live state
    def poll(self):
        e = self.eng
        color = {"ok": P["ok"], "rec": P["bad"], "busy": P["warn"], "err": P["bad"]}
        pr = e.model_progress()
        kind, txt = "ok", tr("st_ready")
        if e.loading and e.phase == "download":
            kind, txt = "busy", tr("st_downloading") + (f" {round((pr or 0) * 100)}%")
        elif e.loading:
            kind, txt = "busy", tr("st_loading") + "…"
        elif e.paused_name:
            kind, txt = "busy", tr("st_paused")
        elif e.model_error:
            kind, txt = "err", tr("st_error")
        elif e.state == "recording":
            kind, txt = "rec", tr("st_recording")
        elif e.state == "processing":
            kind, txt = "busy", tr("st_processing")
        info = e.rec_info()
        if info and kind == "rec":
            sec = info["elapsed"]
            clock = f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"
            txt += " " + clock
        elif info and kind == "busy" and e.state == "processing":
            txt = tr("st_processing_n", done=info["done"], total=info["total"])
        self.pill.setText(f"<span style='color:{color[kind]}'>●</span>&nbsp; {html.escape(txt)}")
        recording = kind == "rec"
        if recording != self._last_icon:
            self._last_icon = recording
            self.tray.setIcon(self.icon_rec if recording else self.icon_idle)
        self.model_bar.setVisible(e.loading)
        if e.loading:
            if pr is not None:
                self.model_bar.setRange(0, 100)
                self.model_bar.setValue(max(2, round(pr * 100)))
            else:
                self.model_bar.setRange(0, 0)  # busy indicator while the model is being loaded into memory
        downloading = e.loading and e.phase == "download"
        if downloading or e.paused_name:
            nm = e.settings["stt"]["model"].replace("--", "/")
            self.set_status(self.model_dl, tr("dl_paused" if e.paused_name else "dl_model", name=nm,
                                              done=fmt_size(e.dl_bytes), total=fmt_size(e.dl_total)))
            self.model_bar.setVisible(True)
            if e.paused_name:
                self.model_bar.setRange(0, 100)
                self.model_bar.setValue(max(2, round(100 * e.dl_bytes / max(1, e.dl_total))))
            self.btn_dlpause.setText("▶" if e.paused_name else "⏸")
            self.btn_dlpause.setToolTip(tr("dl_resume" if e.paused_name else "dl_pause"))
        else:
            self.set_status(self.model_dl, "")
        self.dl_row.setVisible(bool(downloading or e.paused_name))
        cur_model = e.settings["stt"]["model"]
        if time.time() - getattr(self, "_local_scan", 0) > 3:      # pick up model folders added while running
            self._local_scan = time.time()
            if tuple(sorted(e.scan_local_models())) != getattr(self, "_local_sig", ()):
                self.fill_models(self.cb_model.currentData() or "")
        c = e.choose
        if c and c["name"] == cur_model and c is not getattr(self, "_choose_seen", None):
            self._choose_seen = c
            QTimer.singleShot(0, lambda c=c: self.ask_choice(c))
        self.refresh_variants()
        if e.model_error.startswith("pack:"):
            pack = e.model_error[5:]
            if self._pack_seen != pack:
                self._pack_seen = pack
                self.set_collapsed("sec_engines", False)  # show where to install it
            self.set_status(self.model_err, tr("pack_needed", name=tr("pack_" + pack)), "bad")
        elif e.model_error in ("torch_cpu", "torch_arch"):
            self.set_status(self.model_err, tr(e.model_error), "bad")
        elif c and c["name"] == cur_model and not e.loading:
            self.set_status(self.model_err, tr("choose_pending"), "muted")
        elif e.model_error:
            link = f" <a href='retry'>{html.escape(tr('retry'))}</a>"
            self.set_status(self.model_err, html.escape(tr("model_err", err=e.model_error)) + link, "bad", True)
        else:
            self.set_status(self.model_err, "")
        self.set_status(self.hk_err, " · ".join(e.hk_errors), "bad")
        if e.last_text and self.last.text() != e.last_text:
            self.last.setText(e.last_text)
            set_prop(self.last, "kind", "")
            QTimer.singleShot(0, lambda: self.last_scroll.verticalScrollBar().setValue(self.last_scroll.verticalScrollBar().maximum()))
        if self.log_view.isVisible():
            lines = "\n".join(list(E.LOG)[-80:])
            if self.log_view.toPlainText() != lines:
                self.log_view.setPlainText(lines)
                self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
        self.poll_pull()
        cur = self.cb_model.currentData() or ""
        self.btn_modeldel.setVisible(any(c["id"] == cur for c in self.S["custom_models"]) or e.model_downloaded(cur))
        self.poll_device()
        self.poll_packs()
        self.poll_add()
        self.poll_files()
        self.poll_flags()

    # ---- audio / video file -> text
    FILE_FILTER = "*" + " *".join(E.FJ.AUDIO_EXTS)

    def start_files(self, paths):
        self.flush_save()  # the job uses the saved settings
        code = self.eng.start_file_job(paths)
        if code != "ok":
            self.set_status(self.files_status, tr("files_err_" + code), "bad")

    def pick_files(self, *_):
        fl, _f = QFileDialog.getOpenFileNames(self, tr("files_pick"), str(self.eng.notes_dir()),
                                              tr("files_filter") + f" ({self.FILE_FILTER});;*")
        if fl:
            self.start_files(fl)

    def pick_files_dir(self, *_):
        d = QFileDialog.getExistingDirectory(self, tr("files_pick_dir"), str(Path.home()))
        if d:
            self.start_files([d])

    def open_files_out(self, *_):
        j = self.eng.file_job
        if j and j.get("folder"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(j["folder"]))

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.start_files(paths)

    def poll_files(self):
        j = self.eng.file_job
        if not j:
            return
        busy = j["busy"]
        self.btn_fpick.setEnabled(not busy)
        self.btn_fdir.setEnabled(not busy)
        self.btn_fcancel.setVisible(busy)
        self.files_bar.setVisible(busy)
        if busy:
            self.files_bar.setRange(0, 100)
            self.files_bar.setValue(max(1, j["pct"]))
            pct = f" {j['pct']}%" if j["stage"] == "transcribe" else ""
            self.set_status(self.files_status, tr("files_progress", i=j["index"] + 1, n=j["total"], name=j["name"],
                                                  stage=tr("files_st_" + j["stage"]) + pct))
            self.btn_fopen.hide()
            return
        lines = []
        if j["outputs"]:
            lines.append(tr("files_done", n=len(j["outputs"]), folder=j["folder"]))
        if j["cancel"]:
            lines.append(tr("files_cancelled"))
        lines += [f"{n}: {m}" for n, m in j["errors"]]
        self.set_status(self.files_status, "\n".join(lines), "bad" if j["errors"] else "ok")
        self.btn_fopen.setVisible(bool(j["outputs"]))

    def poll_flags(self):
        e = self.eng
        if e.append_rev != self._rev_seen:  # an Alt+key shortcut flipped a tick: mirror it without re-saving
            self._rev_seen = e.append_rev
            was, self.filling = self.filling, True
            for k, t in self.ticks.items():
                t.setChecked(e.settings["append"][k])
            self.filling = was
        if e.sel_rev != self._sel_seen:  # a model download was cancelled: the selection was cleared
            self._sel_seen = e.sel_rev
            self.sync_models()
        if e.need_model > self._need_seen:  # a hotkey was pressed while no model is chosen
            self._need_seen = e.need_model
            self.set_collapsed("sec_speech", False)
            self.on_model_changed()
            self.bring_to_front()
            self.tray.showMessage("VoiceNote", tr("need_model_msg"), QSystemTrayIcon.Warning, 4000)

    def poll_device(self):
        e = self.eng
        g = e.pack_state.get("gpu") or {}
        running = bool(g.get("busy"))
        text, kind = "", "muted"
        info = tr("dev_gpu_info", name=e.gpu[0], driver=e.gpu[1] or "?") if e.gpu else ""
        if running:
            text = self.pack_text(g)
        elif g.get("error"):
            text, kind = tr("pack_failed", err=g["error"]), "bad"
        elif e.model is not None and not e.loading:
            text = tr("dev_running", dev=e.device_used.upper())
            if e.device_note == "no_nvidia":
                text += "\n" + tr("dev_note_no_nvidia")
            elif e.device_note in ("gpu_unavailable", "gpu_failed"):
                text += "\n" + tr("dev_note_unavail", err=e.device_detail)
                kind = "bad" if e.device_note == "gpu_failed" else "muted"
            if e.cuda_dead:
                text += "\n" + tr("dev_restart")
        if info and not running:
            text = (text + "\n" if text else "") + info
            if e.gpu and E.driver_too_old(e.gpu[1]):
                text += "\n" + tr("dev_driver_old")
                kind = "bad"
        # an NVIDIA GPU exists but the CUDA libraries are not installed: point to the single place to install them
        if e.nvidia and e.gpu_missing and not running and not e.pack_installed("gpu"):
            text = (text + "\n" if text else "") + tr("dev_need_pack")
        self.set_status(self.dev_status, text, kind)

    def poll_add(self):
        a = self.eng.add_state
        if not a:
            return
        self.btn_add.setEnabled(not a["busy"])
        if a["busy"] or a is self._add_seen:
            return
        self._add_seen = a
        code, detail, added = a["code"], a["detail"], a["added"]
        if code in ("", "exists") and added:
            self.S["custom_models"] = copy.deepcopy(self.eng.settings["custom_models"])
            self.fill_models(self.cb_model.currentData() or "")
            self.cb_model.setCurrentIndex(self.cb_model.findData(added))  # selecting it autosaves and starts the download
            self.ed_add.clear()
            msg, kind = tr("add_exists" if code == "exists" else "add_ok", name=added.replace("--", "/")), "ok"
            if detail == "no_mmproj":
                msg += " " + tr("add_gguf_single")
        else:
            kind = "bad"
            msg = {"bad_link": tr("add_bad"), "not_found": tr("add_nf", name=detail),
                   "unsupported": tr("add_unsupported", detail=detail),
                   "network": tr("add_net", err=detail)}.get(code, detail)
        self.set_status(self.add_status, msg, kind)

    def poll_pull(self):
        p = self.eng.pull
        running = bool(p and not p["done"] and not p["error"])
        self.btn_pull.setEnabled(not running)
        self.pull_bar.setVisible(running)
        if running:
            frac = p["completed"] / p["total"] if p["total"] else 0
            self.pull_bar.setValue(max(2, round(frac * 100)))
            if p["total"]:
                txt = tr("dl_progress_pct", name=p["name"], status=p["status"], pct=round(frac * 100),
                         done=fmt_size(p["completed"]), total=fmt_size(p["total"]))
            else:
                txt = tr("dl_progress", name=p["name"], status=p["status"])
            self.set_status(self.pull_status, txt)
        elif p and p["error"]:
            self.set_status(self.pull_status, tr("dl_failed", err=p["error"]), "bad")
        elif p and p["done"] and self._pull_active == p["name"]:
            self._pull_active = None  # a download started from this window just finished: select the model
            self.set_status(self.pull_status, tr("dl_done", name=p["name"]), "ok")
            self.refresh_models(select=p["name"])
            self.mark()

    # ---- window / app lifecycle
    def bring_to_front(self, *_):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, ev):
        self.flush_save()
        if self.tray.isVisible():
            ev.ignore()
            self.hide()
            if not getattr(self, "_told", False):
                self._told = True
                self.tray.showMessage("VoiceNote", tr("tray_running"), QSystemTrayIcon.Information, 3000)
        else:
            ev.accept()

    def ask_quit(self, *_):
        if QMessageBox.question(self, tr("quit_title"), tr("quit_confirm")) == QMessageBox.Yes:
            self.quit_app()

    def quit_app(self, *_):
        """Exit completely: save pending changes, release the speech model and end the process."""
        try:
            self.flush_save()
            self.tray.hide()
            self.eng.shutdown()
        finally:
            os._exit(0)  # guarantees that no background thread keeps the process (and the model) alive


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("VoiceNote")

    # single instance: a second launch just asks the first one to show its window
    sock = QLocalSocket()
    sock.connectToServer(SINGLE_INSTANCE_NAME)
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.flush()
        sock.waitForBytesWritten(500)
        return 0
    QLocalServer.removeServer(SINGLE_INSTANCE_NAME)
    server = QLocalServer()
    server.listen(SINGLE_INSTANCE_NAME)

    E.MODELS_DIR.mkdir(exist_ok=True)
    eng = E.Engine()
    tr.set(eng.settings["ui"]["language"])
    win = MainWindow(eng, app)
    app.aboutToQuit.connect(eng.shutdown)

    def on_connection():
        server.nextPendingConnection()
        win.bring_to_front()

    server.newConnection.connect(on_connection)
    eng.start()
    win.show()
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        (BASE / "crash.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
