"""Helpers for turning audio/video files into text (pure functions, no I/O except file listing; easy to test)."""
import re
from pathlib import Path

# what PyAV (installed with faster-whisper) can normally decode
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".webm", ".ogg", ".opus", ".flac", ".aac", ".wma", ".mov",
              ".avi", ".m4b", ".aiff", ".aif", ".3gp", ".amr", ".ts")



def collect_files(paths):
    """Files from the given files/folders (folders are searched recursively), only known audio/video types."""
    out, seen = [], set()
    for p in paths:
        p = Path(p)
        found = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        for f in found:
            if f.suffix.lower() in AUDIO_EXTS and f.exists() and str(f) not in seen:
                seen.add(str(f))
                out.append(f)
    return out


def split_cue(start, end, text, max_chars=84):
    """Break a long piece of text into subtitle-sized cues; time is shared in proportion to the text length."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…。！？])\s+", text)
    pieces = []
    for part in parts:
        while len(part) > max_chars:
            cut = max(part.rfind(", ", 0, max_chars), part.rfind(" ", 0, max_chars))
            if cut < max_chars // 3:
                cut = max_chars
            pieces.append(part[:cut + 1].strip())
            part = part[cut + 1:].strip()
        if part:
            pieces.append(part)
    total = sum(len(x) for x in pieces) or 1
    out, t = [], start
    for x in pieces:
        d = (end - start) * len(x) / total
        out.append((t, t + d, x))
        t += d
    return out


def fmt_ts(sec, sep=","):
    ms = max(0, int(round(sec * 1000)))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def to_srt(cues):
    return "".join(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{t}\n\n" for i, (s, e, t) in enumerate(cues, 1))


def to_vtt(cues):
    return "WEBVTT\n\n" + "".join(f"{fmt_ts(s, '.')} --> {fmt_ts(e, '.')}\n{t}\n\n" for s, e, t in cues)


def paragraphs(text, size=500):
    """Readable paragraphs: sentences are grouped until a paragraph is about `size` characters long."""
    sentences = re.split(r"(?<=[.!?…。！？])\s+", re.sub(r"\s+", " ", text).strip())
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > size:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def unique_base(folder, stem, exts):
    """`stem` or `stem (2)`, ... such that none of stem+ext exists yet in `folder`."""
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .") or "audio"
    n, name = 1, stem
    while any((Path(folder) / (name + e)).exists() for e in exts):
        n += 1
        name = f"{stem} ({n})"
    return name
