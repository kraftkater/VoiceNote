"""Build dist/VoiceNote-<version>.zip: the portable program without any runtime data.

Usage:  python tools/build_release.py
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

INCLUDE = [
    "audio_pipeline.py", "engine.py", "engine_worker.py", "filejob.py", "voicenote.py",
    "sources.json", "setup.ps1", "run.bat", "requirements.txt",
    "README.md", "README.ru.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "VERSION",
]
INCLUDE_DIRS = ["lang"]


def main():
    out_dir = ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    target = out_dir / f"VoiceNote-{VERSION}.zip"
    files = [ROOT / n for n in INCLUDE]
    for d in INCLUDE_DIRS:
        files += sorted(p for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    missing = [str(f.relative_to(ROOT)) for f in files if not f.exists()]
    if missing:
        sys.exit("Missing files: " + ", ".join(missing))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path("VoiceNote") / f.relative_to(ROOT))
    print(f"{target}  ({target.stat().st_size / 1024:.0f} KB, {len(files)} files)")


if __name__ == "__main__":
    main()
