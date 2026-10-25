"""Every UI string of the English table must exist in each lang/*.json with the same {placeholders}."""
import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LANGS = sorted(p for p in (ROOT / "lang").glob("*.json") if not p.name.startswith("_"))


def _english():
    tree = ast.parse((ROOT / "voicenote.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "EN" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("EN dictionary not found in voicenote.py")


def _ph(s):
    return sorted(re.findall(r"\{\w*\}", s))


EN = _english()


def test_english_table_not_empty():
    assert len(EN) > 100


def test_native_names_cover_all_languages():
    native = json.loads((ROOT / "lang" / "_native.json").read_text(encoding="utf-8"))
    for p in LANGS:
        assert p.stem in native, p.name


@pytest.mark.parametrize("path", LANGS, ids=lambda p: p.stem)
def test_language_is_complete_and_consistent(path):
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table.get("_name"), "missing _name"
    missing = [k for k in EN if k not in table]
    assert not missing, f"missing keys: {missing[:10]}"
    extra = [k for k in table if k not in ("_name", "languages") and k not in EN]   # "languages": optional localized language names
    assert not extra, f"unknown keys: {extra[:10]}"
    bad = [k for k in EN if k in table and _ph(EN[k]) != _ph(table[k])]
    assert not bad, f"placeholder mismatch: {bad[:10]}"
