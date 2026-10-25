import json

import engine as E


def test_parse_hf_repo():
    assert E.parse_hf_repo("https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF") == "ggml-org/Qwen3-ASR-1.7B-GGUF"
    assert E.parse_hf_repo("huggingface.co/a/b/tree/main?x=1") == "a/b"
    assert E.parse_hf_repo("owner/name") == "owner/name"
    assert E.parse_hf_repo("https://huggingface.co/datasets/a/b") is None
    assert E.parse_hf_repo("https://example.com/a/b") is None
    assert E.parse_hf_repo("nonsense") is None


def test_safe_prefix():
    assert E.safe_prefix('a<b>:"c"') == "abc"
    assert E.safe_prefix("  ...  ") == "note"
    assert len(E.safe_prefix("x" * 200)) <= 60


def test_pick_gguf_prefers_accurate_quant_and_f16_projector():
    names = ["m-Q4_K_M.gguf", "m-Q8_0.gguf", "m-BF16.gguf", "mmproj-f16.gguf", "mmproj-q8_0.gguf", "m-Q8_0-00001-of-00002.gguf"]
    assert E.pick_gguf(names) == ("m-Q8_0.gguf", "mmproj-f16.gguf")


def test_pick_gguf_without_projector_or_files():
    assert E.pick_gguf(["x-q4_k_m.gguf"]) == ("x-q4_k_m.gguf", "")
    assert E.pick_gguf(["readme.md"]) == ("", "")


def test_classify_repo():
    assert E.classify_repo(["model.bin", "config.json", "tokenizer.json"])[:2] == ("ct2-whisper", True)
    assert E.classify_repo(["a.gguf", "mmproj-f16.gguf"], "o/n")[:2] == ("gguf", True)
    assert E.classify_repo(["a.nemo"])[:2] == ("nemo", False)
    assert E.classify_repo(["tokens.txt", "model.int8.onnx"], "o/sherpa-sense-voice")[:2] == ("sensevoice", True)
    assert E.classify_repo(["config.json", "m.safetensors"], "Qwen/Qwen3-ASR-1.7B")[:2] == ("qwen3asr", True)
    assert E.classify_repo(["README.md"])[:2] == ("unknown", False)


def test_deep_merge_keeps_only_known_keys():
    base = {"a": 1, "b": {"c": 2}}
    assert E.deep_merge(base, {"a": 5, "b": {"c": 3, "zzz": 1}, "x": 9}) == {"a": 5, "b": {"c": 3}}
    assert base == {"a": 1, "b": {"c": 2}}


def test_migrate_old_settings():
    out = E.migrate({"version": 1, "hotkeys": {"cancel": "esc"}})
    assert out["hotkeys"]["cancel"] == "pause"
    assert out["version"] == E.SETTINGS_VERSION
    assert E.migrate("garbage") == {}


def test_validate_defaults_are_clean():
    clean, errs = E.validate({})
    assert errs == []
    assert clean["stt"]["device"] in ("auto", "cpu", "cuda")


def test_validate_rejects_duplicate_hotkeys():
    _, errs = E.validate({"hotkeys": {"dictate": "f9", "dictate_llm": "f9"}})
    assert any("different" in e for e in errs)


def test_load_sources_ignores_bad_types_and_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "SOURCES_FILE", tmp_path / "nope.json")
    assert E.load_sources()["hf_base"] == E._SOURCE_DEFAULTS["hf_base"]
    f = tmp_path / "sources.json"
    f.write_text(json.dumps({"hf_base": "https://mirror.example", "llamacpp_release_api": 5}), encoding="utf-8")
    monkeypatch.setattr(E, "SOURCES_FILE", f)
    s = E.load_sources()
    assert s["hf_base"] == "https://mirror.example"
    assert s["llamacpp_release_api"] == E._SOURCE_DEFAULTS["llamacpp_release_api"]
    f.write_text("{broken", encoding="utf-8")
    assert E.load_sources()["hf_base"] == E._SOURCE_DEFAULTS["hf_base"]


def test_lang_tokens(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_text('{"x": "<|en|> <|ru|> <|startoftranscript|> <|yue|>"}', encoding="utf-8")
    assert E.Engine._lang_tokens(p) == {"en", "ru", "yue"}
    assert E.Engine._lang_tokens(tmp_path / "missing.txt") is None


def _engine_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "MODELS_DIR", tmp_path)
    eng = E.Engine.__new__(E.Engine)     # no hotkeys, no audio: only what scan_local_models needs
    eng.settings = {"custom_models": []}
    eng.local_models_cache = {}
    return eng


def test_scan_local_models(tmp_path, monkeypatch):
    eng = _engine_stub(tmp_path, monkeypatch)
    w = tmp_path / "my-whisper"
    w.mkdir()
    for f in ("model.bin", "config.json", "tokenizer.json"):
        (w / f).write_bytes(b"x")
    g = tmp_path / "my-gguf"
    g.mkdir()
    (g / "m-Q8_0.gguf").write_bytes(b"x")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "readme.md").write_text("x")
    half = tmp_path / "half"          # whisper folder with a file missing
    half.mkdir()
    (half / "model.bin").write_bytes(b"x")
    (tmp_path / "tiny").mkdir()       # built-in model names are not listed twice
    (tmp_path / "tiny" / "model.bin").write_bytes(b"x")
    found = eng.scan_local_models()
    assert sorted(found) == ["my-gguf", "my-whisper"]
    assert eng.model_kind("my-gguf") == "gguf"
    assert eng.model_kind("my-whisper") == "whisper"
