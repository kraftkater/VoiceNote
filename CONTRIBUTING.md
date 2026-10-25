# Contributing to VoiceNote

Thanks for your interest! Bug reports, translations, documentation fixes and code are all welcome.

## Ground rules

- Be kind — see the [Code of Conduct](CODE_OF_CONDUCT.md).
- For anything bigger than a small fix, **open an issue first** so we can agree on the approach before you invest time.
- Keep pull requests focused: one logical change per PR, unrelated clean-ups in a separate PR.
- Security problems: do **not** open a public issue — see [SECURITY.md](SECURITY.md).

## Reporting bugs

Use the *Bug report* issue form. The most useful reports contain: VoiceNote version, Windows version, GPU and driver,
the selected model and processor, the steps to reproduce, and the relevant logs
(`crash.log`, `worker.log`, `packs/install.log`, and the *Log* section of the window). Remove anything private first.

## Development setup

VoiceNote targets **Windows 10/11 and Python 3.11** (the version the portable installer downloads).
The UI and hotkeys only run on Windows; the pure logic and the tests run anywhere.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # Windows: everything the app needs
pip install -r requirements-dev.txt
python voicenote.py               # Windows only
```

Tests and linting (these also run in CI, on Linux, without a sound card or Qt):

```bash
ruff check .
pytest
```

`tests/conftest.py` replaces `sounddevice`, `keyboard` and `pyperclip` with stubs, so the engine can be imported on any OS.

## Code style

- English identifiers and **English code comments** (user-visible text goes through the translation tables).
- Keep the line length around 120 characters; `ruff` enforces the basics (config in `pyproject.toml`).
- Prefer small, targeted changes over rewrites; keep unrelated code untouched.
- Never hard-code download links or package names: they belong in `sources.json` (and its defaults in `engine.py`).
- Anything that can fail on a user's machine (network, GPU, drivers, antivirus) must fail **softly**: a clear message in the UI
  and a line in the log, never an unexplained crash or an endless spinner.

## Translations

UI strings live in two places:

- `EN` dictionary in `voicenote.py` — the source of truth;
- one file per language in `lang/` (`ru.json`, `uk.json`, …); `lang/_native.json` holds native language names.

To add or fix a language:

1. Copy `lang/de.json` to `lang/<code>.json` and translate the values (keep `{placeholders}` exactly as in English).
2. Set `"_name"` to the language's name in that language.
3. Run `pytest tests/test_translations.py` — it checks that every English key exists and that placeholders match.

A missing key falls back to English at runtime, so partial translations are acceptable in a draft PR.

## Pull request checklist

- [ ] `ruff check .` and `pytest` pass.
- [ ] New or changed user-visible text exists in `EN` **and** in every `lang/*.json` (or the PR says which are pending).
- [ ] New behavior has a test when it is pure logic; hardware-dependent changes describe how you tested them
      (Windows version, GPU, driver, model).
- [ ] [`CHANGELOG.md`](CHANGELOG.md) has an entry under *Unreleased*.
- [ ] No secrets, personal paths or large binaries are committed.

## Releases (maintainers)

1. Update `VERSION` and move the *Unreleased* entries in `CHANGELOG.md` under the new version.
2. `python tools/build_release.py` builds `dist/VoiceNote-<version>.zip` locally for a smoke test.
3. Tag and push: `git tag v<version> && git push origin v<version>` — the *Release* workflow builds the archive and attaches it
   to a GitHub release.

By contributing you agree that your contribution is licensed under the project's [MIT License](LICENSE).
