import filejob as FJ


def test_fmt_ts():
    assert FJ.fmt_ts(0) == "00:00:00,000"
    assert FJ.fmt_ts(3661.5) == "01:01:01,500"
    assert FJ.fmt_ts(1.25, ".") == "00:00:01.250"
    assert FJ.fmt_ts(-5) == "00:00:00,000"


def test_srt_and_vtt():
    cues = [(0.0, 1.5, "Hello"), (1.5, 3.0, "World")]
    srt = FJ.to_srt(cues)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n")
    vtt = FJ.to_vtt(cues)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello")


def test_split_cue_keeps_time_range_and_text():
    text = "One short sentence. " + "word " * 60 + "end."
    cues = FJ.split_cue(10.0, 30.0, text, max_chars=60)
    assert len(cues) > 2
    assert cues[0][0] == 10.0 and abs(cues[-1][1] - 30.0) < 1e-6
    assert all(len(c[2]) <= 61 for c in cues)
    assert all(a[1] == b[0] or abs(a[1] - b[0]) < 1e-9 for a, b in zip(cues, cues[1:]))


def test_split_cue_empty():
    assert FJ.split_cue(0, 1, "   ") == []


def test_paragraphs_group_sentences():
    text = " ".join(f"Sentence number {i}." for i in range(40))
    paras = FJ.paragraphs(text, size=200)
    assert len(paras) > 3
    assert " ".join(paras) == text


def test_collect_files_filters_and_dedups(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.WAV").write_bytes(b"x")
    found = FJ.collect_files([tmp_path, tmp_path / "a.mp3"])
    assert sorted(p.name for p in found) == ["a.mp3", "c.WAV"]


def test_unique_base(tmp_path):
    assert FJ.unique_base(tmp_path, "talk", (".txt", ".srt")) == "talk"
    (tmp_path / "talk.srt").write_text("x")
    assert FJ.unique_base(tmp_path, "talk", (".txt", ".srt")) == "talk (2)"
    (tmp_path / "talk (2).txt").write_text("x")
    assert FJ.unique_base(tmp_path, "talk", (".txt", ".srt")) == "talk (3)"
    assert FJ.unique_base(tmp_path, 'a:b*c', (".txt",)) == "a_b_c"
