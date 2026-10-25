import numpy as np

from audio_pipeline import SAMPLE_RATE, Chunker


def _tone(seconds, amp=0.2):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _feed(ch, audio, block=1600):
    for i in range(0, len(audio), block):
        ch.feed(audio[i:i + block])


def test_cut_at_pause_after_min_length():
    out = []
    ch = Chunker(out.append, min_s=5, max_s=20, pause_s=0.5)
    _feed(ch, np.concatenate([_tone(6), np.zeros(SAMPLE_RATE, np.float32), _tone(2)]))
    ch.flush()
    assert len(out) == 2
    assert 5.5 < len(out[0]) / SAMPLE_RATE < 7.5


def test_hard_cut_without_pause_and_nothing_lost():
    out = []
    ch = Chunker(out.append, min_s=5, max_s=10, pause_s=0.5)
    audio = _tone(25)
    _feed(ch, audio)
    ch.flush()
    assert len(out) >= 3
    assert all(len(c) / SAMPLE_RATE <= 10.5 for c in out)
    assert sum(len(c) for c in out) == len(audio)


def test_flush_ignores_tiny_remainder():
    out = []
    ch = Chunker(out.append)
    _feed(ch, _tone(0.1))
    ch.flush()
    assert out == []
