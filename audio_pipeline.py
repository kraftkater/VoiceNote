"""Cutting a long recording into chunks at natural pauses (pure numpy, no I/O, easy to test)."""
from collections import deque

import numpy as np

SAMPLE_RATE = 16000


class Chunker:
    """Collects audio blocks and hands out chunks of roughly 15-40 seconds.

    A chunk is cut when at least `min_s` seconds are buffered and a pause (`pause_s` of quiet) has just
    happened. If no pause shows up, the chunk is cut at the quietest spot of the last seconds once `max_s`
    is reached, so words are (almost) never split. The quiet threshold adapts to the microphone's noise level.
    """

    def __init__(self, emit, sr=SAMPLE_RATE, min_s=15.0, max_s=40.0, pause_s=0.5):
        self.emit, self.sr = emit, sr
        self.min_n, self.max_n, self.pause_n = int(min_s * sr), int(max_s * sr), int(pause_s * sr)
        self.buf, self.n, self.quiet, self.total = [], 0, 0, 0
        self.rms_hist = deque(maxlen=200)

    def threshold(self):
        if len(self.rms_hist) < 10:
            return 0.006
        return float(np.clip(2.0 * np.percentile(self.rms_hist, 10), 0.003, 0.03))

    def feed(self, block):
        """block: 1-D float32 array in [-1, 1]."""
        if len(block) == 0:
            return
        rms = float(np.sqrt(np.mean(block * block)))
        self.rms_hist.append(rms)
        self.quiet = self.quiet + len(block) if rms < self.threshold() else 0
        self.buf.append(block)
        self.n += len(block)
        self.total += len(block)
        if self.n >= self.min_n and self.quiet >= self.pause_n:
            self._cut(self.n)
        elif self.n >= self.max_n:
            self._cut_quietest()

    def _joined(self):
        return self.buf[0] if len(self.buf) == 1 else np.concatenate(self.buf)

    def _cut(self, at):
        a = self._joined()
        rest = a[at:]
        self.buf = [rest] if len(rest) else []
        self.n, self.quiet = len(rest), 0
        self.emit(a[:at])

    def _cut_quietest(self):
        a = self._joined()
        win = int(0.2 * self.sr)
        tail = a[-int(8 * self.sr):]
        k = len(tail) // win
        if k == 0:
            return self._cut(len(a))
        energy = (tail[:k * win].reshape(k, win) ** 2).mean(axis=1)
        at = len(a) - len(tail) + int(np.argmin(energy)) * win + win // 2
        self._cut(at)

    def flush(self):
        """Emit whatever is left (ignored if shorter than 0.3 s)."""
        if self.n >= 0.3 * self.sr:
            a = self._joined()
            self.buf, self.n, self.quiet = [], 0, 0
            self.emit(a)
        else:
            self.buf, self.n, self.quiet = [], 0, 0
