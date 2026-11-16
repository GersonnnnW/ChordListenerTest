"""Tempo, beat and bar tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .pitch import FPS, HOP

_N_FFT = 1024


def spectrogram(samples: np.ndarray, n_fft: int = _N_FFT, hop: int = HOP) -> np.ndarray:
    """Magnitude STFT, shape (frames, bins), on the same frame grid as the pitch model."""
    pad = n_fft // 2
    x = np.pad(samples.astype(np.float32, copy=False), (pad, pad))
    n = 1 + (len(x) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    out = np.empty((n, n_fft // 2 + 1), dtype=np.float32)
    step = 2048  # frames per block, keeps memory flat on long files
    view = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    for i in range(0, n, step):
        out[i : i + step] = np.abs(np.fft.rfft(view[i : i + step] * window, axis=1))
    return out


def onset_envelope(samples: np.ndarray, pitch_onsets: np.ndarray | None = None) -> np.ndarray:
    """Onset strength per frame: spectral flux over coarse log-spaced bands."""
    mag = spectrogram(samples)
    freqs = np.fft.rfftfreq(_N_FFT, 1 / SAMPLE_RATE)
    edges = np.geomspace(40, SAMPLE_RATE / 2, 41)
    bands = np.stack([mag[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
    bands = np.log1p(100 * bands)
    flux = np.maximum(bands[1:] - bands[:-1], 0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    if pitch_onsets is not None and len(pitch_onsets):
        n = min(len(flux), len(pitch_onsets))
        flux = flux[:n]
        note_flux = pitch_onsets[:n].sum(axis=1)
        if note_flux.std() > 0 and flux.std() > 0:
            flux = flux / flux.std() + 0.5 * note_flux / note_flux.std()
    # Remove the slow-moving loudness floor so only peaks remain.
    width = int(FPS * 0.4) | 1
    kernel = np.ones(width) / width
    flux = np.maximum(flux - np.convolve(flux, kernel, mode="same"), 0)
    std = flux.std()
    return (flux / std if std > 0 else flux).astype(np.float64)


def estimate_tempo(env: np.ndarray, min_bpm: float = 45.0, max_bpm: float = 220.0) -> float:
    """Global tempo from the autocorrelation of the onset envelope."""
    if len(env) < int(FPS * 2):
        return 120.0
    n = 1 << int(np.ceil(np.log2(len(env) * 2)))
    spec = np.fft.rfft(env - env.mean(), n)
    acf = np.fft.irfft(spec * np.conj(spec))[: len(env)]
    acf /= max(acf[0], 1e-9)
    lags = np.arange(len(acf))
    lo, hi = int(60 * FPS / max_bpm), min(int(60 * FPS / min_bpm), len(acf) // 2 - 1)
    if hi <= lo:
        return 120.0

    def at(lag: np.ndarray) -> np.ndarray:
        return np.interp(lag, lags, acf)

    cand = np.arange(lo, hi + 1, 0.25)
    # A real beat period also shows up at its multiples.
    score = at(cand) + 0.5 * at(cand * 2) + 0.25 * at(cand * 3) + 0.25 * at(cand * 4)
    bpm = 60 * FPS / cand
    # The pulse is equally periodic at half and double speed, so which level
    # counts as "the beat" comes down to this preference. Centred where it
    # resolved the most real recordings correctly (slow songs otherwise read
    # at double tempo).
    score = score * np.exp(-0.5 * (np.log2(bpm / 95.0) / 0.9) ** 2)
    return float(bpm[int(score.argmax())])


def track_beats(env: np.ndarray, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Dynamic-programming beat tracker (Ellis, 2007). Returns beat times in seconds."""
    period = 60.0 * FPS / bpm
    n = len(env)
    if n < period * 2:
        return np.array([])
    # Smooth so that a beat a frame or two away from an onset still scores.
    width = np.arange(-int(period / 8), int(period / 8) + 1)
    local = np.convolve(env, np.exp(-0.5 * (width * 32.0 / period) ** 2), mode="same")
    score = local.copy()
    back = np.full(n, -1, dtype=np.int64)
    offsets = np.arange(-int(round(2 * period)), -int(round(period / 2)))
    penalty = -tightness * np.log(-offsets / period) ** 2
    for i in range(n):
        j = i + offsets
        valid = j >= 0
        if not valid.any():
            continue
        cand = score[j[valid]] + penalty[valid]
        best = int(cand.argmax())
        score[i] = local[i] + cand[best]
        back[i] = j[valid][best]
    # Start the backtrace from the best beat near the end of the track.
    tail = max(0, n - int(period * 1.5))
    i = tail + int(score[tail:].argmax())
    beats = [i]
    while back[i] >= 0:
        i = int(back[i])
        beats.append(i)
    beats = np.array(beats[::-1], dtype=np.float64)
    # Drop leading/trailing beats that sit in silence.
    strength = local[beats.astype(int)]
    floor = 0.1 * np.median(strength[strength > 0]) if (strength > 0).any() else 0
    keep = np.nonzero(strength > floor)[0]
    if len(keep) >= 2:
        beats = beats[keep[0] : keep[-1] + 1]
    return beats / FPS


@dataclass
class BeatGrid:
    """Beats with bar structure. `beats[i]` is in bar `(i + offset) // per_bar`."""

    times: np.ndarray
    per_bar: int
    first_downbeat: int  # index into `times` of the first bar line

    @property
    def bpm(self) -> float:
        if len(self.times) < 2:
            return 120.0
        return float(60.0 / np.median(np.diff(self.times)))

    def position(self, seconds: np.ndarray | float) -> np.ndarray:
        """Map time to a fractional beat index, extrapolating past both ends."""
        t = np.atleast_1d(np.asarray(seconds, dtype=np.float64))
        times = self.times
        idx = np.interp(t, times, np.arange(len(times)))
        period = float(np.median(np.diff(times))) if len(times) > 1 else 0.5
        idx = np.where(t < times[0], (t - times[0]) / period, idx)
        idx = np.where(t > times[-1], len(times) - 1 + (t - times[-1]) / period, idx)
        return idx

    def time_of(self, beat_index: float) -> float:
        times = self.times
        period = float(np.median(np.diff(times))) if len(times) > 1 else 0.5
        if beat_index <= 0:
            return float(times[0] + beat_index * period)
        if beat_index >= len(times) - 1:
            return float(times[-1] + (beat_index - (len(times) - 1)) * period)
        lo = int(beat_index)
        return float(times[lo] + (beat_index - lo) * (times[lo + 1] - times[lo]))


def fallback_grid(duration: float, bpm: float = 120.0) -> BeatGrid:
    """A steady grid for audio with no detectable pulse."""
    period = 60.0 / bpm
    count = max(2, int(duration / period) + 1)
    return BeatGrid(np.arange(count) * period, 4, 0)


def find_bars(novelty: np.ndarray, accents: np.ndarray, hint: int | None = None) -> tuple[int, int]:
    """Pick the meter (3 or 4 beats per bar) and which beat is the downbeat.

    `novelty[i]` is how much the harmony changes at beat i and `accents[i]` how
    strong the low end is there: chords change and bass notes land on bar lines.
    """
    n = len(novelty)
    if n < 8:
        return hint or 4, 0

    def norm(v: np.ndarray) -> np.ndarray:
        s = v.std()
        return (v - v.mean()) / s if s > 0 else np.zeros_like(v)

    signal = norm(novelty) + 0.5 * norm(accents)
    best = (-np.inf, 4, 0)
    for per_bar in ([hint] if hint else [4, 3]):
        for phase in range(per_bar):
            on = signal[phase::per_bar]
            mask = np.ones(n, dtype=bool)
            mask[phase::per_bar] = False
            value = on.mean() - signal[mask].mean()
            if per_bar == 3:
                value -= 0.15  # 4/4 is far more common; need clear evidence for 3/4
            if value > best[0]:
                best = (value, per_bar, phase)
    return best[1], best[2]
