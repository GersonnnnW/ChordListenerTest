"""Is the recording at concert pitch?

Plenty of records are not: tape machines ran fast or slow, bands tune to each
other rather than to A = 440 Hz, and some tune low on purpose. When everything
sits a third of a semitone flat, every note straddles two piano keys and both
chord and note detection suffer. Measuring the offset and undoing it before
analysis fixes that.
"""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE

_N_FFT = 8192
_HOP = 4096


def estimate_tuning(samples: np.ndarray) -> tuple[float, float]:
    """How far the music sits from standard pitch, in cents (-50..50), and how sure we are (0..1).

    Finds the strongest spectral peaks, measures each one's distance to the
    nearest equal-tempered note, and takes the circular average: partials of
    in-tune notes cluster at one offset, noise and drums spread evenly.
    """
    if len(samples) < _N_FFT * 2:
        return 0.0, 0.0
    window = np.hanning(_N_FFT).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(samples.astype(np.float32, copy=False), _N_FFT)[::_HOP]
    view = view[:: max(1, len(view) // 400)]  # a few hundred frames are plenty
    mag = np.abs(np.fft.rfft(view * window, axis=1))
    lo, hi = int(100 * _N_FFT / SAMPLE_RATE), int(2000 * _N_FFT / SAMPLE_RATE)
    band = mag[:, lo - 1 : hi + 1]
    mid, left, right = band[:, 1:-1], band[:, :-2], band[:, 2:]
    peaks = (mid > left) & (mid >= right) & (mid > 0.1 * band.max(axis=1, keepdims=True)) & (mid > 0.02 * mag.max())
    if peaks.sum() < 50:
        return 0.0, 0.0
    # Parabolic interpolation on log magnitude gives the peak's true frequency.
    a, b, c = np.log(left[peaks] + 1e-9), np.log(mid[peaks] + 1e-9), np.log(right[peaks] + 1e-9)
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    rows, cols = np.nonzero(peaks)
    freq = (cols + lo + shift) * SAMPLE_RATE / _N_FFT
    cents = 1200 * np.log2(freq / 440.0)
    angle = 2 * np.pi * (cents % 100) / 100
    weight = mid[peaks]
    vector = (weight * np.exp(1j * angle)).sum() / weight.sum()
    offset = float(np.angle(vector) / (2 * np.pi) * 100)
    return offset, float(abs(vector))


def retune(samples: np.ndarray, cents: float) -> tuple[np.ndarray, float]:
    """Shift the audio to concert pitch by playing it slightly faster or slower.

    Returns the new samples and `stretch`: a time in the new audio multiplied
    by `stretch` is the matching time in the original.
    """
    stretch = float(2 ** (-cents / 1200))  # flat music (negative cents) is played faster
    positions = np.arange(0, len(samples) - 1, stretch)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32), stretch
