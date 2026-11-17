"""Which detected notes are the tune?

The tune is usually, but not always, the highest note sounding; it moves in
small steps; it lasts; it sits in a singable range; and it is played or sung
by one voice, so it never has two notes at once. A small classifier scores
each detected note for "this is the tune" from features like these, and a
dynamic-programming pass then picks the single best line through the song
(see tools/notefilter/melody.py for training).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .pipeline import Analysis
from .theory import QUALITIES

_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10, 11)

FEATURES = [
    "pitch", "strength", "confidence", "log_seconds",
    "together", "higher_together", "higher_sounding", "lower_sounding", "gap_to_top",
    "above_local", "above_skyline", "near_before", "near_after",
    "in_chord", "in_scale", "off_beat", "relative_strength",
]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "melody.json")
_model = None


def features(analysis: Analysis) -> np.ndarray:
    notes = analysis.notes
    n = len(notes)
    out = np.zeros((n, len(FEATURES)), dtype=np.float32)
    if not n:
        return out
    start = np.array([x.start for x in notes])
    end = np.array([x.end for x in notes])
    pitch = np.array([x.pitch for x in notes], dtype=np.float64)
    strength = np.array([x.velocity for x in notes])
    confidence = np.array([x.confidence for x in notes])
    order = np.argsort(start, kind="stable")
    start_sorted = start[order]

    # The chord sounding at each note, and the key's scale.
    chord_tones = [frozenset()] * n
    spans = [(c.start, c.end, frozenset((c.chord.root + i) % 12 for i in QUALITIES[c.chord.quality].intervals)) for c in analysis.chords if c.chord]
    for i in range(n):
        for lo, hi, tones in spans:
            if lo - 0.05 <= start[i] < hi:
                chord_tones[i] = tones
                break
    scale = {(analysis.key.tonic + d) % 12 for d in (_MINOR if analysis.key.minor else _MAJOR)}
    beat = analysis.grid.position(start) % 1.0
    off_beat = np.minimum(beat, 1.0 - beat)

    # The highest note starting at each onset, for a local "skyline".
    tops = {}
    for i in range(n):
        key = round(start[i] / 0.05)
        tops[key] = max(tops.get(key, 0.0), pitch[i])
    top_times = np.array(sorted(tops)) * 0.05
    top_pitch = np.array([tops[k] for k in sorted(tops)])

    for i in range(n):
        s, p = start[i], pitch[i]
        lo, hi = np.searchsorted(start_sorted, [s - 0.05, s + 0.05])
        together = order[lo:hi]
        lo4, hi4 = np.searchsorted(start_sorted, [s - 4.0, s + 4.0])
        nearby = order[lo4:hi4]
        sounding = nearby[(start[nearby] <= s + 0.02) & (end[nearby] > s + 0.02)]
        t_lo, t_hi = np.searchsorted(top_times, [s - 4.0, s + 4.0])
        lb, hb = np.searchsorted(start_sorted, [s - 0.6, s - 0.05])
        la, ha = np.searchsorted(start_sorted, [s + 0.05, s + 0.6])
        before, after = order[lb:hb], order[la:ha]
        loudest = strength[together].max() if len(together) else strength[i]
        out[i] = [
            (p - 60) / 12,
            strength[i],
            confidence[i],
            np.log(max(end[i] - s, 0.03)),
            len(together),
            float((pitch[together] > p).sum()),
            float((pitch[sounding] > p).sum()),
            float((pitch[sounding] < p).sum()),
            (max(pitch[sounding].max() if len(sounding) else p, p) - p) / 12,
            (p - np.median(pitch[nearby])) / 12,
            (p - np.median(top_pitch[t_lo:t_hi])) / 12 if t_hi > t_lo else 0.0,
            min(np.abs(pitch[before] - p).min() if len(before) else 24, 24) / 12,
            min(np.abs(pitch[after] - p).min() if len(after) else 24, 24) / 12,
            1.0 if int(p) % 12 in chord_tones[i] else 0.0,
            1.0 if int(p) % 12 in scale else 0.0,
            off_beat[i],
            strength[i] / max(loudest, 1e-3),
        ]
    return out


def _load() -> dict | None:
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            _model = {}
        else:
            with open(_MODEL_PATH) as fh:
                raw = json.load(fh)
            raw["layers"] = [(np.array(w, dtype=np.float32), np.array(b, dtype=np.float32)) for w, b in raw["layers"]]
            raw["mean"], raw["scale"] = np.array(raw["mean"], dtype=np.float32), np.array(raw["scale"], dtype=np.float32)
            _model = raw
    return _model or None


def available() -> bool:
    return _load() is not None


def probabilities(x: np.ndarray, model: dict | None = None) -> np.ndarray:
    model = model or _load()
    h = (x - model["mean"]) / model["scale"]
    for i, (w, b) in enumerate(model["layers"]):
        h = h @ w + b
        if i < len(model["layers"]) - 1:
            h = np.maximum(h, 0)
    return 1.0 / (1.0 + np.exp(-h[:, 0]))


def best_line(starts: np.ndarray, ends: np.ndarray, pitch: np.ndarray, chance: np.ndarray, threshold: float, leap_cost: float) -> list[int]:
    """Indices of the single most likely tune: one note at a time, each worth its chance, leaps cost extra."""
    n = len(starts)
    if not n:
        return []
    order = np.lexsort((-chance, starts))
    starts, ends, pitch, chance = starts[order], ends[order], pitch[order], chance[order]
    gain = chance - threshold
    best = gain.copy()
    back = np.full(n, -1)
    for j in range(n):
        lo = max(0, j - 64)
        prev = np.arange(lo, j)
        prev = prev[starts[prev] < starts[j]]
        if not len(prev):
            continue
        leap = np.abs(pitch[prev] - pitch[j])
        rest = np.maximum(starts[j] - ends[prev], 0)
        cost = leap_cost * np.maximum(leap - 2, 0) / 5.0 * np.exp(-rest / 16.0)
        total = best[prev] - cost
        k = int(total.argmax())
        if total[k] > 0 and total[k] + gain[j] > best[j]:
            best[j] = total[k] + gain[j]
            back[j] = prev[k]
    j = int(best.argmax())
    if best[j] <= 0:
        return []
    line = []
    while j >= 0:
        line.append(int(order[j]))
        j = int(back[j])
    return line[::-1]
