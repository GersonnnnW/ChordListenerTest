"""Telling played notes from phantom ones.

The pitch network lights up for more than the notes that were played: an
octave above a strong note, a string that is still ringing, a drum hit. A
fixed loudness threshold cannot separate these from real but quiet notes, so
it either lets phantoms through or throws real notes away.

Instead, candidate notes are collected generously and each is judged by a
small classifier that looks at the note in context: how it starts, what is
sounding an octave or a twelfth below it, whether the same key was already
ringing, how crowded the moment is. The classifier is trained on thousands of
notes from recordings where the truth is known (see tools/notefilter).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .audio import SAMPLE_RATE
from .pitch import FPS, HOP, MIDI_OFFSET, N_PITCHES, Activations, Note, decode_events

# Extra candidates are collected well below the usual thresholds; the classifier does the judging.
CANDIDATE_ONSET = 0.2
CANDIDATE_FRAME = 0.15
CANDIDATE_SECONDS = 0.07
OLD_STRENGTH = 0.35  # what the fixed thresholds used to demand

# (first frame, end frame, key, mean activation, has a clear attack, found by the fixed thresholds too)
Candidate = tuple[int, int, int, float, bool, bool]

_BELOW = (12, 19, 24, 28, 31, 7)  # where a note's "parent" would be if it were only an overtone
_ABOVE = (12, 19, 24, 7)
FEATURES = (
    ["strength", "peak", "attack", "log_seconds", "key", "before", "after", "start", "steady", "has_attack"]
    + [f"below{k}" for k in _BELOW]
    + [f"below{k}_attack" for k in _BELOW]
    + [f"above{k}" for k in _ABOVE]
    + ["neighbour", "neighbour_attack", "crowd", "energy", "rank", "ratio_below", "frame_gain", "louder_same_time"]
    + ["old", "passes_old"]
    + ["partial1", "partial2", "partial3", "partial4", "half", "rise"]  # read from the audio spectrum itself
)
_N_FFT = 4096
_SPEC_HOP = 512

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "notefilter.json")
_model = None


def candidates(act: Activations, settings: tuple[float, float, float] | None = None) -> list[Candidate]:
    """Every note the fixed thresholds find, plus fainter ones they would have missed."""
    onset, frame, seconds = settings or (CANDIDATE_ONSET, CANDIDATE_FRAME, CANDIDATE_SECONDS)
    old = decode_events(act)
    starts: dict[int, list[int]] = {}
    for t0, _, f, _, _ in old:
        starts.setdefault(f, []).append(t0)
    out: list[Candidate] = [(*event, True) for event in old]
    for event in decode_events(act, onset, frame, seconds):
        if not any(abs(event[0] - t) <= 4 for t in starts.get(event[2], ())):
            out.append((*event, False))
    out.sort()
    return out


def spectrum(samples: np.ndarray) -> np.ndarray:
    """Decibels above each moment's typical level, on a fine frequency grid."""
    pad = _N_FFT // 2
    x = np.pad(samples.astype(np.float32, copy=False), (pad, pad))
    view = np.lib.stride_tricks.sliding_window_view(x, _N_FFT)[::_SPEC_HOP]
    window = np.hanning(_N_FFT).astype(np.float32)
    out = np.empty((len(view), _N_FFT // 2 + 1), dtype=np.float32)
    for i in range(0, len(view), 1024):
        out[i : i + 1024] = 20 * np.log10(np.abs(np.fft.rfft(view[i : i + 1024] * window, axis=1)) + 1e-6)
    out -= np.median(out[:, 20:1500], axis=1, keepdims=True)
    return out


def _spectral(spec: np.ndarray, t0: np.ndarray, t1: np.ndarray, key: np.ndarray) -> list[np.ndarray]:
    """Is there energy where this note's partials should be, and did it just appear?"""
    n = len(spec)
    f0 = 27.5 * 2 ** (key / 12.0)
    first = np.clip(t0 * HOP // _SPEC_HOP, 0, n - 1)
    last = np.clip(np.maximum(t1 * HOP // _SPEC_HOP, first + 1), 1, n)
    head = np.clip(first[:, None] + np.arange(4)[None, :], 0, n - 1)
    head = np.minimum(head, (last - 1)[:, None])  # the note's first tenth of a second
    before = np.clip(first[:, None] - 1 - np.arange(4)[None, :], 0, n - 1)

    def level(freq: np.ndarray, frames: np.ndarray) -> np.ndarray:
        bins = np.clip(np.round(freq * _N_FFT / SAMPLE_RATE).astype(int), 1, spec.shape[1] - 2)
        near = np.stack([spec[frames, (bins + d)[:, None]] for d in (-1, 0, 1)])  # allow for slightly off-pitch notes
        value = near.max(axis=0).mean(axis=1)
        return np.where(freq < SAMPLE_RATE / 2 - 50, np.clip(value, -20, 60) / 20.0, -1.0)

    partials = [level(f0 * h, head) for h in (1, 2, 3, 4)]
    return [*partials, level(f0 / 2, head), partials[0] - level(f0, before)]


def features(act: Activations, events: list[Candidate], spec: np.ndarray) -> np.ndarray:
    """One row of numbers per candidate note, describing it and its surroundings."""
    frames, onsets = act.frames, act.onsets
    n_frames = frames.shape[0]
    out = np.zeros((len(events), len(FEATURES)), dtype=np.float32)
    if not events:
        return out
    t0 = np.array([e[0] for e in events])
    t1 = np.array([e[1] for e in events])
    key = np.array([e[2] for e in events])
    strength = np.array([e[3] for e in events], dtype=np.float64)
    old = np.array([1.0 if e[5] else 0.0 for e in events])

    # Running sums make "average activation of key f over frames a..b" a subtraction.
    total = np.concatenate([np.zeros((1, N_PITCHES)), np.cumsum(frames, axis=0, dtype=np.float64)])
    strong = np.concatenate([np.zeros((1, N_PITCHES)), np.cumsum(frames > 0.5, axis=0, dtype=np.float64)])
    # Strongest attack within two frames either side, for every frame and key.
    padded = np.pad(onsets, ((2, 2), (0, 0)))
    nearby = np.max([padded[i : i + n_frames] for i in range(5)], axis=0)
    active = (frames > 0.3).sum(axis=1)
    energy = frames.sum(axis=1)

    def mean(a: np.ndarray, b: np.ndarray, f: np.ndarray, sums: np.ndarray = total) -> np.ndarray:
        valid = (f >= 0) & (f < N_PITCHES)
        fc = np.clip(f, 0, N_PITCHES - 1)
        a, b = np.clip(a, 0, n_frames), np.clip(b, 0, n_frames)
        value = (sums[b, fc] - sums[a, fc]) / np.maximum(b - a, 1)
        return np.where(valid & (b > a), value, 0.0)

    def attack(f: np.ndarray) -> np.ndarray:
        valid = (f >= 0) & (f < N_PITCHES)
        return np.where(valid, nearby[t0, np.clip(f, 0, N_PITCHES - 1)], 0.0)

    peak = np.empty(len(events))
    for i, (a, b, f) in enumerate(zip(t0.tolist(), t1.tolist(), key.tolist())):
        peak[i] = frames[a:b, f].max()

    at = np.minimum(t0 + 2, n_frames - 1)
    here = frames[at, key]
    own_attack = attack(key)
    below = [mean(t0, t1, key - k) for k in _BELOW]
    columns = [
        strength,
        peak,
        own_attack,
        np.log((t1 - t0) / FPS),
        key / N_PITCHES,
        mean(t0 - 8, t0 - 2, key),  # was this key already sounding?
        mean(t1, t1 + 6, key),
        mean(t0, t0 + 5, key),
        mean(t0, t1, key, strong),
        np.array([1.0 if e[4] else 0.0 for e in events]),
        *below,
        *[attack(key - k) for k in _BELOW],
        *[mean(t0, t1, key + k) for k in _ABOVE],
        np.maximum(mean(t0, t1, key - 1), mean(t0, t1, key + 1)),
        np.maximum(attack(key - 1), attack(key + 1)),
        active[at].astype(np.float64),
        energy[at],
        (frames[at] > here[:, None]).sum(axis=1) / np.maximum(active[at], 1),
        strength / np.maximum(np.max(below[:5], axis=0), 0.05),
        here - frames[np.maximum(t0 - 3, 0), key],
        (nearby[t0] > own_attack[:, None]).sum(axis=1).astype(np.float64),
        old,
        old * (strength >= OLD_STRENGTH),
        *_spectral(spec, t0, t1, key),
    ]
    return np.stack(columns, axis=1).astype(np.float32)


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


def probabilities(x: np.ndarray, model: dict | None = None) -> np.ndarray:
    """Chance that each candidate is a note someone played (a small neural network, run in numpy)."""
    model = model or _load()
    if model is None:
        raise RuntimeError("The note filter model is missing")
    h = (x - model["mean"]) / model["scale"]
    for i, (w, b) in enumerate(model["layers"]):
        h = h @ w + b
        if i < len(model["layers"]) - 1:
            h = np.maximum(h, 0)
    return 1.0 / (1.0 + np.exp(-h[:, 0]))


def available() -> bool:
    return _load() is not None


def decide(x: np.ndarray, chance: np.ndarray, drop_below: float, add_from: float) -> np.ndarray:
    """Which candidates to keep.

    The fixed thresholds are a sound default on any kind of music, so their
    notes are only removed when the classifier is confident they are false,
    and other candidates are only added when it is confident they are real.
    """
    passes_old = x[:, FEATURES.index("passes_old")] > 0.5
    return (passes_old & (chance >= drop_below)) | (chance >= add_from)


def filtered_notes(act: Activations, samples: np.ndarray) -> list[Note]:
    """The notes to write down, each with the classifier's confidence in it."""
    model = _load()
    if model is None:
        return []
    events = candidates(act, tuple(model["candidates"]))  # as generous as the model was trained on
    if not events:
        return []
    x = features(act, events, spectrum(samples))
    chance = probabilities(x, model)
    keep = decide(x, chance, model["drop_below"], model["add_from"])
    notes: list[Note] = []
    sounding: dict[int, Note] = {}
    for (t0, t1, f, strength, _, _), ok, p in zip(events, keep, chance):
        previous = sounding.get(f)
        start, end = t0 / FPS, t1 / FPS
        if ok:
            if previous is not None and previous.end > start:
                previous.end = start  # the same key struck again: the earlier note stops here
            note = Note(start, end, f + MIDI_OFFSET, min(1.0, strength), float(p))
            notes.append(note)
            sounding[f] = note
        elif previous is not None and start - previous.end <= 0.12:
            previous.end = max(previous.end, end)  # the key never stopped: one long note, not two
    return [n for n in notes if n.end - n.start >= 0.03]
