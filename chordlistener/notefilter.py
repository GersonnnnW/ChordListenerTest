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

from .pitch import FPS, MIDI_OFFSET, N_PITCHES, Activations, Note, decode_events

# Collect candidates well below the usual thresholds; the classifier does the judging.
CANDIDATE_ONSET = 0.3
CANDIDATE_FRAME = 0.2
CANDIDATE_SECONDS = 0.07

_BELOW = (12, 19, 24, 28, 31, 7)  # where a note's "parent" would be if it were only an overtone
_ABOVE = (12, 19, 24, 7)
FEATURES = (
    ["strength", "peak", "attack", "log_seconds", "key", "before", "after", "start", "steady", "has_attack"]
    + [f"below{k}" for k in _BELOW]
    + [f"below{k}_attack" for k in _BELOW]
    + [f"above{k}" for k in _ABOVE]
    + ["neighbour", "neighbour_attack", "crowd", "energy", "rank", "ratio_below", "frame_gain", "louder_same_time"]
)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "notefilter.json")
_model = None


def candidates(act: Activations, settings: tuple[float, float, float] | None = None) -> list[tuple[int, int, int, float, bool]]:
    onset, frame, seconds = settings or (CANDIDATE_ONSET, CANDIDATE_FRAME, CANDIDATE_SECONDS)
    return decode_events(act, onset, frame, seconds)


def features(act: Activations, events: list[tuple[int, int, int, float, bool]]) -> np.ndarray:
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


def filtered_notes(act: Activations, threshold: float | None = None) -> list[Note]:
    """Candidates the classifier believes in. A rejected re-attack extends the note it interrupted."""
    model = _load()
    if model is None:
        return []
    events = candidates(act, tuple(model["candidates"]))  # as generous as the model was trained on
    if not events:
        return []
    chance = probabilities(features(act, events), model)
    keep = chance >= (model["threshold"] if threshold is None else threshold)
    notes: list[Note] = []
    sounding: dict[int, Note] = {}
    for (t0, t1, f, strength, _), ok, p in zip(events, keep, chance):
        previous = sounding.get(f)
        if ok:
            note = Note(t0 / FPS, t1 / FPS, f + MIDI_OFFSET, min(1.0, strength), float(p))
            notes.append(note)
            sounding[f] = note
        elif previous is not None and t0 / FPS - previous.end <= 0.12:
            previous.end = max(previous.end, t1 / FPS)  # the key never stopped: one long note, not two
    return notes
