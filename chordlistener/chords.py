"""Chord and key recognition from pitch activations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .pitch import MIDI_OFFSET, N_PITCHES
from .theory import QUALITIES, VOCABULARIES, Chord, Key

_PITCHES = np.arange(N_PITCHES) + MIDI_OFFSET


def _register(center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((_PITCHES - center) / width) ** 2)


def _fold(weights: np.ndarray) -> np.ndarray:
    """(88, 12) matrix folding weighted piano keys onto pitch classes."""
    out = np.zeros((N_PITCHES, 12))
    out[np.arange(N_PITCHES), _PITCHES % 12] = weights
    return out


_TREBLE = _fold(_register(62, 14))
_BASS = _fold(_register(38, 7) * (_PITCHES < 55))


def segment_chroma(frames: np.ndarray, boundaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average pitch activations between consecutive frame boundaries.

    Returns (treble chroma, bass chroma), each (segments, 12).
    """
    cumulative = np.concatenate([np.zeros((1, N_PITCHES)), np.cumsum(frames, axis=0, dtype=np.float64)])
    b = np.clip(boundaries.astype(int), 0, len(frames))
    spans = np.maximum(b[1:] - b[:-1], 1)[:, None]
    mean = (cumulative[b[1:]] - cumulative[b[:-1]]) / spans
    return mean @ _TREBLE, mean @ _BASS


# Krumhansl-Kessler key profiles.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# How strongly each chord (interval above the tonic, quality family) points to a key.
_MAJOR_DEGREES = {(0, "maj"): 1.0, (5, "maj"): 0.7, (7, "maj"): 0.8, (9, "min"): 0.6, (2, "min"): 0.5, (4, "min"): 0.4, (10, "maj"): 0.15, (11, "dim"): 0.3}
_MINOR_DEGREES = {(0, "min"): 1.0, (5, "min"): 0.7, (7, "maj"): 0.8, (7, "min"): 0.5, (8, "maj"): 0.6, (3, "maj"): 0.5, (10, "maj"): 0.6, (2, "dim"): 0.3}
_FAMILY = {"maj": "maj", "7": "maj", "maj7": "maj", "6": "maj", "sus4": "maj", "sus2": "maj", "aug": "maj", "min": "min", "min7": "min", "min6": "min", "dim": "dim", "m7b5": "dim", "dim7": "dim"}


def estimate_key(chroma: np.ndarray, spans: "list[ChordSpan] | None" = None) -> tuple[Key, float]:
    """Key from the pitch-class distribution and the chord progression.

    Pitch statistics alone cannot tell a key from its relative (G major vs
    E minor); which chords carry the weight, and where the song starts and
    ends, usually can.
    """
    total = chroma.sum(axis=0)
    if total.sum() <= 0:
        return Key(0, False), 0.0
    voiced = [s for s in (spans or []) if s.chord is not None]
    beats = sum(s.end_beat - s.start_beat for s in voiced)
    scores = []
    for minor, profile, degrees in ((False, _MAJOR_PROFILE, _MAJOR_DEGREES), (True, _MINOR_PROFILE, _MINOR_DEGREES)):
        for tonic in range(12):
            score = 0.5 * float(np.corrcoef(total, np.roll(profile, tonic))[0, 1])
            if beats:
                fit = 0.0
                for s in voiced:
                    degree = ((s.chord.root - tonic) % 12, _FAMILY[s.chord.quality])
                    fit += (s.end_beat - s.start_beat) * degrees.get(degree, -0.4)
                score += fit / beats
                home = "min" if minor else "maj"
                for edge in (voiced[0], voiced[-1]):
                    if edge.chord.root == tonic and _FAMILY[edge.chord.quality] == home:
                        score += 0.1
            scores.append((score, tonic, minor))
    scores.sort(reverse=True)
    best, runner_up = scores[0], scores[1]
    confidence = float(np.clip(0.4 + (best[0] - runner_up[0]) * 2.0, 0.05, 1.0))
    return Key(best[1], best[2]), round(confidence, 3)


# Plain triads are the safe reading; richer chords have to earn their place.
# maj7 is penalised most because it is exactly "two neighbouring triads blurred
# together" (Gmaj7 = G + Bm), the commonest way to miss a quick change.
_BIAS = {"maj": 0.0, "min": 0.0, "7": -0.07, "min7": -0.06, "maj7": -0.075, "sus4": -0.06, "sus2": -0.06, "dim": -0.06, "aug": -0.06}


@dataclass
class ChordSpan:
    start_beat: int  # index of the first beat segment
    end_beat: int  # exclusive
    chord: Chord | None  # None = no chord
    confidence: float


class ChordRecognizer:
    """Template matching per beat, then Viterbi decoding over the song."""

    def __init__(self, vocabulary: str = "standard"):
        tier = VOCABULARIES.get(vocabulary, 1)
        self.states: list[tuple[int, str]] = []
        templates, bass_templates, bias = [], [], []
        for quality in QUALITIES.values():
            if quality.tier > tier:
                continue
            for root in range(12):
                t = np.zeros(12)
                for k, interval in enumerate(quality.intervals):
                    # The seventh/sixth is usually quieter than the triad.
                    t[(root + interval) % 12] = 1.0 if k < 3 else 0.7
                templates.append(t / np.linalg.norm(t))
                b = np.zeros(12)
                b[root] = 1.0
                bass_templates.append(b)
                self.states.append((root, quality.id))
                # Asking for the full vocabulary signals that extended chords are
                # expected, so they are held back less.
                bias.append(_BIAS.get(quality.id, -0.09) * (0.65 if tier == 2 else 1.0))
        self.templates = np.array(templates)
        self.bass_templates = np.array(bass_templates)
        self.bias = np.array(bias)
        self.n = len(self.states)

    def emissions(self, treble: np.ndarray, bass: np.ndarray) -> np.ndarray:
        """Score every chord (plus a trailing 'no chord' column) for every beat."""
        energy = treble.sum(axis=1)
        t = treble / np.maximum(np.linalg.norm(treble, axis=1, keepdims=True), 1e-9)
        b = bass / np.maximum(np.linalg.norm(bass, axis=1, keepdims=True), 1e-9)
        score = t @ self.templates.T + 0.22 * (b @ self.bass_templates.T) + self.bias
        # Normalising hides loudness, so a beat that only catches the edge of a
        # note would look like a clean chord. Fade chord scores out as the beat
        # gets quiet and let "no chord" win.
        reference = np.percentile(energy, 90) if len(energy) else 0.0
        presence = np.clip(energy / max(0.2 * reference, 1e-9), 0.0, 1.0)
        score = score * presence[:, None]
        no_chord = np.full(len(energy), 0.42)
        return np.concatenate([score, no_chord[:, None]], axis=1)

    def decode(self, treble: np.ndarray, bass: np.ndarray, bar_position: np.ndarray | None = None, per_bar: int = 4, change_cost: float = 0.22) -> list[ChordSpan]:
        n_seg = len(treble)
        if n_seg == 0:
            return []
        model = _learned()
        if model is not None:
            return self._decode_learned(model, treble, bass, bar_position)
        em = self.emissions(treble, bass)
        n_states = em.shape[1]
        # Chords mostly change on bar lines (then mid-bar), so changing
        # elsewhere costs more.
        cost = np.full(n_seg, change_cost)
        if bar_position is not None:
            cost = np.where(bar_position == 0, change_cost * 0.6, change_cost * 1.3)
            if per_bar == 4:
                cost = np.where(bar_position == 2, change_cost * 0.9, cost)
        score = em[0].copy()
        back = np.zeros((n_seg, n_states), dtype=np.int32)
        for i in range(1, n_seg):
            best = int(score.argmax())
            switch = score[best] - cost[i]
            stay = score >= switch
            back[i] = np.where(stay, np.arange(n_states), best)
            score = np.where(stay, score, switch) + em[i]
        path = np.empty(n_seg, dtype=np.int32)
        path[-1] = int(score.argmax())
        for i in range(n_seg - 1, 0, -1):
            path[i - 1] = back[i, path[i]]

        spans: list[ChordSpan] = []
        start = 0
        for i in range(1, n_seg + 1):
            if i == n_seg or path[i] != path[start]:
                state = int(path[start])
                chord = None
                confidence = 0.0
                if state < self.n:
                    root, quality = self.states[state]
                    seg_bass = bass[start:i].sum(axis=0)
                    chord = Chord(root, quality, _slash_bass(root, quality, seg_bass))
                    rival = np.partition(em[start:i, : self.n], -2, axis=1)[:, -2]
                    margin = float((em[start:i, state] - rival).mean())
                    confidence = float(np.clip(0.5 + margin * 4.0, 0.05, 1.0)) * float(np.clip(em[start:i, state].mean(), 0, 1))
                spans.append(ChordSpan(start, i, chord, round(confidence, 3)))
                start = i
        return spans


    def _decode_learned(self, model: dict, treble: np.ndarray, bass: np.ndarray, bar_position: np.ndarray | None) -> list[ChordSpan]:
        """Root and major/minor from the learned model; the exact chord type from the templates."""
        probs = learned_probabilities(model, treble, bass, bar_position)
        em = np.log(probs + 1e-6)
        n_seg, n_states = em.shape
        cost = model["change"]
        score = em[0].copy()
        back = np.zeros((n_seg, n_states), dtype=np.int32)
        for i in range(1, n_seg):
            best = int(score.argmax())
            switch = score[best] - cost
            stay = score >= switch
            back[i] = np.where(stay, np.arange(n_states), best)
            score = np.where(stay, score, switch) + em[i]
        path = np.empty(n_seg, dtype=np.int32)
        path[-1] = int(score.argmax())
        for i in range(n_seg - 1, 0, -1):
            path[i - 1] = back[i, path[i]]

        templates = self.emissions(treble, bass)[:, : self.n]
        spans: list[ChordSpan] = []
        start = 0
        for i in range(1, n_seg + 1):
            if i == n_seg or path[i] != path[start]:
                state = int(path[start])
                chord, confidence = None, 0.0
                if state < 24:
                    root, family = state % 12, ("maj" if state < 12 else "min")
                    # Which chord of that family (C, C7, Cmaj7, Csus4...) fits the notes best.
                    candidates = [k for k, (r, q) in enumerate(self.states) if r == root and _minorish(q) == (family == "min")]
                    k = max(candidates, key=lambda c: float(templates[start:i, c].mean())) if candidates else None
                    quality = self.states[k][1] if k is not None else family
                    chord = Chord(root, quality, _slash_bass(root, quality, bass[start:i].sum(axis=0)))
                    confidence = float(probs[start:i, state].mean())
                spans.append(ChordSpan(start, i, chord, round(confidence, 3)))
                start = i
        return spans


# ---------------------------------------------------------------- the learned model
#
# Trained on recordings with timed chord labels: real guitar (GuitarSet), real
# piano and voice (Schubert Winterreise) and full-band songs (AAM), each source
# weighted equally (tools/train_chords.py). One small logistic model, shared by
# all twelve roots: rotate the pitch-class profiles so the candidate root is at
# C, and it says how likely this beat is a major chord, a minor chord, or
# neither on that root. It sees the beat, its neighbours and its whole bar, so
# a passing note (the sixth in a boogie riff) does not flip the chord.

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "chords.json")
_model_cache: dict = {}


def _learned() -> dict | None:
    if "model" not in _model_cache:
        model = None
        if os.path.exists(_MODEL_PATH) and os.environ.get("CL_CHORD_MODEL", "learned") != "templates":
            with open(_MODEL_PATH) as fh:
                raw = json.load(fh)
            model = _parse_model(raw)
        _model_cache["model"] = model
    return _model_cache["model"]


def _parse_model(raw: dict) -> dict:
    """A one-layer (logistic) or small multi-layer model, as written by tools/train_chords.py."""
    if "layers" in raw:
        layers = [(np.array(w), np.array(b)) for w, b in raw["layers"]]
    else:
        layers = [(np.array(raw["coef"]).T, np.array(raw["intercept"]))]
    return {"layers": layers, "change": float(raw["change"])}


def _scores(model: dict, x: np.ndarray) -> np.ndarray:
    """Per row: (not this root, major, minor) scores. Hidden layers use ReLU."""
    layers = model["layers"]
    for w, b in layers[:-1]:
        x = np.maximum(x @ w + b, 0.0)
    w, b = layers[-1]
    return x @ w + b


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def beat_features(treble: np.ndarray, bass: np.ndarray, bar_position: np.ndarray | None) -> tuple[list[np.ndarray], np.ndarray]:
    """The profiles the model reads per beat (before rotation), and how present the music is."""
    energy = treble.sum(axis=1)
    reference = np.percentile(energy, 90) if len(energy) else 1.0
    presence = np.clip(energy / max(0.2 * reference, 1e-9), 0.0, 1.0)
    t, b = _unit(treble), _unit(bass)
    previous = np.vstack([t[:1], t[:-1]])
    following = np.vstack([t[1:], t[-1:]])
    bars = np.cumsum(bar_position == 0) if bar_position is not None else np.arange(len(t)) // 4
    bar_mean = np.zeros_like(t)
    for k in np.unique(bars):
        inside = bars == k
        bar_mean[inside] = _unit(treble[inside].sum(axis=0, keepdims=True))
    return [t, b, previous, following, bar_mean], presence


def learned_probabilities(model: dict, treble: np.ndarray, bass: np.ndarray, bar_position: np.ndarray | None) -> np.ndarray:
    """(beats, 25): probability of each major chord (C..B), each minor chord, and no chord."""
    parts, presence = beat_features(treble, bass, bar_position)
    n = len(presence)
    scores = np.zeros((n, 12, 3))
    for root in range(12):
        x = np.hstack([np.roll(p, -root, axis=1) for p in parts] + [presence[:, None]])
        scores[:, root] = _scores(model, x)
    # Per root: softmax over (not this root, major, minor).
    e = np.exp(scores - scores.max(axis=2, keepdims=True))
    p = e / e.sum(axis=2, keepdims=True)
    out = np.zeros((n, 25))
    out[:, :12] = p[:, :, 1]
    out[:, 12:24] = p[:, :, 2]
    out[:, 24] = np.exp(np.log(p[:, :, 0] + 1e-9).mean(axis=1)) ** 3 * np.exp(-6)
    out /= out.sum(axis=1, keepdims=True)
    # The recordings it learned from have hardly any silence, so a quiet beat is
    # "no chord" by how quiet it is, as in the template method.
    out *= presence[:, None]
    out[:, 24] += 1.0 - presence
    return out


def _minorish(quality: str) -> bool:
    """Minor family: a minor third and no major third (minor, m7, dim, m7b5...)."""
    intervals = QUALITIES[quality].intervals
    return 3 in intervals and 4 not in intervals


def _slash_bass(root: int, quality: str, bass: np.ndarray) -> int | None:
    """Report an inversion only when the bass clearly sits on another chord tone."""
    total = bass.sum()
    if total <= 0:
        return None
    tones = [(root + i) % 12 for i in QUALITIES[quality].intervals[:3]]
    low = int(bass.argmax())
    if low != root and low in tones and bass[low] > 0.55 * total and bass[low] > 2.5 * bass[root]:
        return low
    return None
