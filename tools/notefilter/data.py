"""Turn recordings with known notes into labelled examples for the note filter."""

from __future__ import annotations

import numpy as np

from chordlistener import notefilter
from chordlistener.pitch import FPS, MIDI_OFFSET, decode_notes, run_model
from chordlistener.tuning import estimate_tuning, retune

TOLERANCE = 0.07  # seconds: how close a detected attack must be to the real one


def match(detected: list[tuple[float, int]], reference: list[tuple[float, int]], tolerance: float = TOLERANCE) -> np.ndarray:
    """Pair detected (time, pitch) notes with reference ones, each reference used once. True where paired."""
    by_pitch: dict[int, list[float]] = {}
    for start, pitch in reference:
        by_pitch.setdefault(pitch, []).append(start)
    hit = np.zeros(len(detected), dtype=bool)
    # Closest pairs first, so a late re-detection cannot steal the match from the real attack.
    order = sorted(range(len(detected)), key=lambda i: min((abs(s - detected[i][0]) for s in by_pitch.get(detected[i][1], [])), default=9.0))
    for i in order:
        start, pitch = detected[i]
        starts = by_pitch.get(pitch)
        if not starts:
            continue
        j = int(np.argmin([abs(s - start) for s in starts]))
        if abs(starts[j] - start) <= tolerance:
            starts.pop(j)
            hit[i] = True
    return hit


def distinct(reference: list[tuple[float, int]], window: float = 0.03) -> list[tuple[float, int]]:
    """Two instruments striking the same pitch together are one note on the page."""
    out: list[tuple[float, int]] = []
    last: dict[int, float] = {}
    for start, pitch in sorted(reference):
        if pitch in last and start - last[pitch] <= window:
            continue
        last[pitch] = start
        out.append((start, pitch))
    return out


def prepare(samples: np.ndarray):
    """Normalise and bring to concert pitch exactly as the real pipeline does. Returns (activations, time stretch)."""
    peak = float(np.abs(samples).max()) or 1.0
    samples = samples * (0.9 / peak)
    cents, confidence = estimate_tuning(samples)
    stretch = 1.0
    if confidence >= 0.35 and abs(cents) >= 15:
        samples, stretch = retune(samples, cents)
    return run_model(samples), stretch


def examples(samples: np.ndarray, reference: list[tuple[float, int]]) -> dict:
    """Candidates, their features and whether each is a real note; plus what the fixed thresholds would give."""
    act, stretch = prepare(samples)
    reference = distinct([(start / stretch, pitch) for start, pitch in reference])
    events = notefilter.candidates(act)
    detected = [(t0 / FPS, f + MIDI_OFFSET) for t0, _, f, _, _ in events]
    baseline = [(n.start, n.pitch) for n in decode_notes(act)]
    return {
        "x": notefilter.features(act, events),
        "y": match(detected, reference),
        "n_reference": len(reference),
        "baseline_found": int(match(baseline, reference).sum()),
        "baseline_written": len(baseline),
    }


def score(y: np.ndarray, keep: np.ndarray, n_reference: int) -> tuple[float, float, float]:
    """Precision, recall and F-measure of the kept candidates."""
    found = int((y & keep).sum())
    precision = found / max(int(keep.sum()), 1)
    recall = found / max(n_reference, 1)
    return precision, recall, (2 * precision * recall / (precision + recall) if found else 0.0)
