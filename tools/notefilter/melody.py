"""Train the melody model and write chordlistener/models/melody.json.

    python -m tools.notefilter.melody TRAIN.pkl[,TRAIN2.pkl...] TEST.pkl [--write]

Each .pkl holds a list of (unseen, seed, melody notes [(start, pitch)], Analysis)
for generated songs, made with the analysis pipeline exactly as it runs.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from chordlistener import arrange as A
from chordlistener import melody
from tools.notefilter.data import match


def labelled(rows):
    xs, ys = [], []
    for _, _, lead, analysis in rows:
        x = melody.features(analysis)
        y = match([(n.start, n.pitch) for n in analysis.notes], lead, 0.07)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def line_score(rows, chance_of, threshold: float, leap_cost: float) -> dict:
    """How well the chosen line matches the real tune (precision, recall), split by seen/unseen instruments."""
    totals = {False: [0, 0, 0], True: [0, 0, 0]}
    for unseen, _, lead, analysis in rows:
        total = (len(analysis.grid.times) - 1) * 4
        sources: list[int] = []
        notes, _ = A.quantise(analysis, total, sources=sources)
        if not notes:
            totals[unseen][2] += len(lead)
            continue
        chance = chance_of(analysis)[sources]
        starts = np.array([n[0] for n in notes])
        ends = np.array([n[0] + n[1] for n in notes])
        pitch = np.array([n[2] for n in notes], dtype=float)
        line = melody.best_line(starts, ends, pitch, chance, threshold, leap_cost)
        beats = np.array(analysis.grid.times)
        page = [(float(np.interp(notes[i][0] / 4, np.arange(len(beats)), beats)), notes[i][2]) for i in line]
        t = totals[unseen]
        t[0] += int(match(page, lead, 0.1).sum())
        t[1] += len(page)
        t[2] += len(lead)
    out = {}
    for unseen, (hit, written, truth) in totals.items():
        if truth:
            p, r = hit / max(written, 1), hit / truth
            out["unseen" if unseen else "seen"] = (p, r, 2 * p * r / max(p + r, 1e-9))
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    train_rows = [r for path in sys.argv[1].split(",") for r in pickle.load(open(path, "rb"))]
    test_rows = pickle.load(open(sys.argv[2], "rb"))
    tune_rows = [r for r in train_rows if r[1] % 4 == 3]  # a quarter held back for choosing the cut-offs
    fit_rows = [r for r in train_rows if r[1] % 4 != 3]
    x, y = labelled(fit_rows)
    print(f"training on {len(y)} notes from {len(fit_rows)} songs, {int(y.sum())} of them melody")
    scaler = StandardScaler().fit(x)
    net = MLPClassifier(hidden_layer_sizes=(48, 24), alpha=1e-3, batch_size=256, learning_rate_init=2e-3, max_iter=120, early_stopping=True, n_iter_no_change=10, random_state=0)
    net.fit(scaler.transform(x), y)
    chance_of = lambda analysis: net.predict_proba(scaler.transform(melody.features(analysis)))[:, 1]  # noqa: E731

    best, choice = -1.0, (0.3, 0.3)
    for threshold in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        for leap_cost in (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0):
            value = line_score(tune_rows, chance_of, threshold, leap_cost)["seen"][2]
            if value > best:
                best, choice = value, (threshold, leap_cost)
    print(f"chosen: threshold {choice[0]}, leap cost {choice[1]} (tuning F {best:.3f})")

    def old_tracker(rows):
        totals = {}
        for unseen, _, lead, analysis in rows:
            total = (len(analysis.grid.times) - 1) * 4
            notes, _ = A.quantise(analysis, total)
            line = A.find_melody(notes)
            beats = np.array(analysis.grid.times)
            page = [(float(np.interp(s / 4, np.arange(len(beats)), beats)), p) for s, _, p, _ in line]
            t = totals.setdefault("unseen" if unseen else "seen", [0, 0, 0])
            t[0] += int(match(page, lead, 0.1).sum())
            t[1] += len(page)
            t[2] += len(lead)
        return {k: (h / max(w, 1), h / n, 2 * (h / max(w, 1)) * (h / n) / max(h / max(w, 1) + h / n, 1e-9)) for k, (h, w, n) in totals.items()}

    before, after = old_tracker(test_rows), line_score(test_rows, chance_of, *choice)
    for kind in ("seen", "unseen"):
        b, a = before[kind], after[kind]
        print(f"  test, {kind:<6} instruments: before P {b[0]:.3f} R {b[1]:.3f} F {b[2]:.3f}   after P {a[0]:.3f} R {a[1]:.3f} F {a[2]:.3f}")

    if "--write" in sys.argv:
        out = {
            "features": melody.FEATURES,
            "mean": [round(float(v), 6) for v in scaler.mean_],
            "scale": [round(float(v), 6) for v in scaler.scale_],
            "layers": [([[round(float(v), 5) for v in row] for row in w], [round(float(v), 5) for v in b]) for w, b in zip(net.coefs_, net.intercepts_)],
            "threshold": choice[0],
            "leap_cost": choice[1],
        }
        path = os.path.join(os.path.dirname(melody.__file__), "models", "melody.json")
        with open(path, "w") as fh:
            json.dump(out, fh, separators=(",", ":"))
        print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
