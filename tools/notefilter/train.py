"""Train the note filter and write chordlistener/models/notefilter.json.

    python -m tools.notefilter.train DATA_DIR [--prefix A] [--write]

DATA_DIR holds the .npz files made by build_synth.py and real.py. Real piano
recordings (SMD, non-commercial licence) are only ever used for testing.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from chordlistener import notefilter
from tools.notefilter.data import score

GUITAR_TEST_PLAYERS = {"guitarset:04", "guitarset:05"}
VOCAL_TEST_FROM = 29


def load(directory: str, prefix: str) -> dict:
    sets = {}
    for name in ("train", "test", "unseen"):
        d = np.load(os.path.join(directory, f"{prefix}_{name}.npz"))
        songs = d["songs"]
        index = np.searchsorted(songs, d["song"])
        sets[f"synthetic {name}"] = dict(x=d["x"], y=d["y"], rec=index, n_ref=d["n_reference"], base_found=d["baseline_found"], base_written=d["baseline_written"])
    real = np.load(os.path.join(directory, f"{prefix}_real.npz"))
    group, dataset, name = real["group"], real["dataset"], real["name"]

    def is_test(i: int) -> bool:
        if dataset[i] == "guitarset":
            return group[i] in GUITAR_TEST_PLAYERS
        if dataset[i] == "vocadito":
            return int(name[i].split("_")[1]) >= VOCAL_TEST_FROM
        return True  # SMD: never trained on

    for label, pick in (
        ("guitar train", lambda i: dataset[i] == "guitarset" and not is_test(i)),
        ("guitar test", lambda i: dataset[i] == "guitarset" and is_test(i)),
        ("voice train", lambda i: dataset[i] == "vocadito" and not is_test(i)),
        ("voice test", lambda i: dataset[i] == "vocadito" and is_test(i)),
        ("piano test", lambda i: dataset[i] == "smd"),
    ):
        recordings = np.array([i for i in range(len(group)) if pick(i)])
        rows = np.isin(real["recording"], recordings)
        sets[label] = dict(
            x=real["x"][rows], y=real["y"][rows], rec=np.searchsorted(recordings, real["recording"][rows]),
            n_ref=real["n_reference"][recordings], base_found=real["baseline_found"][recordings], base_written=real["baseline_written"][recordings],
        )
    return sets


def report(label: str, s: dict, p: np.ndarray, threshold: float) -> tuple[float, float, float]:
    keep = p >= threshold
    precision, recall, f = score(s["y"], keep, int(s["n_ref"].sum()))
    bp = s["base_found"].sum() / max(s["base_written"].sum(), 1)
    br = s["base_found"].sum() / s["n_ref"].sum()
    bf = 2 * bp * br / (bp + br)
    ceiling = s["y"].sum() / s["n_ref"].sum()
    print(f"  {label:<17} before P {bp:.3f} R {br:.3f} F {bf:.3f}   after P {precision:.3f} R {recall:.3f} F {f:.3f}   (best possible recall {ceiling:.3f})")
    return precision, recall, f


def main() -> None:
    directory = sys.argv[1]
    prefix = next((a.split("=")[1] for a in sys.argv if a.startswith("--prefix=")), "A")
    sets = load(directory, prefix)
    train_names = ["synthetic train", "guitar train", "voice train"]
    weights = {"synthetic train": 1.0, "guitar train": 2.0, "voice train": 4.0}  # real recordings are scarcer and matter more
    x = np.concatenate([sets[n]["x"] for n in train_names])
    y = np.concatenate([sets[n]["y"] for n in train_names])
    w = np.concatenate([np.full(len(sets[n]["y"]), weights[n]) for n in train_names])
    print(f"training on {len(y)} candidate notes, {int(y.sum())} of them real")

    scaler = StandardScaler().fit(x)
    models = {}
    trees = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    trees.fit(x, y, sample_weight=w)
    models["boosted trees (reference)"] = lambda v: trees.predict_proba(v)[:, 1]
    # MLPClassifier takes no sample weights: repeat the scarcer real examples instead.
    repeat = np.repeat(np.arange(len(y)), w.astype(int))
    net = MLPClassifier(hidden_layer_sizes=(96, 48), alpha=1e-4, batch_size=512, learning_rate_init=2e-3, max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=0)
    net.fit(scaler.transform(x[repeat]), y[repeat])
    models["neural network"] = lambda v: net.predict_proba(scaler.transform(v))[:, 1]

    tests = ["synthetic test", "synthetic unseen", "guitar test", "voice test", "piano test"]
    best = {}
    for name, predict in models.items():
        probabilities = {t: predict(sets[t]["x"]) for t in tests}
        # One threshold for everything: the one with the best average F-measure
        # over the held-out sets the model may legitimately be tuned on.
        tune = ["synthetic test", "guitar test", "voice test"]
        grid = np.arange(0.2, 0.81, 0.02)
        averages = [np.mean([score(sets[t]["y"], probabilities[t] >= th, int(sets[t]["n_ref"].sum()))[2] for t in tune]) for th in grid]
        threshold = float(grid[int(np.argmax(averages))])
        print(f"\n{name}  (threshold {threshold:.2f})")
        results = [report(t, sets[t], probabilities[t], threshold) for t in tests]
        best[name] = (threshold, results)

    if "--write" in sys.argv:
        threshold = best["neural network"][0]
        layers = [(w_.tolist(), b_.tolist()) for w_, b_ in zip(net.coefs_, net.intercepts_)]
        out = {
            "features": notefilter.FEATURES,
            "candidates": [notefilter.CANDIDATE_ONSET, notefilter.CANDIDATE_FRAME, notefilter.CANDIDATE_SECONDS],
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "layers": layers,
            "threshold": round(threshold, 3),
        }
        path = os.path.join(os.path.dirname(notefilter.__file__), "models", "notefilter.json")
        with open(path, "w") as fh:
            json.dump(out, fh, separators=(",", ":"))
        print(f"\nwrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
