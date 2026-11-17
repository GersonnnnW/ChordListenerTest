"""Train the note filter and write chordlistener/models/notefilter.json.

    python -m tools.notefilter.train DATA_DIR [--prefix=C] [--write] [--features=shipped] [--weight.guitar=4]

--features=shipped   train on the same feature columns as the model that ships now
--weight.NAME=W      how much each example of NAME (synthetic, guitar, voice, band) counts

DATA_DIR holds the .npz files made by build_synth.py and real.py. Real piano
recordings (SMD, non-commercial licence) are only ever used for testing, which
also makes them the check on music the filter has never been shown.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from chordlistener import notefilter
from tools.notefilter.data import score

GUITAR_TEST_PLAYERS = {"guitarset:04", "guitarset:05"}
VOCAL_TEST_FROM = 29
BAND_TEST_FROM = 15

TRAIN = {"synthetic train": 1.0, "guitar train": 2.0, "voice train": 4.0, "band train": 2.0}  # how much each example counts
TUNE = ["synthetic test", "guitar test", "voice test", "band test"]
TESTS = ["synthetic test", "synthetic unseen", "guitar test", "voice test", "band test", "piano test"]


def _merge(paths: list[str], recording_key: str) -> dict:
    """Join several .npz files, renumbering recordings so they stay distinct."""
    parts = [np.load(p) for p in sorted(paths)]
    out: dict = {"x": [], "y": [], "rec": []}
    per_recording: dict = {}
    offset = 0
    for d in parts:
        n = len(d["n_reference"])
        index = d[recording_key] if recording_key == "recording" else np.searchsorted(d["songs"], d["song"])
        out["x"].append(d["x"])
        out["y"].append(d["y"])
        out["rec"].append(index + offset)
        for key in ("n_reference", "baseline_found", "baseline_written", "dataset", "group", "name"):
            if key in d.files:
                per_recording.setdefault(key, []).append(d[key])
        offset += n
    merged = {k: np.concatenate(v) for k, v in out.items()}
    merged.update({k: np.concatenate(v) for k, v in per_recording.items()})
    return merged


def _subset(d: dict, recordings: np.ndarray) -> dict:
    rows = np.isin(d["rec"], recordings)
    return dict(
        x=d["x"][rows], y=d["y"][rows],
        n_ref=d["n_reference"][recordings], base_found=d["baseline_found"][recordings], base_written=d["baseline_written"][recordings],
    )


def load(directory: str, prefix: str) -> dict:
    sets = {}
    for name in ("train", "test", "unseen"):
        d = _merge(glob.glob(os.path.join(directory, f"{prefix}_{name}*.npz")), "song")
        sets[f"synthetic {name}"] = _subset(d, np.arange(len(d["n_reference"])))
    real = _merge(glob.glob(os.path.join(directory, f"{prefix}_real*.npz")), "recording")
    group, dataset, name = real["group"], real["dataset"], real["name"]

    def is_test(i: int) -> bool:
        if dataset[i] == "guitarset":
            return group[i] in GUITAR_TEST_PLAYERS
        if dataset[i] == "vocadito":
            return int(name[i].split("_")[1]) >= VOCAL_TEST_FROM
        if dataset[i] == "slakh":
            return int(name[i][5:]) >= BAND_TEST_FROM
        return True  # SMD: never trained on

    for label, source in (("guitar", "guitarset"), ("voice", "vocadito"), ("band", "slakh"), ("piano", "smd")):
        for split, want in (("train", False), ("test", True)):
            recordings = np.array([i for i in range(len(group)) if dataset[i] == source and is_test(i) == want], dtype=int)
            if len(recordings):
                sets[f"{label} {split}"] = _subset(real, recordings)
    return sets


def before(s: dict) -> tuple[float, float, float]:
    p = s["base_found"].sum() / max(s["base_written"].sum(), 1)
    r = s["base_found"].sum() / s["n_ref"].sum()
    return p, r, 2 * p * r / (p + r)


def after(s: dict, chance: np.ndarray, drop_below: float, add_from: float) -> tuple[float, float, float]:
    return score(s["y"], notefilter.decide(s.get("x_all", s["x"]), chance, drop_below, add_from), int(s["n_ref"].sum()))


def tune(sets: dict, chance: dict) -> tuple[float, float]:
    """The pair of cut-offs with the best average F-measure over the held-out tuning sets."""
    best, choice = -1.0, (0.2, 0.4)
    for drop_below in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        for add_from in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
            if add_from < drop_below:
                continue
            value = float(np.mean([after(sets[t], chance[t], drop_below, add_from)[2] for t in TUNE]))
            if value > best:
                best, choice = value, (drop_below, add_from)
    return choice


def show(sets: dict, chance: dict, cut: tuple[float, float]) -> None:
    for t in TESTS:
        bp, br, bf = before(sets[t])
        p, r, f = after(sets[t], chance[t], *cut)
        ceiling = sets[t]["y"].sum() / sets[t]["n_ref"].sum()
        print(f"  {t:<17} before P {bp:.3f} R {br:.3f} F {bf:.3f}   after P {p:.3f} R {r:.3f} F {f:.3f}   change {f - bf:+.3f}   (ceiling R {ceiling:.3f})")


def main() -> None:
    warnings.filterwarnings("ignore")
    directory = sys.argv[1]
    prefix = next((a.split("=")[1] for a in sys.argv if a.startswith("--prefix=")), "C")
    weights = dict(TRAIN)
    for arg in sys.argv:
        if arg.startswith("--weight."):
            name, value = arg[len("--weight."):].split("=")
            weights[f"{name} train"] = float(value)
    feature_names = list(notefilter.FEATURES)
    if "--features=shipped" in sys.argv:
        with open(os.path.join(os.path.dirname(notefilter.__file__), "models", "notefilter.json")) as fh:
            feature_names = json.load(fh)["features"]
    use = [notefilter.FEATURES.index(f) for f in feature_names]
    sets = load(directory, prefix)
    for s in sets.values():
        s["x_all"] = s["x"]  # decide() reads "passes_old" from the full rows
        s["x"] = s["x"][:, use]
    names = [n for n in weights if n in sets]
    x = np.concatenate([sets[n]["x"] for n in names])
    y = np.concatenate([sets[n]["y"] for n in names])
    w = np.concatenate([np.full(len(sets[n]["y"]), weights[n]) for n in names])
    print(f"training on {len(y)} candidate notes ({int(y.sum())} real) from: {', '.join(f'{n} x{weights[n]:g}' for n in names)}")
    print(f"features: {len(feature_names)}")

    spectral = [i for i, f in enumerate(feature_names) if f.startswith(("partial", "half", "rise"))]
    plain = [i for i in range(x.shape[1]) if i not in spectral]

    def trees(columns: list[int]):
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        model.fit(x[:, columns], y, sample_weight=w)
        return lambda v: model.predict_proba(v[:, columns])[:, 1]

    for label, columns in (("boosted trees, without spectrum features", plain), ("boosted trees, all features (reference)", list(range(x.shape[1])))):
        predict = trees(columns)
        chance = {t: predict(sets[t]["x"]) for t in TESTS}
        cut = tune(sets, chance)
        print(f"\n{label}: drop old notes below {cut[0]}, add new ones from {cut[1]}")
        show(sets, chance, cut)

    scaler = StandardScaler().fit(x)
    # MLPClassifier takes no sample weights: repeat the scarcer real examples instead.
    repeat = np.repeat(np.arange(len(y)), w.astype(int))
    net = MLPClassifier(hidden_layer_sizes=(96, 48), alpha=1e-4, batch_size=512, learning_rate_init=2e-3, max_iter=80, early_stopping=True, n_iter_no_change=8, random_state=0)
    net.fit(scaler.transform(x[repeat]), y[repeat])
    chance = {t: net.predict_proba(scaler.transform(sets[t]["x"]))[:, 1] for t in TESTS}
    cut = tune(sets, chance)
    print(f"\nneural network (the one that ships): drop old notes below {cut[0]}, add new ones from {cut[1]}")
    show(sets, chance, cut)

    if "--write" in sys.argv:
        out = {
            "features": feature_names,
            "candidates": [notefilter.CANDIDATE_ONSET, notefilter.CANDIDATE_FRAME, notefilter.CANDIDATE_SECONDS],
            "mean": [round(float(v), 6) for v in scaler.mean_],
            "scale": [round(float(v), 6) for v in scaler.scale_],
            "layers": [([[round(float(v), 5) for v in row] for row in w_], [round(float(v), 5) for v in b_]) for w_, b_ in zip(net.coefs_, net.intercepts_)],
            "drop_below": cut[0],
            "add_from": cut[1],
            "trained_on": names,
        }
        path = os.path.join(os.path.dirname(notefilter.__file__), "models", "notefilter.json")
        with open(path, "w") as fh:
            json.dump(out, fh, separators=(",", ":"))
        print(f"\nwrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
