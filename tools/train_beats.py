"""Train the tempo model on recordings with known beats.

    python -m tools.train_beats DATASETS [--aam N] [--cache FILE]

GuitarSet (real guitar, recorded to a click; CC BY 4.0) and AAM (full-band
songs from real instrument samples; CC BY 4.0, streamed from Zenodo) come with
every beat's time. Each recording goes through the onset analysis the app
uses; for its strongest candidate tempos the model learns which one is the
beat (chordlistener/beats.py). GuitarSet players 04-05 and every fifth AAM song
are kept out of training and used for the report. Writes
chordlistener/models/tempo.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import tempfile
import zipfile
from collections import Counter

import numpy as np

from chordlistener import beats
from chordlistener.audio import load_audio
from chordlistener.pitch import run_model

AAM_MIXES = "https://zenodo.org/api/records/5794629/files/0001-1000-audio-mixes.zip/content"
AAM_NOTES = "https://zenodo.org/api/records/5794629/files/0001-1000-annotations-v1.1.0.zip/content"


def _envelope(x: np.ndarray) -> np.ndarray:
    peak = np.abs(x).max()
    x = x * (0.9 / peak) if peak > 0 else x
    return beats.onset_envelope(x, run_model(x).onsets).astype(np.float32)


def collect(root: str, aam_songs: int, have: dict, cache: str | None) -> list[dict]:
    items = []

    def keep(item):
        items.append(item)
        have[item["name"]] = item
        print(item["source"], item["name"], flush=True)
        if cache and len(items) % 25 == 0:
            pickle.dump(list(have.values()), open(cache, "wb"))

    for path in sorted(glob.glob(os.path.join(root, "guitarset", "ann", "*.jams"))):
        name = os.path.basename(path)[:-5]
        if name in have:
            items.append(have[name])
            continue
        data = json.load(open(path))
        marks = [a for a in data["annotations"] if a["namespace"] == "beat_position"][0]["data"]
        x = load_audio(os.path.join(root, "guitarset", "audio", name + "_mic.wav"))
        keep({"name": name, "source": "guitarset", "env": _envelope(x), "beats": np.array([m["time"] for m in marks])})
    if aam_songs:
        from tools.remote_zip import RemoteFile

        mixes, notes = zipfile.ZipFile(RemoteFile(AAM_MIXES)), zipfile.ZipFile(RemoteFile(AAM_NOTES))
        names = set(notes.namelist())
        for k in range(1, aam_songs + 1):
            name = f"aam_{k:04d}"
            if name in have:
                items.append(have[name])
                continue
            info = next((n for n in names if n.endswith(f"{k:04d}_beatinfo.arff")), None)
            if info is None:
                continue
            times = [float(line.split(",")[0]) for line in notes.read(info).decode().splitlines() if line[:1].isdigit()]
            with tempfile.NamedTemporaryFile(suffix=".flac") as fh:
                fh.write(mixes.read(f"{k:04d}_mix.flac"))
                fh.flush()
                x = load_audio(fh.name)
            keep({"name": name, "source": "aam", "env": _envelope(x), "beats": np.array(times)})
    return items


def is_test(item) -> bool:
    if item["source"] == "guitarset":
        return item["name"][:2] in ("04", "05")
    return int(item["name"][4:]) % 5 == 0


def true_bpm(item) -> float:
    return 60.0 / float(np.median(np.diff(item["beats"])))


def verdict(bpm: float, truth: float) -> str:
    r = bpm / truth
    return "right" if abs(r - 1) < 0.04 else "half" if abs(r - 0.5) < 0.02 else "double" if abs(r - 2) < 0.08 else "other"


def report(items, pick, label) -> None:
    for source in ("guitarset", "aam"):
        test = [i for i in items if i["source"] == source and is_test(i)]
        if test:
            c = Counter(verdict(pick(i), true_bpm(i)) for i in test)
            print(f"  {label:10} {source:10} {len(test):3} recordings: tempo right {c['right'] / len(test):.0%}, half {c['half'] / len(test):.0%}, double {c['double'] / len(test):.0%}, other {c['other'] / len(test):.0%}")


def main() -> None:
    from sklearn.linear_model import LogisticRegression

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--aam", type=int, default=300)
    parser.add_argument("--cache")
    args = parser.parse_args()
    have = {}
    if args.cache and os.path.exists(args.cache):
        have = {i["name"]: i for i in pickle.load(open(args.cache, "rb"))}
    items = collect(args.root, args.aam, have, args.cache)
    if args.cache:
        pickle.dump(list(have.values()), open(args.cache, "wb"))

    X, y = [], []
    for item in items:
        if is_test(item):
            continue
        truth = true_bpm(item)
        for bpm, features in beats.tempo_candidates(item["env"]):
            X.append(features)
            y.append(int(abs(bpm / truth - 1) < 0.04))
    clf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(np.array(X), np.array(y))
    model = {"coef": clf.coef_[0], "intercept": float(clf.intercept_[0])}

    def learned(item):
        options = beats.tempo_candidates(item["env"])
        return options[int(np.argmax([f @ model["coef"] + model["intercept"] for _, f in options]))][0] if options else 120.0

    report(items, lambda i: beats.classic_tempo(i["env"]), "before")
    report(items, learned, "after")
    with open(beats._TEMPO_MODEL, "w") as fh:
        json.dump({"coef": np.round(model["coef"], 5).tolist(), "intercept": round(model["intercept"], 5)}, fh)
    print(f"wrote {beats._TEMPO_MODEL}")


if __name__ == "__main__":
    main()
