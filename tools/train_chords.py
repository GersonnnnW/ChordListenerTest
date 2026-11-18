"""Train the chord recogniser on real recordings with timed chord labels.

    python -m tools.train_chords DATASETS [--cache FILE]

Uses GuitarSet's accompaniment recordings (CC BY 4.0): each has its chords
written out with their exact times. Every recording goes through the normal
analysis to get per-beat pitch profiles; the model learns, for any root, how
likely a beat is a major chord, a minor chord or neither (chordlistener/chords.py).

Players 00-02 train, 03 picks how much a chord change costs, 00-03 train the
final model, and 04-05 are only ever used to report the result. Writes
chordlistener/models/chords.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np

from chordlistener import chords, pipeline
from chordlistener.audio import load_audio
from chordlistener.pitch import FPS

NAMES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def label(value: str):
    """'A:min7/b3' -> (9, 'min'); 'N' -> None."""
    if ":" not in value:
        return None
    root, quality = value.split(":")
    quality = quality.split("/")[0].split("(")[0]
    return NAMES[root], "min" if quality.startswith(("min", "hdim", "dim")) else "maj"


def collect(root: str) -> list[dict]:
    """Per-beat treble/bass profiles and bar positions for every accompaniment recording."""
    captured: dict = {}
    original = chords.ChordRecognizer.decode

    def spy(self, treble, bass, bar_position=None, per_bar=4, **kw):
        captured.update(treble=treble, bass=bass, bar_position=bar_position)
        return original(self, treble, bass, bar_position, per_bar, **kw)

    def spy_segments(frames, boundaries):
        captured["starts"] = (boundaries[:-1] + pipeline._LEAD_FRAMES) / FPS
        captured["ends"] = (boundaries[1:] + pipeline._LEAD_FRAMES) / FPS
        return segment(frames, boundaries)

    segment = pipeline.segment_chroma
    chords.ChordRecognizer.decode, pipeline.segment_chroma = spy, spy_segments
    retune = pipeline._RETUNE_FROM
    pipeline._RETUNE_FROM = 1000.0  # no resampling, so beat times stay exact
    try:
        items = []
        for path in sorted(glob.glob(os.path.join(root, "guitarset", "ann", "*_comp.jams"))):
            name = os.path.basename(path)[:-5]
            pipeline.analyze_samples(load_audio(os.path.join(root, "guitarset", "audio", name + "_mic.wav")))
            data = json.load(open(path))
            marks = [a for a in data["annotations"] if a["namespace"] == "chord"][0]["data"]
            truth = [(c["time"], c["time"] + c["duration"], label(c["value"])) for c in marks]
            items.append({"name": name, "truth": truth, **{k: captured[k] for k in ("treble", "bass", "bar_position", "starts", "ends")}})
            print(name, flush=True)
        return items
    finally:
        chords.ChordRecognizer.decode, pipeline.segment_chroma = original, segment
        pipeline._RETUNE_FROM = retune


def _at(truth, t):
    for a, b, value in truth:
        if a <= t < b:
            return value if value else "N"
    return None


def beat_labels(item) -> list:
    out = []
    for s, e in zip(item["starts"], item["ends"]):
        votes: dict = {}
        for t in np.arange(s, e, 0.05):
            value = _at(item["truth"], t)
            if value is not None:
                votes[value] = votes.get(value, 0) + 1
        out.append(max(votes, key=votes.get) if votes else None)
    return out


def training_set(items):
    X, y = [], []
    for item in items:
        parts, presence = chords.beat_features(item["treble"], item["bass"], item["bar_position"])
        labels = beat_labels(item)
        for root in range(12):
            rotated = np.hstack([np.roll(p, -root, axis=1) for p in parts] + [presence[:, None]])
            for i, value in enumerate(labels):
                if value is None:
                    continue
                y.append(0 if value == "N" or value[0] != root else (1 if value[1] == "maj" else 2))
                X.append(rotated[i])
    return np.array(X), np.array(y)


def evaluate(items, model) -> tuple[float, float, float]:
    """(share of time the chord is right, share of real changes found within 0.25 s, share of found changes that are real)."""
    saved = dict(chords._model_cache)
    chords._model_cache["model"] = model
    recognizer = chords.ChordRecognizer("standard")
    right = total = found = changes = real = guessed = 0
    try:
        for item in items:
            spans = recognizer.decode(item["treble"], item["bass"], item["bar_position"])
            est = []
            for span in spans:
                value = None
                if span.chord is not None:
                    value = (span.chord.root, "min" if chords._minorish(span.chord.quality) else "maj")
                est.append((item["starts"][span.start_beat], item["ends"][span.end_beat - 1], value))
            for t in np.arange(0, item["truth"][-1][1], 0.05):
                want = _at(item["truth"], t)
                if want is None or want == "N":
                    continue
                have = next((v for a, b, v in est if a <= t < b), None)
                total += 1
                right += have == want
            true_changes = [a for (a, _, v), (_, _, w) in zip(item["truth"][1:], item["truth"]) if v != w]
            our_changes = [a for (a, _, v), (_, _, w) in zip(est[1:], est) if v != w]
            changes += len(true_changes)
            found += sum(any(abs(x - y) <= 0.25 for y in our_changes) for x in true_changes)
            guessed += len(our_changes)
            real += sum(any(abs(x - y) <= 0.25 for y in true_changes) for x in our_changes)
    finally:
        chords._model_cache.clear()
        chords._model_cache.update(saved)
    return right / max(total, 1), found / max(changes, 1), real / max(guessed, 1)


def fit(items, change: float) -> dict:
    from sklearn.linear_model import LogisticRegression

    X, y = training_set(items)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X, y)
    return {"coef": clf.coef_, "intercept": clf.intercept_, "change": change}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--cache", help="keep the per-beat profiles in this file between runs")
    args = parser.parse_args()
    if args.cache and os.path.exists(args.cache):
        items = pickle.load(open(args.cache, "rb"))
    else:
        items = collect(args.root)
        if args.cache:
            pickle.dump(items, open(args.cache, "wb"))
    player = lambda item: item["name"][:2]  # noqa: E731
    train = [i for i in items if player(i) in ("00", "01", "02")]
    valid = [i for i in items if player(i) == "03"]
    test = [i for i in items if player(i) in ("04", "05")]

    print("templates only (before):", "right %.1f%% | changes found %.1f%% | changes real %.1f%%" % tuple(100 * v for v in evaluate(test, None)))
    model = fit(train, 1.0)
    best = max((evaluate(valid, {**model, "change": c})[0], c) for c in (0.5, 1.0, 2.0, 3.0, 4.0))
    final = fit(train + valid, best[1])
    print(f"change cost {best[1]} (chosen on player 03)")
    print("learned model (after):  ", "right %.1f%% | changes found %.1f%% | changes real %.1f%%" % tuple(100 * v for v in evaluate(test, final)))
    with open(chords._MODEL_PATH, "w") as fh:
        json.dump({"coef": np.round(final["coef"], 5).tolist(), "intercept": np.round(final["intercept"], 5).tolist(), "change": final["change"]}, fh)
    print(f"wrote {chords._MODEL_PATH}")


if __name__ == "__main__":
    main()
