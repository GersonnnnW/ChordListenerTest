"""Train the chord recogniser on recordings with timed chord labels.

    python -m tools.train_chords DATASETS [--cache FILE] [--aam N]

Three sources, all openly licensed, each split so the reported numbers come
from recordings never trained on:

  GuitarSet   real guitar accompaniment (CC BY 4.0)        players 00-02 train, 03 tunes, 04-05 test
  Winterreise real piano and voice, Schubert (CC BY 3.0)  performance HU33 trains, SC06 tests
  AAM         full-band songs from real instrument        first 75% train, next 12.5% tune, last 12.5% test
              samples, composed by computer (CC BY 4.0)    (N songs streamed from Zenodo, default 400)

Expected under DATASETS: guitarset/{ann,audio} and swd/ (the Winterreise zip
unpacked). Every recording goes through the normal analysis to get per-beat
pitch profiles; the model learns, for any root, how likely a beat is a major
chord, a minor chord or neither (chordlistener/chords.py). The change cost is
chosen on the tuning recordings, balancing right chords against clean change
times. Writes chordlistener/models/chords.json.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import tempfile
import zipfile

import numpy as np

from chordlistener import chords, pipeline
from chordlistener.audio import load_audio
from chordlistener.pitch import FPS

NAMES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11, "B#": 0}
AAM_URL = "https://zenodo.org/api/records/5794629/files/{part}-audio-mixes.zip/content"
AAM_ANN = "https://zenodo.org/api/records/5794629/files/{part}-annotations-v1.1.0.zip/content"


def label(value: str):
    """'A:min7/b3', 'Amin', 'A#maj' -> (root, 'maj' | 'min'); 'N', 'N.C.', 'X' -> None."""
    value = value.strip().strip('"').strip("'")
    if ":" in value:
        root, quality = value.split(":", 1)
    else:
        m = re.match(r"^([A-G][#b]?)(.*)$", value)
        if not m:
            return None
        root, quality = m.groups()
    if root not in NAMES:
        return None
    quality = quality.split("/")[0].split("(")[0]
    return NAMES[root], "min" if quality.startswith(("min", "hdim", "dim")) else "maj"


def analyse(samples: np.ndarray) -> dict:
    """Per-beat treble/bass profiles, bar positions and beat times, as the analysis sees them."""
    captured: dict = {}
    original, segment = chords.ChordRecognizer.decode, pipeline.segment_chroma

    def spy(self, treble, bass, bar_position=None, per_bar=4, **kw):
        captured.update(treble=treble.astype(np.float32), bass=bass.astype(np.float32), bar_position=bar_position)
        return original(self, treble, bass, bar_position, per_bar, **kw)

    def spy_segments(frames, boundaries):
        captured["starts"] = (boundaries[:-1] + pipeline._LEAD_FRAMES) / FPS
        captured["ends"] = (boundaries[1:] + pipeline._LEAD_FRAMES) / FPS
        return segment(frames, boundaries)

    chords.ChordRecognizer.decode, pipeline.segment_chroma = spy, spy_segments
    retune = pipeline._RETUNE_FROM
    pipeline._RETUNE_FROM = 1000.0  # no resampling, so beat times stay exact
    try:
        pipeline.analyze_samples(samples)
    finally:
        chords.ChordRecognizer.decode, pipeline.segment_chroma = original, segment
        pipeline._RETUNE_FROM = retune
    return captured


def guitarset(root: str):
    for path in sorted(glob.glob(os.path.join(root, "guitarset", "ann", "*_comp.jams"))):
        name = os.path.basename(path)[:-5]
        data = json.load(open(path))
        marks = [a for a in data["annotations"] if a["namespace"] == "chord"][0]["data"]
        truth = [(c["time"], c["time"] + c["duration"], label(c["value"])) for c in marks]
        split = {"03": "valid", "04": "test", "05": "test"}.get(name[:2], "train")
        yield name, "guitarset", split, truth, lambda name=name: load_audio(os.path.join(root, "guitarset", "audio", name + "_mic.wav"))


def winterreise(root: str):
    base = os.path.join(root, "swd")
    for path in sorted(glob.glob(os.path.join(base, "02_Annotations", "ann_audio_chord", "*.csv"))):
        name = os.path.basename(path)[:-4]
        wav = os.path.join(base, "01_RawData", "audio_wav", name + ".wav")
        if not os.path.exists(wav):
            continue  # audio is only published for two of the performances
        rows = list(csv.DictReader(open(path), delimiter=";"))
        truth = [(float(r["start"]), float(r["end"]), label(r["majmin"])) for r in rows]
        yield name, "winterreise", "test" if name.endswith("SC06") else "train", truth, lambda wav=wav: load_audio(wav)


def aam(count: int, audio_dir: str | None = None):
    from tools.remote_zip import RemoteFile

    if count <= 0:
        return
    mixes = zipfile.ZipFile(RemoteFile(AAM_URL.format(part="0001-1000")))
    notes = zipfile.ZipFile(RemoteFile(AAM_ANN.format(part="0001-1000")))
    names = set(notes.namelist())
    for k in range(1, count + 1):
        song = f"{k:04d}"
        info = next((n for n in names if n.endswith(f"{song}_beatinfo.arff")), None)
        if info is None:
            continue
        beats = []
        for line in notes.read(info).decode().splitlines():
            if line and line[0].isdigit():
                time, _, _, chord = line.split(",", 3)
                beats.append((float(time), label(chord)))
        truth = [(t, beats[i + 1][0] if i + 1 < len(beats) else t + (t - beats[i - 1][0] if i else 0.5), c) for i, (t, c) in enumerate(beats)]
        split = "train" if k <= 0.75 * count else ("valid" if k <= 0.875 * count else "test")

        def load(song=song):
            kept = os.path.join(audio_dir, f"{song}.npy") if audio_dir else None
            if kept and os.path.exists(kept):
                return np.load(kept).astype(np.float32) / 32767
            with tempfile.NamedTemporaryFile(suffix=".flac") as fh:
                fh.write(mixes.read(f"{song}_mix.flac"))
                fh.flush()
                x = load_audio(fh.name)
            if kept:
                np.save(kept, (np.clip(x, -1, 1) * 32767).astype(np.int16))  # downloading is the slow part
            return x

        yield "aam_" + song, "aam", split, truth, load


def collect(root: str, aam_songs: int, have: dict, cache: str | None = None, audio_dir: str | None = None) -> list[dict]:
    items = []
    for source in (guitarset(root), winterreise(root), aam(aam_songs, audio_dir)):
        for name, origin, split, truth, load in source:
            if name in have:
                have[name]["split"] = split  # the split depends on how many songs are used this time
                items.append(have[name])
                continue
            item = {"name": name, "source": origin, "split": split, "truth": truth, **analyse(load())}
            items.append(item)
            have[name] = item
            print(origin, split, name, flush=True)
            if cache and len(items) % 25 == 0:
                pickle.dump(list(have.values()), open(cache, "wb"))  # progress survives an interruption
    return items


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
    X, y, w = [], [], []
    # Each source counts the same in total, so the many beats of the generated
    # songs do not drown out the real recordings.
    beats = {}
    for item in items:
        beats[item.get("source")] = beats.get(item.get("source"), 0) + len(item["starts"])
    for item in items:
        share = 1.0 / (len(beats) * beats[item.get("source")])
        parts, presence = chords.beat_features(item["treble"], item["bass"], item["bar_position"])
        labels = beat_labels(item)
        for root in range(12):
            rotated = np.hstack([np.roll(p, -root, axis=1) for p in parts] + [presence[:, None]])
            for i, value in enumerate(labels):
                if value is None:
                    continue
                y.append(0 if value == "N" or value[0] != root else (1 if value[1] == "maj" else 2))
                X.append(rotated[i])
                w.append(share)
    w = np.array(w)
    return np.array(X), np.array(y), w * len(w) / w.sum()


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


def fit(items, change: float, hidden: int = 0) -> dict:
    """A logistic model, or with `hidden` > 0 a small network with one hidden layer of that size."""
    X, y, w = training_set(items)
    if hidden:
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1e-3, max_iter=60, early_stopping=True, random_state=0).fit(X, y, sample_weight=w)
        layers = [(c, i) for c, i in zip(clf.coefs_, clf.intercepts_)]
    else:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=3000, C=1.0).fit(X, y, sample_weight=w)
        layers = [(clf.coef_.T, clf.intercept_)]
    return chords._parse_model({"layers": [[a.tolist(), b.tolist()] for a, b in layers], "change": change})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--cache", help="keep the per-beat profiles in this file between runs")
    parser.add_argument("--aam", type=int, default=400, help="AAM songs to stream (0 = none)")
    parser.add_argument("--aam-audio", help="keep downloaded AAM songs in this folder (about 8 MB each) for the next run")
    args = parser.parse_args()
    have = {}
    if args.cache and os.path.exists(args.cache):
        have = {item["name"]: item for item in pickle.load(open(args.cache, "rb")) if "source" in item}
    if args.aam_audio:
        os.makedirs(args.aam_audio, exist_ok=True)
    items = collect(args.root, args.aam, have, args.cache, args.aam_audio)
    if args.cache:
        pickle.dump(items, open(args.cache, "wb"))
    train = [i for i in items if i["split"] == "train"]
    valid = [i for i in items if i["split"] == "valid"]
    tests = {source: [i for i in items if i["split"] == "test" and i["source"] == source] for source in ("guitarset", "winterreise", "aam")}

    def report(title, model):
        for source, test in tests.items():
            if test:
                print(f"  {title:22} {source:12} right %.1f%% | changes found %.1f%% | changes real %.1f%%" % tuple(100 * v for v in evaluate(test, model)))

    report("templates only", None)
    model = fit(train, 1.0)

    def balance(c: float) -> float:
        # Right chords and clean change times count equally: too cheap a change
        # finds a few more chords but invents changes that are not there.
        # Averaged per source, so each kind of music has its say.
        scores = []
        for source in {i["source"] for i in valid}:
            right, found, real = evaluate([i for i in valid if i["source"] == source], {**model, "change": c})
            scores.append((right + 2 * found * real / max(found + real, 1e-9)) / 2)
        return float(np.mean(scores))

    best = max((balance(c), c) for c in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0))
    final = fit(train + valid, best[1])
    print(f"change cost {best[1]} (chosen on the tuning recordings); trained on {len(train) + len(valid)} recordings")
    report("learned model", final)
    with open(chords._MODEL_PATH, "w") as fh:
        json.dump({"layers": [[np.round(w, 5).tolist(), np.round(b, 5).tolist()] for w, b in final["layers"]], "change": final["change"]}, fh)
    print(f"wrote {chords._MODEL_PATH}")


if __name__ == "__main__":
    main()
