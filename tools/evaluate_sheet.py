"""How much of what was played ends up on the page, and how much of the page was played.

    python -m tools.evaluate_sheet [--real DATA_DIR] [--songs N]

Scores the *written* score (after note filtering, quantising and arranging)
against the true notes: generated songs always, and real recordings when a
data directory prepared for tools/notefilter is given.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from chordlistener import AnalysisOptions, analyze, analyze_samples
from chordlistener.audio import load_audio
from chordlistener.document import document_from_analysis
from tools import songgen
from tools.notefilter import songs
from tools.notefilter.data import distinct, match

OPTIONS = AnalysisOptions()
TOLERANCE = 0.1  # seconds; written notes sit on a grid, so allow a little more than for raw detection


def written(doc: dict, staves: list[list]) -> list[tuple[float, int]]:
    beats = np.array(doc["beats"])
    out = []
    for events in staves:
        for start, _, pitches in events:
            when = float(np.interp(start / 4, np.arange(len(beats)), beats))
            out += [(when, p) for p in pitches]
    return out


def compare(page: list[tuple[float, int]], truth: list[tuple[float, int]]) -> tuple[int, int, int]:
    truth = distinct(truth)
    return int(match(page, truth, TOLERANCE).sum()), len(page), len(truth)


def summarise(label: str, counts: list[tuple[int, int, int]]) -> str:
    hit, page, truth = (sum(c[i] for c in counts) for i in range(3))
    precision, recall = hit / max(page, 1), hit / max(truth, 1)
    if "right only" in label:
        return f"{label:<34} written notes that are right {precision:6.1%}"
    f = 2 * precision * recall / max(precision + recall, 1e-9)
    return f"{label:<34} written notes that are right {precision:6.1%} | played notes that are written {recall:6.1%} | F {f:.3f}"


def synthetic(count: int, unseen: bool, first: int) -> dict[str, list]:
    rows: dict[str, list] = {"piano (both hands)": [], "right hand (right only)": [], "left hand (right only)": [], "melody line": []}
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "song.wav")
        for seed in range(first, first + count):
            song = songs.make(seed, unseen)
            songgen.render_fluidsynth(song, wav)
            doc = document_from_analysis(analyze(wav, OPTIONS))
            truth = [(s, p) for s, _, p in song.pitched_notes]
            right, left = doc["arrangements"]["piano"]
            rows["piano (both hands)"].append(compare(written(doc, [right, left]), truth))
            # Hands may cross middle C, so a hand's notes are checked against everything played;
            # only "right" is meaningful per hand, "written" is for both hands together.
            rows["right hand (right only)"].append(compare(written(doc, [right]), truth))
            rows["left hand (right only)"].append(compare(written(doc, [left]), truth))
            lead = [(s, p) for t in song.tracks if t.name == "Lead" for s, _, p, _ in t.notes]
            if lead:
                rows["melody line"].append(compare(written(doc, doc["arrangements"]["lead"]), lead))
    return rows


def real(root: str, limit: int) -> dict[str, list]:
    from tools.notefilter import real as recordings
    from tools.notefilter.train import GUITAR_TEST_PLAYERS

    rows: dict[str, list] = {"real piano": [], "real guitar": []}
    for dataset, group, _, audio, notes in recordings.smd(root):
        samples = load_audio(audio, 150.0)
        doc = document_from_analysis(analyze_samples(samples, OPTIONS))
        rows["real piano"].append(compare(written(doc, doc["arrangements"]["piano"]), [n for n in notes if n[0] < 149.5]))
    seen = 0
    for dataset, group, _, audio, notes in recordings.guitarset(root):
        if f"guitarset:{group}" not in GUITAR_TEST_PLAYERS:
            continue
        seen += 1
        if seen > limit:
            break
        doc = document_from_analysis(analyze_samples(load_audio(audio), OPTIONS))
        rows["real guitar"].append(compare(written(doc, doc["arrangements"]["piano"]), notes))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", help="directory with the real datasets (see tools/notefilter/real.py)")
    parser.add_argument("--songs", type=int, default=30)
    parser.add_argument("--accuracy", choices=["standard", "piano", "guitar"], default="standard", help="piano / guitar = high-accuracy mode with that instrument's model")
    parser.add_argument("--real-only", action="store_true", help="skip the generated songs")
    args = parser.parse_args()
    OPTIONS.accuracy = args.accuracy
    for label, unseen, first in () if args.real_only else (("Generated songs", False, 5000), ("Generated songs, unseen instruments", True, 7000)):
        print(label)
        for name, counts in synthetic(args.songs, unseen, first).items():
            print("  " + summarise(name, counts))
    if args.real:
        print("Real recordings (as a piano score)")
        for name, counts in real(args.real, 40).items():
            print("  " + summarise(name, counts))


if __name__ == "__main__":
    main()
