"""How often the tab puts a note on the string the guitarist really used.

    python -m tools.evaluate_tabs DATASETS [--players 04,05]
    python -m tools.evaluate_tabs DATASETS --train     # rewrite models/tabs.json from players 00-03

GuitarSet records every note per string, so its true notes (not detected ones)
are fed to the fingering in chordlistener/tabs.py and each chosen string is
compared with the real one. Strums of more than four notes are left out, as in
the tab itself. The habits are learned from players 00-03; 04-05 are the test
players and must never be learned from.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

from chordlistener import tabs


def excerpt(path: str) -> tuple[float, list[tuple[float, int, int]]]:
    """(beats per second, [(time, pitch, string)]) for one GuitarSet annotation file."""
    data = json.load(open(path))
    notes = []
    for annotation in data["annotations"]:
        if annotation["namespace"] == "note_midi":
            string = int(annotation["annotation_metadata"]["data_source"])
            notes += [(d["time"], int(round(d["value"])), string) for d in annotation["data"]]
    bpm = float(re.search(r"-(\d+)-", os.path.basename(path)).group(1))
    return bpm / 60.0, sorted(notes)


def groups(notes, window: float = 0.03):
    out: list[list[tuple[float, int, int]]] = []
    for note in notes:
        if out and note[0] - out[-1][0][0] <= window:
            out[-1].append(note)
        else:
            out.append([note])
    return out


def sequences(root: str, players: set[str], kind: str = ""):
    """Per excerpt: [(time in beats, pitches, {pitch: true string})], strums left out."""
    for path in sorted(glob.glob(os.path.join(root, "guitarset", "ann", "*.jams"))):
        name = os.path.basename(path)
        if name[:2] not in players or kind not in name:
            continue
        rate, notes = excerpt(path)
        seq = []
        for g in groups(notes):
            seen: dict[int, int] = {}
            for _, pitch, string in g:
                seen.setdefault(pitch, string)
            if len(seen) > tabs.MAX_NOTES or not tabs.fingerings(tuple(sorted(seen)), tabs.TUNINGS["standard"]):
                continue
            seq.append((g[0][0] * rate, tuple(sorted(seen)), seen))
        yield seq


def score(root: str, players: set[str], kind: str = "") -> tuple[int, int]:
    right = total = 0
    for seq in sequences(root, players, kind):
        chosen = tabs.place([(t, p) for t, p, _ in seq], tabs.TUNINGS["standard"])
        for (_, pitches, truth), places in zip(seq, chosen):
            for pitch, (string, _) in zip(pitches, places):
                right += string == truth[pitch]
                total += 1
    return right, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--players", default="04,05")
    parser.add_argument("--train", action="store_true", help="learn the habits from players 00-03 and write models/tabs.json")
    args = parser.parse_args()
    if args.train:
        standard = tabs.TUNINGS["standard"]
        played = [[(t, [(truth[p], p - standard[truth[p]]) for p in pitches]) for t, pitches, truth in seq] for seq in sequences(args.root, {"00", "01", "02", "03"})]
        with open(tabs._MODEL_PATH, "w") as fh:
            json.dump(tabs.learn(played), fh, separators=(",", ":"))
        print(f"wrote {tabs._MODEL_PATH}")
    players = set(args.players.split(","))
    for kind in ("solo", "comp", ""):
        right, total = score(args.root, players, kind)
        print(f"{kind or 'all':5} notes on the right string: {right / max(total, 1):.1%} of {total}")


if __name__ == "__main__":
    main()
