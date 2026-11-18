"""Export people's reviewed versions as training and test material.

    python -m tools.export_versions DATA_DIR [--min-score 0.5] [-o versions.jsonl]

Reads DATA_DIR/versions (written by the website and API) and prints one JSON
line per shared version that listeners rated accurate enough, with its chords
on the recording's clock:

    {"song": <audio hash>, "title", "artist", "author", "how", "votes_up", "votes_down",
     "score", "beats_per_bar", "chords": [[start seconds, end seconds, "Am7"], ...]}

The song is identified by the hash of its audio, so the same audio can be
analysed again to learn from the chords people confirmed. Votes on the
automatic version (auto.json) are included as {"song", "automatic": true, ...}:
a song whose automatic chords were voted down is a good one to check by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from chordlistener.server.versions import rating, tally


def timed_chords(doc: dict) -> list[list]:
    beats = doc["beats"]
    marks = doc["chords"]
    out = []
    for i, (beat, symbol) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(beats) - 1
        out.append([beats[beat], beats[end], symbol])
    return out


def export(data_dir: str, min_score: float):
    root = os.path.join(data_dir, "versions")
    if not os.path.isdir(root):
        return
    for song in sorted(os.listdir(root)):
        folder = os.path.join(root, song)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            record = json.load(open(os.path.join(folder, name), encoding="utf-8"))
            up, down = tally(record.get("votes", {}))
            score = rating(up, down)
            if name == "auto.json":
                yield {"song": song, "automatic": True, "votes_up": up, "votes_down": down, "score": score}
                continue
            if score < min_score:
                continue
            doc = record["document"]
            yield {
                "song": song, "title": record.get("title", ""), "artist": record.get("artist", ""),
                "author": record.get("author", ""), "how": record.get("how", ""),
                "votes_up": up, "votes_down": down, "score": score,
                "beats_per_bar": doc["beats_per_bar"], "chords": timed_chords(doc),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--min-score", type=float, default=0.5, help="lowest review score to include (0-1; 0.5 needs about 4 thumbs up and none down)")
    parser.add_argument("-o", "--out", help="write here instead of the screen")
    args = parser.parse_args()
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    count = 0
    for row in export(args.data_dir, args.min_score):
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
    print(f"{count} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
