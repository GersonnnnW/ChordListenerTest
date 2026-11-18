"""Export people's reviewed versions as training and test material.

    python -m tools.export_versions DATA_DIR [--min-stars 4] [--min-ratings 3] [-o versions.jsonl]

Reads DATA_DIR/versions (written by the website and API) and prints one JSON
line per shared version that listeners rated accurate enough, with its chords
on the clock of the recording it was made on:

    {"song", "recording": <audio hash>, "title", "artist", "author", "how", "ratings",
     "stars", "beats_per_bar", "chords": [[start seconds, end seconds, "Am7"], ...]}

The recording is identified by the hash of its audio, so that audio can be
analysed again to learn from the chords people confirmed. Ratings of the
automatic versions are included as {"song", "recording", "automatic": true, ...}:
a recording whose automatic chords were rated low is a good one to check by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from chordlistener.server.versions import stars_summary


def timed_chords(doc: dict) -> list[list]:
    beats = doc["beats"]
    marks = doc["chords"]
    out = []
    for i, (beat, symbol) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(beats) - 1
        out.append([beats[beat], beats[end], symbol])
    return out


def export(data_dir: str, min_stars: float = 4.0, min_ratings: int = 3):
    root = os.path.join(data_dir, "versions")
    if not os.path.isdir(root):
        return
    for song in sorted(os.listdir(root)):
        folder = os.path.join(root, song)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            record = json.load(open(os.path.join(folder, name), encoding="utf-8"))
            rated = stars_summary(record.get("ratings", {}))
            if name.startswith("auto-"):
                yield {"song": song, "recording": record.get("recording", ""), "automatic": True, "ratings": rated["ratings"], "stars": rated["stars"]}
                continue
            if rated["ratings"] < min_ratings or rated["stars"] < min_stars:
                continue
            doc = record["document"]
            yield {
                "song": song, "recording": record.get("recording", ""), "title": record.get("title", ""), "artist": record.get("artist", ""),
                "author": record.get("author", ""), "how": record.get("how", ""), "ratings": rated["ratings"], "stars": rated["stars"],
                "beats_per_bar": doc["beats_per_bar"], "chords": timed_chords(doc),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--min-stars", type=float, default=4.0, help="lowest average rating to include")
    parser.add_argument("--min-ratings", type=int, default=3, help="fewest ratings to include")
    parser.add_argument("-o", "--out", help="write here instead of the screen")
    args = parser.parse_args()
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    count = 0
    for row in export(args.data_dir, args.min_stars, args.min_ratings):
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
    print(f"{count} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
