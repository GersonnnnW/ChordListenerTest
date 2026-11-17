"""Render random songs and save labelled note candidates.  python -m tools.notefilter.build_synth OUT N FIRST_SEED [--unseen]"""

import os
import sys
import tempfile

import numpy as np

from chordlistener.audio import load_audio
from tools import songgen
from tools.notefilter import data, songs


def main() -> None:
    out, count, first = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    unseen = "--unseen" in sys.argv
    for arg in sys.argv:
        if arg.startswith("--cand="):  # try other candidate thresholds: --cand=onset,frame,seconds
            from chordlistener import notefilter

            notefilter.CANDIDATE_ONSET, notefilter.CANDIDATE_FRAME, notefilter.CANDIDATE_SECONDS = map(float, arg[7:].split(","))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "song.wav")
        for seed in range(first, first + count):
            song = songs.make(seed, unseen)
            songgen.render_fluidsynth(song, wav)
            item = data.examples(load_audio(wav), [(start, pitch) for start, _, pitch in song.pitched_notes])
            item["song"] = seed
            rows.append(item)
            if len(rows) % 20 == 0:
                print(len(rows), "songs", flush=True)
    np.savez_compressed(
        out,
        x=np.concatenate([r["x"] for r in rows]),
        y=np.concatenate([r["y"] for r in rows]),
        song=np.concatenate([np.full(len(r["y"]), r["song"]) for r in rows]),
        songs=np.array([r["song"] for r in rows]),
        n_reference=np.array([r["n_reference"] for r in rows]),
        baseline_found=np.array([r["baseline_found"] for r in rows]),
        baseline_written=np.array([r["baseline_written"] for r in rows]),
    )


if __name__ == "__main__":
    main()
