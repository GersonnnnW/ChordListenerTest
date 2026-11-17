"""Real recordings with human-checked notes.

  GuitarSet  acoustic guitar, 360 excerpts            CC BY 4.0      (training + testing)
  vocadito   solo singing, 40 excerpts                CC BY 4.0      (training + testing)
  BabySlakh  full-band songs rendered with studio     CC BY 4.0      (training + testing)
             sample libraries, 20 songs
  SMD        piano on a Disklavier, aligned MIDI      CC BY-NC-SA    (testing only: non-commercial)

python -m tools.notefilter.real OUT.npz DATA_DIR
"""

from __future__ import annotations

import glob
import json
import os
import struct
import sys

import numpy as np

from chordlistener.audio import load_audio
from tools.notefilter import data


def read_midi_notes(path: str, skip_drum_channel: bool = True) -> list[tuple[float, int]]:
    """(start in seconds, pitch) of every note in a Standard MIDI File, honouring tempo changes."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"MThd"
    _, _, n_tracks, division = struct.unpack(">IHHH", raw[4:14])
    pos = 14
    tempos: list[tuple[int, int]] = []
    ons: list[tuple[int, int]] = []
    for _ in range(n_tracks):
        assert raw[pos : pos + 4] == b"MTrk"
        end = pos + 8 + struct.unpack(">I", raw[pos + 4 : pos + 8])[0]
        pos += 8
        tick, status = 0, 0
        while pos < end:
            delta = 0
            while True:
                byte = raw[pos]
                pos += 1
                delta = (delta << 7) | (byte & 0x7F)
                if not byte & 0x80:
                    break
            tick += delta
            if raw[pos] & 0x80:
                status = raw[pos]
                pos += 1
            if status == 0xFF:
                kind = raw[pos]
                pos += 1
                length = 0
                while True:
                    byte = raw[pos]
                    pos += 1
                    length = (length << 7) | (byte & 0x7F)
                    if not byte & 0x80:
                        break
                if kind == 0x51:
                    tempos.append((tick, int.from_bytes(raw[pos : pos + 3], "big")))
                pos += length
            elif status in (0xF0, 0xF7):
                length = 0
                while True:
                    byte = raw[pos]
                    pos += 1
                    length = (length << 7) | (byte & 0x7F)
                    if not byte & 0x80:
                        break
                pos += length
            else:
                kind = status & 0xF0
                if kind in (0xC0, 0xD0):
                    pos += 1
                else:
                    a, b = raw[pos], raw[pos + 1]
                    pos += 2
                    if kind == 0x90 and b > 0 and not (skip_drum_channel and (status & 0x0F) == 9):
                        ons.append((tick, a))
        pos = end
    tempos = sorted(tempos) or [(0, 500000)]
    if tempos[0][0] != 0:
        tempos.insert(0, (0, 500000))

    def seconds(tick: int) -> float:
        total, last_tick, micro = 0.0, 0, tempos[0][1]
        for at, value in tempos:
            if at >= tick:
                break
            total += (at - last_tick) * micro / 1e6 / division
            last_tick, micro = at, value
        return total + (tick - last_tick) * micro / 1e6 / division

    return sorted((seconds(tick), pitch) for tick, pitch in ons)


def guitarset(root: str):
    for path in sorted(glob.glob(os.path.join(root, "guitarset", "ann", "*.jams"))):
        name = os.path.basename(path)[:-5]
        wav = os.path.join(root, "guitarset", "audio", name + "_mic.wav")
        if not os.path.exists(wav):
            continue
        notes = []
        for annotation in json.load(open(path))["annotations"]:
            if annotation["namespace"] == "note_midi":
                notes += [(d["time"], int(round(d["value"]))) for d in annotation["data"]]
        # The first two digits are the player: keep players apart when splitting.
        yield "guitarset", name[:2], name, wav, sorted(notes)


def vocadito(root: str):
    for wav in sorted(glob.glob(os.path.join(root, "vocadito", "Audio", "*.wav"))):
        name = os.path.basename(wav)[:-4]
        rows = np.loadtxt(os.path.join(root, "vocadito", "Annotations", "Notes", name + "_notesA1.csv"), delimiter=",", ndmin=2)
        notes = [(float(start), int(round(69 + 12 * np.log2(hz / 440.0)))) for start, hz, _ in rows if hz > 0]
        yield "vocadito", name, name, wav, notes


def slakh(root: str):
    import yaml

    for folder in sorted(glob.glob(os.path.join(root, "slakh", "babyslakh_16k", "Track*"))):
        meta = yaml.safe_load(open(os.path.join(folder, "metadata.yaml")))
        notes = []
        for stem, info in meta["stems"].items():
            midi = os.path.join(folder, "MIDI", stem + ".mid")
            # The metadata's "audio_rendered" flag is unreliable; what is in the mix is what has a stem file.
            if info.get("is_drum") or not os.path.exists(os.path.join(folder, "stems", stem + ".wav")) or not os.path.exists(midi):
                continue
            notes += read_midi_notes(midi, skip_drum_channel=False)
        name = os.path.basename(folder)
        yield "slakh", name, name, os.path.join(folder, "mix.wav"), sorted(notes)


def smd(root: str):
    for mid in sorted(glob.glob(os.path.join(root, "smd", "files", "*.mid"))):
        # The old website offered MP3 previews; the Zenodo release (v2) has WAV files.
        audio = next((mid[:-4] + ext for ext in (".mp3", ".wav") if os.path.exists(mid[:-4] + ext)), None)
        if audio:
            name = os.path.basename(mid)[:-4]
            yield "smd", name, name, audio, read_midi_notes(mid)


def main() -> None:
    """python -m tools.notefilter.real OUT.npz DATA_DIR PART PARTS  (PART of PARTS, to run several at once)"""
    out, root = sys.argv[1], sys.argv[2]
    part, parts = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    todo = []
    for loader, limit in ((guitarset, None), (vocadito, None), (slakh, 240.0), (smd, 150.0)):
        todo += [(item, limit) for item in loader(root)]
    rows = []
    for (dataset, group, name, audio, notes), limit in todo[part::parts]:
        samples = load_audio(audio, limit)
        if limit:
            notes = [n for n in notes if n[0] < limit - 0.5]
        item = data.examples(samples, notes)
        item.update(dataset=dataset, group=f"{dataset}:{group}", name=name)
        rows.append(item)
        if len(rows) % 25 == 0:
            print(len(rows), "recordings", flush=True)
    index = np.concatenate([np.full(len(r["y"]), i) for i, r in enumerate(rows)])
    np.savez_compressed(
        out,
        x=np.concatenate([r["x"] for r in rows]),
        y=np.concatenate([r["y"] for r in rows]),
        recording=index,
        dataset=np.array([r["dataset"] for r in rows]),
        group=np.array([r["group"] for r in rows]),
        name=np.array([r["name"] for r in rows]),
        n_reference=np.array([r["n_reference"] for r in rows]),
        baseline_found=np.array([r["baseline_found"] for r in rows]),
        baseline_written=np.array([r["baseline_written"] for r in rows]),
    )


if __name__ == "__main__":
    main()
