"""Measure accuracy and speed on generated songs with known ground truth.

    python -m tools.evaluate            # FluidSynth renders if available
    python -m tools.evaluate --numpy    # built-in synth only

These are synthetic arrangements, so treat the numbers as a regression check,
not as a claim about accuracy on commercial recordings.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from chordlistener import AnalysisOptions, analyze
from chordlistener.theory import QUALITIES

from . import songgen

SONGS = [
    dict(name="pop-C", progression=[("C", 4), ("G", 4), ("Am", 4), ("F", 4)], bpm=100, key="C", style="piano", repeats=4),
    dict(name="pop-G-guitar", progression=[("G", 4), ("D", 4), ("Em", 4), ("C", 4)], bpm=124, key="G", style="guitar", repeats=4),
    dict(name="minor-Am", progression=[("Am", 4), ("F", 4), ("C", 4), ("G", 4), ("Am", 4), ("Dm", 4), ("E", 8)], bpm=92, key="Am", style="epiano", repeats=2),
    dict(name="sevenths-F", progression=[("Fmaj7", 4), ("Dm7", 4), ("Gm7", 4), ("C7", 4), ("Am7", 4), ("Dm7", 4), ("Gm7", 2), ("C7", 2), ("Fmaj7", 4)], bpm=84, key="F", style="epiano", repeats=2),
    dict(name="two-per-bar-D", progression=[("D", 2), ("A", 2), ("Bm", 2), ("G", 2), ("D", 2), ("G", 2), ("A", 4)], bpm=116, key="D", style="piano", repeats=4),
    dict(name="waltz-Bb", progression=[("Bb", 3), ("Gm", 3), ("Eb", 3), ("F", 3), ("Bb", 3), ("Dm", 3), ("Eb", 3), ("F7", 3)], bpm=132, key="Bb", style="piano", repeats=3, beats_per_bar=3),
    dict(name="flat-keys-Eb", progression=[("Eb", 4), ("Cm", 4), ("Ab", 4), ("Bb", 4), ("Fm", 4), ("Bb7", 4), ("Eb", 8)], bpm=76, key="Eb", style="organ", repeats=2),
    dict(name="sharp-keys-E", progression=[("E", 4), ("B", 4), ("C#m", 4), ("A", 4), ("F#m", 4), ("B", 4), ("E", 4), ("Esus4", 2), ("E", 2)], bpm=140, key="E", style="guitar", repeats=3),
    dict(name="solo-piano", progression=[("C", 4), ("Em", 4), ("F", 4), ("G", 4), ("Am", 4), ("Em", 4), ("Dm", 4), ("G", 4)], bpm=96, key="C", style="piano", repeats=2, drums=False, melody=False),
    dict(name="minor-Em-fast", progression=[("Em", 4), ("C", 4), ("G", 4), ("D", 4), ("Em", 4), ("Am", 4), ("B7", 8)], bpm=152, key="Em", style="guitar", repeats=3),
]


def chord_accuracy(song: songgen.Song, result) -> tuple[float, float]:
    """Time-weighted overlap with the reference: (root correct, root+quality correct)."""
    step = 0.02
    grid = np.arange(0, song.chords[-1][1], step)

    def sample(spans):
        root = np.full(len(grid), -1)
        qual = np.full(len(grid), "", dtype=object)
        for start, end, r, q in spans:
            sel = (grid >= start) & (grid < end)
            root[sel] = r
            qual[sel] = q
        return root, qual

    ref_root, ref_qual = sample(song.chords)
    est_root, est_qual = sample([(c.start, c.end, c.chord.root, c.chord.quality) for c in result.chords if c.chord])
    scored = ref_root >= 0
    root_ok = (ref_root == est_root) & scored
    return float(root_ok.sum() / scored.sum()), float((root_ok & (ref_qual == est_qual)).sum() / scored.sum())


def note_f1(song: songgen.Song, result, tolerance: float = 0.05) -> float:
    reference = song.pitched_notes
    by_pitch: dict[int, list[float]] = {}
    for start, _, pitch in reference:
        by_pitch.setdefault(pitch, []).append(start)
    hits = 0
    for note in result.notes:
        starts = by_pitch.get(note.pitch, [])
        for i, start in enumerate(starts):
            if abs(start - note.start) <= tolerance:
                starts.pop(i)
                hits += 1
                break
    if not hits:
        return 0.0
    precision, recall = hits / len(result.notes), hits / len(reference)
    return 2 * precision * recall / (precision + recall)


def beat_f1(song: songgen.Song, result, tolerance: float = 0.07) -> float:
    est = result.grid.times
    est = est[(est >= song.beats[0] - tolerance) & (est <= song.beats[-1] + tolerance)]
    hits = sum(1 for b in song.beats if len(est) and np.abs(est - b).min() <= tolerance)
    if not hits:
        return 0.0
    precision, recall = hits / len(est), hits / len(song.beats)
    return 2 * precision * recall / (precision + recall)


def evaluate(use_fluidsynth: bool, verbose: bool = False) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, spec in enumerate(SONGS):
            spec = dict(spec)
            name = spec.pop("name")
            song = songgen.make_song(seed=i, **spec)
            path = os.path.join(tmp, f"{name}.wav")
            if use_fluidsynth:
                songgen.render_fluidsynth(song, path)
            else:
                songgen.write_wav(path, songgen.render_numpy(song, seed=i))
            result = analyze(path, AnalysisOptions(vocabulary="standard"))
            root, exact = chord_accuracy(song, result)
            ratio = result.bpm / song.bpm
            rows.append(
                dict(
                    name=name,
                    root=root,
                    exact=exact,
                    notes=note_f1(song, result),
                    beats=beat_f1(song, result),
                    tempo_ok=abs(ratio - 1) < 0.04,
                    tempo=f"{result.bpm:.0f}/{song.bpm:.0f}",
                    meter_ok=result.grid.per_bar == song.beats_per_bar,
                    key_ok=(result.key.tonic, result.key.minor) == (song.key_tonic, song.key_minor),
                    key=result.key.name,
                    speed=result.duration / (result.processing_ms / 1000),
                )
            )
            if verbose:
                print(name, " ".join(f"{c.chord.symbol(result.key) if c.chord else 'N'}" for c in result.chords))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numpy", action="store_true", help="use the built-in synth even if FluidSynth is installed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    use_fluid = songgen.fluidsynth_available() and not args.numpy
    print(f"Renderer: {'FluidSynth' if use_fluid else 'built-in numpy synth'}\n")
    rows = evaluate(use_fluid, args.verbose)
    print(f"{'song':<18}{'root':>6}{'chord':>7}{'notes':>7}{'beats':>7}  {'tempo':<9}{'meter':<6}{'key':<14}{'speed':>7}")
    for r in rows:
        print(
            f"{r['name']:<18}{r['root']:>6.0%}{r['exact']:>7.0%}{r['notes']:>7.2f}{r['beats']:>7.2f}  "
            f"{r['tempo'] + ('' if r['tempo_ok'] else '!'):<9}{'ok' if r['meter_ok'] else 'WRONG':<6}"
            f"{r['key'] + ('' if r['key_ok'] else ' !'):<14}{r['speed']:>6.0f}x"
        )
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    print(f"\n{'mean':<18}{mean('root'):>6.0%}{mean('exact'):>7.0%}{mean('notes'):>7.2f}{mean('beats'):>7.2f}  speed {mean('speed'):.0f}x real time")


if __name__ == "__main__":
    main()
