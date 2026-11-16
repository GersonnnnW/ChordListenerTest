"""Turn a performance (notes in seconds) into notation (notes in bars).

Everything here is on a grid of sixteenth notes. A `Score` is a list of
measures; each measure holds, per staff, a gap-free run of `Element`s (notes,
chords of notes, or rests) whose durations are all directly notatable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pipeline import Analysis
from .theory import Chord, Key

DIVISIONS = 4  # grid steps per quarter note

# (length in sixteenths, note type, dots)
_VALUES = [(16, "whole", 0), (12, "half", 1), (8, "half", 0), (6, "quarter", 1), (4, "quarter", 0), (3, "eighth", 1), (2, "eighth", 0), (1, "16th", 0)]
NOTE_TYPES = {length: (kind, dots) for length, kind, dots in _VALUES}


@dataclass
class ScoreOptions:
    arrangement: str = "piano"  # piano (two staves) | lead (melody line)
    transpose: int = 0
    title: str = ""
    artist: str = ""
    max_notes_per_chord: int = 4
    include_chords: bool = True


@dataclass
class Element:
    start: int  # sixteenths from the start of the measure
    length: int
    pitches: tuple[int, ...] = ()  # empty = rest
    tie_in: bool = False
    tie_out: bool = False


@dataclass
class Measure:
    staves: list[list[Element]]
    harmonies: list[tuple[int, Chord]] = field(default_factory=list)  # (offset in sixteenths, chord)


@dataclass
class Score:
    measures: list[Measure]
    key: Key
    beats_per_bar: int
    bpm: float
    title: str
    artist: str
    n_staves: int
    clefs: list[str]


def _fits(length: int, pos: int, rest: bool, bar: int) -> bool:
    """Whether a value of `length` may start at `pos` without obscuring the beat."""
    if length == 16:
        return pos == 0
    if length == 12:
        return pos % 4 == 0 and (not rest or bar == 12)
    if length == 8:
        return pos % (8 if rest and bar == 16 else 4) == 0
    if length == 6:
        return pos % 4 == 0 and not rest
    if length == 4:
        return pos % (4 if rest else 2) == 0
    if length == 3:
        return pos % 4 == 0 and not rest
    if length == 2:
        return pos % 2 == 0
    return True


def split_duration(pos: int, length: int, bar: int, rest: bool = False) -> list[int]:
    """Break a span inside one bar into notatable values (to be tied together)."""
    out = []
    while length > 0:
        for value, _, _ in _VALUES:
            if value <= length and pos + value <= bar and _fits(value, pos, rest, bar):
                out.append(value)
                pos += value
                length -= value
                break
    return out


def _melody(notes: list[tuple[int, int, int, float]]) -> list[tuple[int, int, int, float]]:
    """Skyline melody: the highest reasonably strong note at each onset."""
    by_onset: dict[int, list[tuple[int, int, int, float]]] = {}
    for note in notes:
        if note[2] >= 55:
            by_onset.setdefault(note[0], []).append(note)
    line = []
    for onset in sorted(by_onset):
        group = by_onset[onset]
        loudest = max(n[3] for n in group)
        line.append(max((n for n in group if n[3] >= 0.6 * loudest), key=lambda n: n[2]))
    if not line:
        return []
    # Discard isolated leaps of more than an octave away from the local line.
    pitches = np.array([n[2] for n in line], dtype=float)
    kept = []
    for i, note in enumerate(line):
        window = pitches[max(0, i - 4) : i + 5]
        if abs(note[2] - np.median(window)) <= 12:
            kept.append(note)
    return kept


def _voice(notes: list[tuple[int, int, int, float]], total: int, limit: int, keep: str) -> list[tuple[int, int, tuple[int, ...]]]:
    """Reduce overlapping notes to one rhythmic line of (start, length, pitches)."""
    groups: dict[int, dict[int, tuple[int, float]]] = {}
    for start, length, pitch, velocity in notes:
        slot = groups.setdefault(start, {})
        if pitch not in slot or slot[pitch][0] < length:
            slot[pitch] = (length, velocity)
    onsets = sorted(groups)
    events = []
    for i, onset in enumerate(onsets):
        slot = groups[onset]
        pitches = sorted(slot)
        if len(pitches) > limit:
            # Always keep the outer voice, then the strongest of the rest.
            anchor = pitches[-1] if keep == "top" else pitches[0]
            rest = sorted((p for p in pitches if p != anchor), key=lambda p: -slot[p][1])[: limit - 1]
            pitches = sorted([anchor] + rest)
        nxt = onsets[i + 1] if i + 1 < len(onsets) else total
        length = min(max(v[0] for v in slot.values()), nxt - onset)
        gap = nxt - (onset + length)
        if 0 < gap <= (2 if length >= 2 else 1) and i + 1 < len(onsets):
            length += gap  # close tiny gaps instead of littering short rests
        if length > 0:
            events.append((onset, length, tuple(pitches)))
    return events


def _measures_for(events: list[tuple[int, int, tuple[int, ...]]], n_bars: int, bar: int, cuts: list[set[int]]) -> list[list[Element]]:
    """Lay events out bar by bar, filling gaps with rests and tying across bar lines."""
    out: list[list[Element]] = [[] for _ in range(n_bars)]

    def emit(start: int, length: int, pitches: tuple[int, ...]) -> None:
        end = start + length
        pos = start
        while pos < end:
            index = pos // bar
            if index >= n_bars:
                break
            inside = pos - index * bar
            span = min(end - pos, bar - inside)
            if not pitches:
                # Rests may be cut freely, so cut them where a chord symbol lands.
                for cut in sorted(cuts[index]):
                    if inside < cut < inside + span:
                        span = cut - inside
                        break
                if inside == 0 and span == bar:
                    out[index].append(Element(0, bar))
                    pos += span
                    continue
            at = inside
            for value in split_duration(inside, span, bar, rest=not pitches):
                out[index].append(Element(at, value, pitches))
                at += value
            if pitches:
                chunk = [e for e in out[index] if e.start >= inside and e.pitches]
                for k, element in enumerate(chunk):
                    element.tie_in = k > 0 or pos > start
                    element.tie_out = k < len(chunk) - 1 or pos + span < end
            pos += span

    cursor = 0
    for start, length, pitches in events:
        if start > cursor:
            emit(cursor, start - cursor, ())
        emit(start, length, pitches)
        cursor = start + length
    if cursor < n_bars * bar:
        emit(cursor, n_bars * bar - cursor, ())
    return out


def build_score(analysis: Analysis, options: ScoreOptions | None = None) -> Score:
    options = options or ScoreOptions()
    grid = analysis.grid
    key = analysis.key.transposed(options.transpose)
    bar = grid.per_bar * DIVISIONS
    n_bars = max(1, (len(grid.times) - 1) // grid.per_bar)
    total = n_bars * bar

    quantised: list[tuple[int, int, int, float]] = []
    if analysis.notes:
        starts = grid.position(np.array([n.start for n in analysis.notes])) * DIVISIONS
        ends = grid.position(np.array([n.end for n in analysis.notes])) * DIVISIONS
        for note, s, e in zip(analysis.notes, starts, ends):
            pitch = note.pitch + options.transpose
            start = int(round(s))
            if not 21 <= pitch <= 108 or start >= total:
                continue
            start = max(0, start)
            length = max(1, int(round(e)) - start)
            quantised.append((start, min(length, total - start), pitch, note.velocity))

    harmonies: list[list[tuple[int, Chord]]] = [[] for _ in range(n_bars)]
    if options.include_chords:
        previous = None
        for event in analysis.chords:
            if event.chord is None:
                previous = None
                continue
            chord = event.chord.transposed(options.transpose)
            position = int(round(event.beat * DIVISIONS))
            if chord != previous and 0 <= position < total:
                harmonies[position // bar].append((position % bar, chord))
            previous = chord
    cuts = [{offset for offset, _ in h} for h in harmonies]
    no_cuts: list[set[int]] = [set() for _ in range(n_bars)]

    limit = options.max_notes_per_chord
    if options.arrangement == "lead":
        line = _voice(_melody(quantised), total, 1, "top")
        staves = [_measures_for(line, n_bars, bar, cuts)]
        clefs = ["G"]
    else:
        upper = _voice([n for n in quantised if n[2] >= 60], total, limit, "top")
        lower = _voice([n for n in quantised if n[2] < 60], total, limit, "bottom")
        staves = [_measures_for(upper, n_bars, bar, cuts), _measures_for(lower, n_bars, bar, no_cuts)]
        clefs = ["G", "F"]

    measures = [Measure([staff[i] for staff in staves], harmonies[i]) for i in range(n_bars)]
    return Score(measures, key, grid.per_bar, analysis.bpm, options.title, options.artist, len(staves), clefs)
