"""Notation building blocks: bars made of notes, chords and rests.

Everything here is on a grid of sixteenth notes. A `Score` is a list of
measures; each measure holds, per staff, a gap-free run of `Element`s whose
durations are all directly notatable. `document.py` decides what goes in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .theory import Key

DIVISIONS = 4  # grid steps per quarter note

# (length in sixteenths, note type, dots)
_VALUES = [(16, "whole", 0), (12, "half", 1), (8, "half", 0), (6, "quarter", 1), (4, "quarter", 0), (3, "eighth", 1), (2, "eighth", 0), (1, "16th", 0)]
NOTE_TYPES = {length: (kind, dots) for length, kind, dots in _VALUES}

# One staff's music: (start, length, pitches), in sixteenths, never overlapping.
Event = tuple[int, int, tuple[int, ...]]


@dataclass
class Element:
    start: int  # sixteenths from the start of the measure
    length: int
    pitches: tuple[int, ...] = ()  # empty = rest
    tie_in: bool = False
    tie_out: bool = False
    slash: bool = False  # rhythm slash: "play the chord", no specific notes
    lyric: str = ""
    frets: tuple[tuple[int, int], ...] = ()  # tablature: (string, fret) per pitch, string 1 = highest
    joins: str = ""  # guitar technique leading to the next note: h p / \\
    joined: str = ""  # guitar technique this note was reached by
    bend: int = 0  # semitones this note is bent up


@dataclass
class Harmony:
    offset: int  # sixteenths from the start of the measure
    text: str  # what to print if the chord cannot be expressed structurally
    root: tuple[str, int] | None = None  # (step, alter)
    quality: str | None = None
    bass: tuple[str, int] | None = None


@dataclass
class Measure:
    staves: list[list[Element]]
    harmonies: list[Harmony] = field(default_factory=list)
    words: str = ""  # lyric text for a bar with no note to hang it on


@dataclass
class Score:
    measures: list[Measure]
    key: Key
    beats_per_bar: int
    bpm: float | None  # None = do not print a tempo mark
    title: str
    artist: str
    subtitle: str
    n_staves: int
    clefs: list[str]  # G, F, or TAB
    part_name: str = ""
    tab_tuning: tuple[int, ...] = ()  # open-string pitches, lowest first, for a TAB staff
    written_octave_up: bool = False  # guitar: written an octave above where it sounds


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


def melody(notes: list[tuple[int, int, int, float]]) -> list[tuple[int, int, int, float]]:
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


def voice(notes: list[tuple[int, int, int, float]], total: int, limit: int, keep: str) -> list[Event]:
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


def lay_out(events: list[Event], n_bars: int, bar: int, cuts: list[set[int]]) -> list[list[Element]]:
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


def slashes(n_bars: int, beats_per_bar: int) -> list[list[Element]]:
    """One rhythm slash per beat: the staff of a chord sheet."""
    return [[Element(beat * DIVISIONS, DIVISIONS, slash=True) for beat in range(beats_per_bar)] for _ in range(n_bars)]


def underlay(elements: list[Element], text: str) -> bool:
    """Spread a bar's lyric over the notes that start in it. False if there are none."""
    words = text.split()
    targets = [e for e in elements if (e.pitches or e.slash) and not e.tie_in]
    if not words:
        return True
    if not targets:
        return False
    # Fill from the first note; whatever does not fit shares the last one.
    for target, word in zip(targets, words):
        target.lyric = word
    if len(words) > len(targets):
        targets[-1].lyric = " ".join(words[len(targets) - 1 :])
    return True
