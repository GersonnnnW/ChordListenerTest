"""Guitar tablature: which string and fret each note is played on.

The notes come from the score document, so corrections made to the notes carry
over. A note can usually be played in several places on the neck (the E above
middle C on five strings), and guitarists choose by habit: where on the neck
they like to play, and how far the hand and fingers move from one note to the
next. Both habits were counted from GuitarSet, whose recordings say which
string every note was really played on (models/tabs.json, made by
`python -m tools.evaluate_tabs DATASETS --train`). The fingering for the whole
piece is then chosen at once (dynamic programming over every playable
fingering of every note or double stop), as the most guitarist-like path.

On GuitarSet players never used for training, 73% of notes land on the string
the guitarist used (68% in solos, 75% in accompaniment); the rest are mostly
the same notes played elsewhere on the neck, equally playable.

Bars that are strummed chords (most attacks are three or more notes at once)
show the shape of each chord instead, stacked, where it starts (chordshapes.py).
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import chordshapes
from .theory import Chord, parse_chord_symbol

TUNINGS = {
    "standard": (40, 45, 50, 55, 59, 64),  # E2 A2 D3 G3 B3 E4, lowest string first
    "drop_d": (38, 45, 50, 55, 59, 64),
}
STRING_NAMES = {"standard": ("E", "A", "D", "G", "B", "e"), "drop_d": ("D", "A", "D", "G", "B", "e")}
MAX_FRET = 20
MAX_NOTES = 4  # more than this at once is a strum, not a line

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tabs.json")
_model = None
GAP_BUCKETS = (0.3, 0.6, 1.1)  # beats between notes: together, quick, steady, after a pause
MAX_SHIFT = 13  # fret moves are counted up to this far
WHERE_WEIGHT = 2.0  # how much "where on the neck" counts against "how the hand moves" (2: best on detected notes)


@dataclass
class TabNote:
    start: int  # sixteenths from the start of the piece
    length: int
    pitches: tuple[int, ...]
    places: tuple[tuple[int, int], ...]  # (string, fret) per pitch; string 0 = lowest
    technique: str = ""  # how this note was reached from the previous one: h p / \\
    bend: int = 0  # semitones this note is bent up (a bent-to note is folded into the note it bends from)
    shape: bool = False  # the chord shape to hold in a strummed bar


@dataclass
class TabBar:
    notes: list[TabNote] = field(default_factory=list)
    strummed: bool = False  # chords only: nothing written


@dataclass
class Tab:
    tuning: str
    bars: list[TabBar]
    bar_length: int  # sixteenths


# ---------------------------------------------------------------- fingering


def _options(pitch: int, tuning: tuple[int, ...]) -> list[tuple[int, int]]:
    return [(s, pitch - open_) for s, open_ in enumerate(tuning) if 0 <= pitch - open_ <= MAX_FRET]


def fingerings(pitches: tuple[int, ...], tuning: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    """Every way to play these pitches together: one string each, within a hand's reach."""
    out = []
    order = sorted(range(len(pitches)), key=lambda i: pitches[i])
    for combo in itertools.product(*(_options(p, tuning) for p in pitches)):
        strings = [combo[i][0] for i in order]
        if any(a >= b for a, b in zip(strings, strings[1:])):
            continue  # one string per note, and a higher note never on a lower string
        fretted = [f for _, f in combo if f > 0]
        if fretted and max(fretted) - min(fretted) > 4:
            continue
        out.append(tuple(combo))
    return out


def _load() -> dict:
    global _model
    if _model is None:
        with open(_MODEL_PATH) as fh:
            raw = json.load(fh)
        _model = {"where": np.array(raw["where"]), "move": np.array(raw["move"])}
    return _model


def _bucket(gap: float) -> int:
    return int(np.searchsorted(GAP_BUCKETS, gap))


def _lead(places) -> tuple[int, int]:
    """The note that leads a group for the hand: the one on the highest string."""
    return max(places)


def place(groups: list[tuple[float, tuple[int, ...]]], tuning: tuple[int, ...], joined: list[bool] | None = None) -> list[tuple[tuple[int, int], ...]]:
    """Choose a fingering for each (time in beats, pitches) group: the most guitarist-like path.

    Costs are -log probabilities counted from GuitarSet: where on the neck each
    note is played (string, fret), and how the leading note moves between groups
    (strings across, frets along, given the time between them). A group marked in
    `joined` (a hammer-on, pull-off or slide from the one before) must stay on the
    same string.
    """
    if not groups:
        return []
    model = _load()
    where, move = model["where"], model["move"]
    options = [fingerings(p, tuning) for _, p in groups]
    leads = [np.array([_lead(f) for f in o]) for o in options]
    statics = [WHERE_WEIGHT * np.array([sum(where[s, f] for s, f in places) for places in o]) for o in options]
    best = statics[0]
    back = []
    for i in range(1, len(groups)):
        table = move[_bucket(groups[i][0] - groups[i - 1][0])]
        across = leads[i][None, :, 0] - leads[i - 1][:, None, 0] + 5
        along = np.clip(leads[i][None, :, 1] - leads[i - 1][:, None, 1], -MAX_SHIFT, MAX_SHIFT) + MAX_SHIFT
        total = best[:, None] + table[across, along]
        if joined and joined[i]:
            total = total + 1e6 * (across != 5)  # a hammer-on, pull-off or slide stays on its string
        links = total.argmin(axis=0)
        best = total[links, np.arange(len(links))] + statics[i]
        back.append(links)
    k = int(best.argmin())
    chosen = [options[-1][k]]
    for i in range(len(groups) - 1, 0, -1):
        k = int(back[i - 1][k])
        chosen.append(options[i - 1][k])
    return chosen[::-1]


def learn(sequences: list[list[tuple[float, list[tuple[int, int]]]]], smoothing: float = 1.0) -> dict:
    """Count the habits from (time in beats, [(string, fret), ...]) sequences played by real guitarists."""
    where = np.full((6, MAX_FRET + 1), smoothing)
    move = np.full((len(GAP_BUCKETS) + 1, 11, 2 * MAX_SHIFT + 1), smoothing)
    for sequence in sequences:
        previous = None
        for time, places in sequence:
            for s, f in places:
                where[s, f] += 1
            lead = _lead(places)
            if previous is not None:
                across = lead[0] - previous[1][0] + 5
                along = int(np.clip(lead[1] - previous[1][1], -MAX_SHIFT, MAX_SHIFT)) + MAX_SHIFT
                move[_bucket(time - previous[0]), across, along] += 1
            previous = (time, lead)
    where = -np.log(where / where.sum())
    move = -np.log(move / move.sum(axis=(1, 2), keepdims=True))
    return {"where": np.round(where, 3).tolist(), "move": np.round(move, 3).tolist()}


def playable(pitches: tuple[int, ...], tuning: tuple[int, ...]) -> tuple[int, ...]:
    """Bring pitches into the guitar's range and keep the top ones that can be played together."""
    low, high = tuning[0], tuning[-1] + MAX_FRET
    moved = []
    for p in pitches:
        while p < low:
            p += 12
        while p > high:
            p -= 12
        moved.append(p)
    kept = tuple(sorted(set(moved), reverse=True)[:MAX_NOTES])
    while kept and not fingerings(kept, tuning):
        kept = kept[:-1]  # drop the lowest until the hand can manage
    return tuple(sorted(kept))


# ---------------------------------------------------------------- from the score


def strummed(groups: list[tuple[int, ...]]) -> bool:
    """A bar of chords: at least half its attacks are three or more notes at once."""
    if not groups:
        return False
    return sum(len(g) >= 3 for g in groups) * 2 >= len(groups)


def tab_from_events(events: list[tuple[int, int, tuple[int, ...]]], n_bars: int, bar: int, beat: int = 4, marks: dict | None = None, chords: list[tuple[int, Chord]] | None = None) -> Tab:
    """Tablature for note events (start, length, pitches) in sixteenths.

    marks: {(start, pitch): technique} for single notes reached by a hammer-on (h),
    pull-off (p), slide (/ or \\) or bend (b) from the note before.
    chords: (start, chord) changes; strummed bars show the shape of each chord."""
    marks = marks or {}
    by_bar: list[list[tuple[int, int, tuple[int, ...]]]] = [[] for _ in range(n_bars)]
    for start, length, pitches in sorted(events):
        if 0 <= start // bar < n_bars and pitches:
            by_bar[start // bar].append((start, length, tuple(sorted(pitches))))
    # Drop D when the music goes down to D or E-flat below the low E.
    lowest = [p for row in by_bar for _, _, ps in row for p in ps if 38 <= p < 40]
    name = "drop_d" if len(lowest) >= 2 else "standard"
    tuning = TUNINGS[name]

    bars = [TabBar(strummed=strummed([ps for _, _, ps in row])) for row in by_bar]
    todo: list[tuple[int, int, int, tuple[int, ...]]] = []  # (bar, start, length, pitches)
    for index, row in enumerate(by_bar):
        if bars[index].strummed:
            continue
        for start, length, pitches in row:
            kept = playable(pitches, tuning)
            if kept:
                todo.append((index, start, length, kept))
    # A technique joins a single note to the single note just before it.
    kinds = []
    for k, (_, start, _, pitches) in enumerate(todo):
        kind = marks.get((start, pitches[0]), "") if len(pitches) == 1 else ""
        if kind and (k == 0 or len(todo[k - 1][3]) != 1 or todo[k - 1][3][0] == pitches[0]):
            kind = ""
        if kind == "b" and not 1 <= pitches[0] - todo[k - 1][3][0] <= 3:
            kind = ""  # a bend reaches up to three semitones
        kinds.append(kind)
    # A bent-to note is played at the fret of the note it bends from, so it is not fingered itself.
    fingered = [k for k, kind in enumerate(kinds) if kind != "b"]
    joined = [bool(kinds[k]) and (i > 0 and fingered[i - 1] == k - 1) for i, k in enumerate(fingered)]
    chosen_for = dict(zip(fingered, place([(todo[k][1] / beat, todo[k][3]) for k in fingered], tuning, joined)))
    previous = None
    for k, (index, start, length, pitches) in enumerate(todo):
        kind = kinds[k]
        if kind == "b" and previous is not None:
            # Bent up to: one note, held through, bent at its own fret.
            previous.bend = pitches[0] - previous.pitches[0]
            previous.length = start + length - previous.start
            continue
        chosen = tuple(chosen_for[k])
        if kind and (previous is None or previous.places[0][0] != chosen[0][0]):
            kind = ""  # could not be kept on one string
        note = TabNote(start, length, pitches, tuple(chosen), kind)
        bars[index].notes.append(note)
        previous = note
    if chords:
        _add_shapes(bars, by_bar, bar, tuning, chords)
    return Tab(name, bars, bar)


def _add_shapes(bars: list[TabBar], by_bar, bar: int, tuning: tuple[int, ...], chords: list[tuple[int, Chord]]) -> None:
    """In strummed bars, the shape of the chord at the start of the bar and at each change in it."""
    changes = sorted(chords)
    times = [t for t, _ in changes]
    position = None
    for index, row in enumerate(bars):
        if not row.strummed:
            continue
        lo, hi = index * bar, (index + 1) * bar
        cuts = sorted({lo} | {t for t in times if lo < t < hi})
        for k, start in enumerate(cuts):
            end = cuts[k + 1] if k + 1 < len(cuts) else hi
            i = int(np.searchsorted(times, start, side="right")) - 1
            if i < 0 or changes[i][1] is None:
                continue
            heard = {p for s, _, ps in by_bar[index] if start <= s < end for p in ps}
            shape = chordshapes.choose(changes[i][1], tuning, heard, position)
            if shape is None:
                continue
            fretted = [f for f in shape if f]
            position = min(fretted) if fretted else 0
            pitches, places = chordshapes.as_places(shape, tuning)
            row.notes.append(TabNote(start, end - start, pitches, places, shape=True))


def tab_from_document(doc: dict, transpose: int = 0, source: str = "notes") -> Tab:
    """Tablature for a (clean) score document: all its notes, or just the melody line."""
    per_bar = doc["beats_per_bar"]
    bar = per_bar * 4
    n_bars = (len(doc["beats"]) - 1) // per_bar
    staves = doc["arrangements"]["lead" if source == "melody" else "piano"]
    together: dict[int, tuple[int, set[int]]] = {}
    for events in staves:
        for start, length, pitches in events:
            shifted = {p + transpose for p in pitches}
            if start in together:
                old_length, old = together[start]
                together[start] = (min(old_length, length), old | shifted)
            else:
                together[start] = (length, shifted)
    starts = sorted(together)
    events = []
    for i, start in enumerate(starts):
        length, ps = together[start]
        if i + 1 < len(starts):
            length = min(length, starts[i + 1] - start)  # one line: a note ends where the next begins
        events.append((start, length, tuple(sorted(ps))))
    marks = {}
    for start, pitch, kind in doc.get("techniques", []):
        for octave in (0, 12):  # a melody line may be written an octave up
            marks[(start, pitch + transpose + octave)] = kind
    chords = []
    for beat, symbol in doc["chords"]:
        parsed = parse_chord_symbol(symbol)
        chords.append((beat * 4, parsed.chord.transposed(transpose) if parsed else None))
    return tab_from_events(events, n_bars, bar, marks=marks, chords=chords)


# ---------------------------------------------------------------- plain-text tablature


def tab_text(tab: Tab, chords: list[tuple[int, str]], title: str = "", bars_per_line: int = 4, capo: int = 0) -> str:
    """Classic ASCII tab, chord symbols above, `bars_per_line` bars to a line.

    chords: (position in sixteenths, symbol), printed above the tab.
    """
    names = STRING_NAMES[tab.tuning]
    width = 2  # characters per sixteenth; two-digit frets fit
    head = []
    if title:
        head.append(title)
    if tab.tuning == "drop_d":
        head.append("Tuning: drop D (D A D G B E)")
    if capo:
        head.append(f"Capo {capo}")
    if any(n.technique or n.bend for b in tab.bars for n in b.notes):
        head.append("h hammer-on   p pull-off   / \\ slide up/down   b bend (7b9: fret 7 bent up to the pitch of fret 9)")
    out = ["\n".join(head) + "\n" if head else ""]
    for first in range(0, len(tab.bars), bars_per_line):
        row = tab.bars[first : first + bars_per_line]
        cells = len(row) * (tab.bar_length * width + 1) + 1
        lines = [["-"] * cells for _ in range(6)]
        above = [" "] * (cells + 8)
        for k, bar in enumerate(row):
            base = k * (tab.bar_length * width + 1)
            for line in lines:
                line[base] = "|"
            for note in bar.notes:
                col = base + 1 + (note.start - (first + k) * tab.bar_length) * width
                for string, fret in note.places:
                    line = lines[5 - string]
                    text = f"{fret}b{fret + note.bend}" if note.bend else str(fret)
                    for i, ch in enumerate(text):
                        if col + i < cells - 1:
                            line[col + i] = ch
                    if note.technique and col - 1 > base and line[col - 1] == "-":
                        line[col - 1] = note.technique
        for line in lines:
            line[-1] = "|"
        bar_start, bar_end = first * tab.bar_length, (first + len(row)) * tab.bar_length
        for position, symbol in chords:
            if bar_start <= position < bar_end:
                k = (position - bar_start) // tab.bar_length
                col = k * (tab.bar_length * width + 1) + 1 + ((position - bar_start) % tab.bar_length) * width
                for i, ch in enumerate(symbol):
                    if col + i < len(above):
                        above[col + i] = ch
        label = f"{first + 1}"
        out.append(" " + "".join(above).rstrip())
        for string in range(5, -1, -1):
            out.append(names[string] + "".join(lines[5 - string]))
        out.append(" " + label)
        out.append("")
    return "\n".join(out).rstrip() + "\n"
