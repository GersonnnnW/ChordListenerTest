"""Guitar chord shapes for strummed bars.

Where the guitar strums chords, a tab shows the shape to hold: the fret on each
string, stacked. Shapes are found rather than looked up: every way of holding
the chord on the neck is listed (each sounding string plays a chord tone, the
root or slash bass is lowest, all the chord's essential tones are there, four
fingers suffice with a barre, only bass strings are left out), and the one
chosen is the one that best matches the notes heard in that bar, preferring
open and low positions and staying near the previous shape.
"""

from __future__ import annotations

from functools import lru_cache

from .theory import QUALITIES, Chord

_REACH = 3  # frets a hand spans above its lowest fretted note
_TOP = 12  # highest position for a shape


@lru_cache(maxsize=512)
def voicings(root: int, quality: str, bass: int | None, tuning: tuple[int, ...]) -> tuple[tuple[int | None, ...], ...]:
    """Every playable shape: a fret (or None for not played) per string, lowest string first."""
    intervals = QUALITIES[quality].intervals
    tones = {(root + i) % 12 for i in intervals}
    lowest = (bass if bass is not None else root) % 12
    if bass is not None:
        tones.add(lowest)
    # The fifth (and for 11/13 chords the ninth/eleventh) may be left out, as guitarists do.
    optional = {(root + 7) % 12} if len(intervals) >= 3 else set()
    if len(intervals) >= 5:
        optional |= {(root + i) % 12 for i in intervals[3:-1]}
    essential = tones - optional
    found = set()
    for low in range(0, _TOP + 1):
        choices = []
        for open_ in tuning:
            options = [None]
            for fret in [0] + list(range(max(1, low), low + _REACH + 1)):
                if (open_ + fret) % 12 in tones:
                    options.append(fret)
            choices.append(sorted(set(options), key=lambda f: (f is None, f)))
        _search(choices, tuning, 0, [], essential, lowest, found)
    return tuple(sorted(found, key=lambda v: tuple(-1 if f is None else f for f in v)))


def _search(choices, tuning, string, picked, essential, lowest, found) -> None:
    if string == len(choices):
        _keep(tuple(picked), tuning, essential, lowest, found)
        return
    for fret in choices[string]:
        picked.append(fret)
        _search(choices, tuning, string + 1, picked, essential, lowest, found)
        picked.pop()


def _keep(shape, tuning, essential, lowest, found) -> None:
    sounding = [(s, f) for s, f in enumerate(shape) if f is not None]
    if len(sounding) < 4:
        return
    # Strings left out: any below the bass and above the top, plus at most one in between (jazz voicings).
    first, last = sounding[0][0], sounding[-1][0]
    if (last - first + 1) - len(sounding) > 1:
        return
    pitches = [tuning[s] + f for s, f in sounding]
    if pitches[0] % 12 != lowest or not essential <= {p % 12 for p in pitches}:
        return
    if fingers(shape) > 4:
        return
    found.add(shape)


def fingers(shape) -> int:
    """Fingers needed, letting the index finger barre the lowest fret across the strings it covers."""
    fretted = [(s, f) for s, f in enumerate(shape) if f]
    if not fretted:
        return 0
    if max(f for _, f in fretted) - min(f for _, f in fretted) > _REACH:
        return 9
    low = min(f for _, f in fretted)
    at_low = [s for s, f in fretted if f == low]
    # A barre needs every string from the first barred one up to be at or above the barre.
    barre_from = at_low[0]
    if len(at_low) >= 2 and all(shape[s] is None or shape[s] >= low for s in range(barre_from, len(shape))) and all(shape[s] != 0 for s in range(barre_from, len(shape))):
        return 1 + sum(1 for _, f in fretted if f > low)
    return len(fretted)


def _position(shape) -> int:
    fretted = [f for f in shape if f]
    return min(fretted) if fretted else 0


# Tuned on GuitarSet accompaniment, players 00-03 (python -m tools.evaluate_tabs DATASETS --shapes):
# the chosen shape matches the guitarist's string and fret on 70% of strings there, 57% on players 04-05.
WEIGHTS = {
    "position": 0.0,  # per fret up the neck
    "strings": 0.0,  # per string sounding
    "exact": 0.9,  # per note of the shape heard at exactly that pitch
    "pitch_class": 0.0,  # per note heard in some octave
    "unheard": 0.0,  # per note of the shape not heard at all
    "fingers": 0.09,  # per finger needed
    "move": 0.068,  # per fret away from the previous shape
    "gap": 0.0,  # a string left out in the middle
    "top_out": 0.15,  # per top string left out
    "open_high": 0.8,  # open strings mixed into a shape up the neck
    "span": 0.4,  # per fret of stretch beyond two
}

# With little heard (fewer than three different notes), what GuitarSet players
# reach for given only the chord name: mostly full shapes, often barred.
PRIOR = {"position": 0.0, "strings": 0.5, "exact": 0, "pitch_class": 0, "unheard": 0, "fingers": 0.0, "move": 0.15, "gap": 1.0, "top_out": 0.5, "open_high": 0.48, "span": 0.6}


def choose(chord: Chord, tuning: tuple[int, ...], heard: set[int], previous: int | None = None, weights: dict | None = None) -> tuple[int | None, ...] | None:
    """The most likely shape for this chord, given the pitches heard (MIDI numbers) and the previous shape's position."""
    options = voicings(chord.root, chord.quality, chord.bass, tuning)
    if not options:
        return None
    heard_classes = {p % 12 for p in heard}
    w = {**(WEIGHTS if len(heard_classes) >= 3 else PRIOR), **(weights or {})}
    best, best_cost = None, float("inf")
    for shape in options:
        sounding = [s for s, f in enumerate(shape) if f is not None]
        pitches = [tuning[s] + shape[s] for s in sounding]
        fretted = [f for f in shape if f]
        position = min(fretted) if fretted else 0
        cost = w["position"] * position - w["strings"] * len(pitches)
        cost -= w["exact"] * sum(p in heard for p in pitches)
        cost -= w["pitch_class"] * sum(p % 12 in heard_classes for p in pitches)
        if heard:
            cost += w["unheard"] * sum(p not in heard and p % 12 not in heard_classes for p in pitches)
        cost += w["fingers"] * fingers(shape)
        cost += w["gap"] * ((sounding[-1] - sounding[0] + 1) - len(sounding))
        cost += w["top_out"] * (len(shape) - 1 - sounding[-1])
        if fretted:
            cost += w["span"] * max(0, max(fretted) - position - 2)
            if position >= 4 and any(f == 0 for f in shape):
                cost += w["open_high"]
        if previous is not None:
            cost += w["move"] * abs(position - previous)
        if cost < best_cost:
            best, best_cost = shape, cost
    return best


def as_places(shape, tuning) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """(pitches, (string, fret) per pitch) for a shape."""
    sounding = [(s, f) for s, f in enumerate(shape) if f is not None]
    return tuple(tuning[s] + f for s, f in sounding), tuple(sounding)


