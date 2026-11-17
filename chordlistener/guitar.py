"""Would these chords be easier to play somewhere else on the neck?

A song recorded in B flat minor is rarely *played* in B flat minor on a
guitar: the player puts a capo on the first fret and plays A minor shapes.
Chord sheets are written that way too, so showing only the sounding chords
can look "wrong" next to every other source. This suggests the transposition
a guitarist would actually use.
"""

from __future__ import annotations

from .theory import Chord

# How comfortable each chord is as an open-position shape, by (root pitch class, quality).
_OPEN = {
    "maj": {0: 1.0, 2: 1.0, 4: 1.0, 7: 1.0, 9: 1.0, 5: 0.55},
    "min": {9: 1.0, 4: 1.0, 2: 1.0, 11: 0.45, 6: 0.4},
    "7": {9: 0.9, 11: 0.9, 0: 0.9, 2: 0.9, 4: 0.9, 7: 0.9},
    "min7": {9: 0.95, 4: 0.95, 2: 0.95, 11: 0.5},
    "maj7": {0: 0.85, 5: 0.85, 7: 0.85, 9: 0.85, 2: 0.85, 4: 0.8},
    "sus4": {2: 0.9, 9: 0.9, 4: 0.9, 7: 0.7},
    "sus2": {2: 0.9, 9: 0.9, 0: 0.7},
    "add9": {0: 0.85, 7: 0.8},
    "5": {},
}
_FAMILY = {"6": "maj", "9": "7", "11": "7", "13": "7", "7sus4": "sus4", "maj9": "maj7", "min9": "min7", "min6": "min", "m7b5": "min7", "dim": "min", "dim7": "min", "aug": "maj"}


def ease(chord: Chord) -> float:
    """1.0 = a first-week open chord, 0.2 = a barre chord."""
    if chord.quality == "5":
        return 0.7  # power chords are the same shape everywhere
    table = _OPEN.get(chord.quality)
    if table is not None:
        return table.get(chord.root, 0.2)
    return 0.8 * _OPEN[_FAMILY.get(chord.quality, "maj")].get(chord.root, 0.25)


def _average(chords: list[tuple[Chord, float]], shift: int) -> float:
    weight = sum(w for _, w in chords)
    return sum(ease(c.transposed(shift)) * w for c, w in chords) / weight


def suggest(chords: list[tuple[Chord, float]]) -> dict | None:
    """Best way to make the chords easier, or None if they are fine as they are.

    `chords` is a list of (chord, how long it lasts). Returns the number of
    semitones to transpose the *written* chords by, and what to do to the
    guitar so they still sound in the recording's key.
    """
    if not chords:
        return None
    here = _average(chords, 0)
    if here >= 0.8:
        return None
    options = []
    for fret in range(1, 8):  # capo: play shapes that many semitones lower
        options.append((_average(chords, -fret) - 0.03 * fret, -fret, {"capo": fret}))  # lower frets preferred
    options.append((_average(chords, 1) - 0.05, 1, {"tune_down": 1}))  # guitar tuned down a half step
    score, shift, how = max(options, key=lambda o: o[0])
    if score - here < 0.12:
        return None
    return {"transpose": shift, **how, "ease_before": round(here, 2), "ease_after": round(score, 2)}
