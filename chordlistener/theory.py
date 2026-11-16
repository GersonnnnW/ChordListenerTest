"""Note names, key-aware spelling and the chord vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STEPS = "FCGDAEB"  # one turn of the line of fifths, F = -1


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0..11
    minor: bool

    @property
    def fifths(self) -> int:
        """Key signature: sharps (+) or flats (-)."""
        major_tonic = (self.tonic + 3) % 12 if self.minor else self.tonic
        f = (major_tonic * 7) % 12
        return f - 12 if f > 6 else f  # F# major = 6 sharps, Db major = 5 flats

    @property
    def name(self) -> str:
        return f"{spell_pc(self.tonic, self)} {'minor' if self.minor else 'major'}"

    def transposed(self, semitones: int) -> "Key":
        return Key((self.tonic + semitones) % 12, self.minor)


C_MAJOR = Key(0, False)


def _lof(pc: int, key: Key) -> int:
    """Line-of-fifths position of a pitch class, spelled the way `key` would."""
    # Twelve consecutive fifths around the key. Minor keys lean sharp so the
    # raised 6th/7th come out right (G# rather than Ab in A minor).
    low = key.fifths - (2 if key.minor else 4)
    pos = (pc * 7) % 12
    while pos < low:
        pos += 12
    while pos >= low + 12:
        pos -= 12
    return pos


def spell(pc: int, key: Key = C_MAJOR) -> tuple[str, int]:
    """Return (step letter, alteration) for a pitch class."""
    pos = _lof(pc % 12, key)
    return _STEPS[(pos + 1) % 7], (pos + 1) // 7


def spell_pc(pc: int, key: Key = C_MAJOR) -> str:
    step, alter = spell(pc, key)
    return step + ("#" * alter if alter > 0 else "b" * -alter)


def spell_midi(pitch: int, key: Key = C_MAJOR) -> tuple[str, int, int]:
    """Return (step, alter, octave) for a MIDI note number."""
    step, alter = spell(pitch % 12, key)
    return step, alter, (pitch - alter) // 12 - 1


def note_name(pitch: int, key: Key = C_MAJOR) -> str:
    step, alter, octave = spell_midi(pitch, key)
    return f"{step}{'#' * alter if alter > 0 else 'b' * -alter}{octave}"


@dataclass(frozen=True)
class Quality:
    id: str
    suffix: str  # appended to the root in a chord symbol
    intervals: tuple[int, ...]
    musicxml: str
    tier: int  # 0 = triads, 1 = common extensions, 2 = everything detectable, 3 = only ever typed in by a person
    degrees: tuple[tuple[int, int, str], ...] = ()  # MusicXML <degree>: (value, alter, add|alter|subtract)


QUALITIES: dict[str, Quality] = {
    q.id: q
    for q in [
        Quality("maj", "", (0, 4, 7), "major", 0),
        Quality("min", "m", (0, 3, 7), "minor", 0),
        Quality("7", "7", (0, 4, 7, 10), "dominant", 1),
        Quality("maj7", "maj7", (0, 4, 7, 11), "major-seventh", 1),
        Quality("min7", "m7", (0, 3, 7, 10), "minor-seventh", 1),
        Quality("sus4", "sus4", (0, 5, 7), "suspended-fourth", 1),
        Quality("sus2", "sus2", (0, 2, 7), "suspended-second", 1),
        Quality("dim", "dim", (0, 3, 6), "diminished", 1),
        Quality("aug", "aug", (0, 4, 8), "augmented", 1),
        Quality("m7b5", "m7b5", (0, 3, 6, 10), "half-diminished", 2),
        Quality("dim7", "dim7", (0, 3, 6, 9), "diminished-seventh", 2),
        Quality("6", "6", (0, 4, 7, 9), "major-sixth", 2),
        Quality("min6", "m6", (0, 3, 7, 9), "minor-sixth", 2),
        Quality("5", "5", (0, 7), "power", 3),
        Quality("9", "9", (0, 4, 7, 10, 2), "dominant-ninth", 3),
        Quality("min9", "m9", (0, 3, 7, 10, 2), "minor-ninth", 3),
        Quality("maj9", "maj9", (0, 4, 7, 11, 2), "major-ninth", 3),
        Quality("11", "11", (0, 4, 7, 10, 2, 5), "dominant-11th", 3),
        Quality("13", "13", (0, 4, 7, 10, 2, 9), "dominant-13th", 3),
        Quality("add9", "add9", (0, 4, 7, 2), "major", 3, ((9, 0, "add"),)),
        Quality("7sus4", "7sus4", (0, 5, 7, 10), "suspended-fourth", 3, ((7, -1, "add"),)),
    ]
}

# Other common ways of writing the same chord.
_ALIASES = {
    "M": "maj", "maj": "maj", "major": "maj",
    "min": "min", "mi": "min", "-": "min", "minor": "min",
    "dom7": "7",
    "M7": "maj7", "ma7": "maj7", "Maj7": "maj7", "Δ": "maj7", "Δ7": "maj7", "j7": "maj7",
    "min7": "min7", "mi7": "min7", "-7": "min7",
    "sus": "sus4",
    "°": "dim", "o": "dim", "+": "aug", "#5": "aug",
    "ø": "m7b5", "ø7": "m7b5", "min7b5": "m7b5", "m7(b5)": "m7b5", "-7b5": "m7b5",
    "°7": "dim7", "o7": "dim7",
    "maj6": "6", "M6": "6", "min6": "min6", "-6": "min6",
    "min9": "min9", "-9": "min9", "M9": "maj9", "Maj9": "maj9",
    "(add9)": "add9", "add2": "add9", "2": "add9",
    "7sus": "7sus4", "sus7": "7sus4",
}
_BY_SUFFIX = {q.suffix: q.id for q in QUALITIES.values()}
_CHORD_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)([^/]*?)(?:/([A-Ga-g])([#b♯♭]?))?$")
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ALTER = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}


@dataclass(frozen=True)
class ParsedChord:
    """A chord symbol someone typed, keeping the spelling they chose."""

    chord: "Chord"
    root: tuple[str, int]  # (step, alter)
    bass: tuple[str, int] | None


def parse_chord_symbol(text: str) -> ParsedChord | None:
    """Understand a chord symbol such as "F#m7", "Bbmaj7/D" or "C-". None if it is not one."""
    match = _CHORD_RE.match(text.strip())
    if not match:
        return None
    step, accidental, suffix, bass_step, bass_accidental = match.groups()
    suffix = suffix.strip()
    quality = _BY_SUFFIX.get(suffix) or _ALIASES.get(suffix) or _BY_SUFFIX.get(suffix.lower()) or _ALIASES.get(suffix.lower())
    if quality is None:
        return None
    step = step.upper()
    root = (_NATURAL[step] + _ALTER[accidental]) % 12
    bass = spelled_bass = None
    if bass_step:
        bass_step = bass_step.upper()
        bass = (_NATURAL[bass_step] + _ALTER[bass_accidental]) % 12
        spelled_bass = (bass_step, _ALTER[bass_accidental])
    return ParsedChord(Chord(root, quality, bass), (step, _ALTER[accidental]), spelled_bass)

VOCABULARIES = {"basic": 0, "standard": 1, "full": 2}


@dataclass(frozen=True)
class Chord:
    root: int  # pitch class
    quality: str
    bass: int | None = None  # pitch class when it differs from the root

    def symbol(self, key: Key = C_MAJOR) -> str:
        text = spell_pc(self.root, key) + QUALITIES[self.quality].suffix
        if self.bass is not None and self.bass != self.root:
            text += "/" + spell_pc(self.bass, key)
        return text

    def pitch_classes(self) -> list[int]:
        return [(self.root + i) % 12 for i in QUALITIES[self.quality].intervals]

    def transposed(self, semitones: int) -> "Chord":
        bass = None if self.bass is None else (self.bass + semitones) % 12
        return Chord((self.root + semitones) % 12, self.quality, bass)
