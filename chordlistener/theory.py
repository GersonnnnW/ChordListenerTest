"""Note names, key-aware spelling and the chord vocabulary."""

from __future__ import annotations

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
    tier: int  # 0 = triads, 1 = common extensions, 2 = everything


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
    ]
}

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
