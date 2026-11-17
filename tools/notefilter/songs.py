"""Random songs for training and testing the note filter.

Every song is a different combination of key, tempo, meter, chord progression,
playing pattern and General MIDI instruments, so that what the filter learns is
about how real and false notes look to the pitch model, not about one sound.
"""

from __future__ import annotations

import random

from tools import songgen

_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Progressions as (scale degree in semitones, quality), for major and minor keys.
_MAJOR = [
    [(0, ""), (7, ""), (9, "m"), (5, "")],
    [(0, ""), (9, "m"), (5, ""), (7, "")],
    [(9, "m"), (5, ""), (0, ""), (7, "")],
    [(0, ""), (5, ""), (7, ""), (5, "")],
    [(2, "m"), (7, ""), (0, ""), (9, "m")],
    [(0, ""), (4, "m"), (5, ""), (7, "")],
    [(0, ""), (7, ""), (5, ""), (0, "")],
    [(5, ""), (0, ""), (7, ""), (9, "m")],
    [(0, ""), (10, ""), (5, ""), (0, "")],
    [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (0, "")],
]
_MINOR = [
    [(0, "m"), (8, ""), (3, ""), (10, "")],
    [(0, "m"), (5, "m"), (7, ""), (0, "m")],
    [(0, "m"), (10, ""), (8, ""), (7, "")],
    [(0, "m"), (3, ""), (10, ""), (5, "m")],
    [(0, "m"), (8, ""), (10, ""), (0, "m")],
    [(0, "m"), (5, "m"), (10, ""), (3, "")],
]
_RICHER = {"": ["", "", "", "maj7", "7", "sus4", "6", "add9"], "m": ["m", "m", "m", "m7", "m7", "m6"]}

COMP = [0, 1, 2, 4, 5, 24, 25, 26, 27, 16, 18, 48, 89, 6, 11, 46]
BASS = [32, 33, 34, 35, 36, 38, 39, 43]
LEAD = [73, 65, 56, 40, 52, 53, 54, 80, 81, 71, 22, 68, 11, 0, 25, 66, 60, 42, 74, 85]
# Instruments kept out of training entirely, to test on sounds the filter has never met.
UNSEEN = {"comp": [2, 27, 18, 46], "bass": [35, 43], "lead": [68, 54, 22, 60, 85]}
PATTERNS = ["block", "strum", "arp", "sustain"]


def random_spec(seed: int, unseen: bool = False) -> dict:
    rng = random.Random(seed)
    minor = rng.random() < 0.35
    tonic = rng.randrange(12)
    beats_per_bar = 3 if rng.random() < 0.12 else 4
    template = rng.choice(_MINOR if minor else _MAJOR)
    rich = rng.random() < 0.35
    progression = []
    for degree, quality in template:
        if rich and rng.random() < 0.5:
            quality = rng.choice(_RICHER[quality])
        length = beats_per_bar if rng.random() < 0.75 or beats_per_bar == 3 else 2
        progression.append((_NAMES[(tonic + degree) % 12] + quality, length))
    bpm = rng.uniform(62, 172)
    bars_once = sum(length for _, length in progression) / beats_per_bar
    repeats = max(1, round(rng.uniform(24, 40) / (bars_once * beats_per_bar * 60 / bpm)))

    def pick(kind: str, pool: list[int]) -> int:
        choices = UNSEEN[kind] if unseen else [p for p in pool if p not in UNSEEN[kind]]
        return rng.choice(choices)

    return dict(
        progression=progression,
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        key=_NAMES[tonic] + ("m" if minor else ""),
        repeats=repeats,
        seed=seed,
        drums=rng.random() < 0.8,
        melody=rng.random() < 0.88,
        programs=(pick("comp", COMP), pick("bass", BASS), pick("lead", LEAD)),
        pattern=rng.choice(PATTERNS),
        lead_centre=rng.choice([62, 67, 72, 74, 76, 79]),
        levels=(rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0), rng.uniform(0.7, 1.05)),
        lead_in=rng.uniform(0.2, 1.2),
    )


def make(seed: int, unseen: bool = False) -> songgen.Song:
    return songgen.make_song(**random_spec(seed, unseen))
