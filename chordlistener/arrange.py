"""Decide which detected notes are written down, and on which staff.

A recording contains far more "notes" than belong on a page: overtones that
ring above the notes actually played, the same string re-detected while it is
still sounding, and timing jitter that turns straight eighths into ragged
sixteenths. Everything here is about writing down what a musician would,
using the two things the analysis knows reliably - the chords and the beat -
to judge the individual notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipeline import Analysis
from .score import DIVISIONS, Event, voice
from .theory import QUALITIES

_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10, 11)  # natural minor plus the raised seventh
_OVERTONES = (12, 19, 24, 28, 31)  # octave, twelfth, two octaves, +third, +fifth
SPLIT = 60  # middle C: below it is the left hand

# (start, length, pitch, strength), start and length in sixteenths
Quantised = tuple[int, int, int, float]


@dataclass
class Arrangement:
    right: list[Event]
    left: list[Event]
    melody: list[Event]
    grid: int  # 1 = sixteenths, 2 = eighths


def choose_grid(positions: np.ndarray) -> int:
    """Sixteenths only if the music really uses them.

    Detected onsets wobble by a few tens of milliseconds. If hardly any land
    squarely between two eighths, the "sixteenths" are that wobble, and the
    page reads far better snapped to eighths.
    """
    if len(positions) < 16:
        return 2
    phase = positions % 2  # position within an eighth, in sixteenths
    off = np.abs(phase - 1.0) < 0.3  # squarely on the in-between sixteenth
    on = (phase < 0.3) | (phase > 1.7)
    return 1 if off.sum() > 0.22 * max(on.sum() + off.sum(), 1) else 2


def quantise(analysis: Analysis, total: int, grid: int | None = None) -> tuple[list[Quantised], int]:
    """Snap notes to the beat grid. Returns (notes, grid step in sixteenths)."""
    if not analysis.notes:
        return [], 2
    beat_grid = analysis.grid
    starts = beat_grid.position(np.array([n.start for n in analysis.notes])) * DIVISIONS
    ends = beat_grid.position(np.array([n.end for n in analysis.notes])) * DIVISIONS
    step = grid or choose_grid(starts)
    out: list[Quantised] = []
    for note, s, e in zip(analysis.notes, starts, ends):
        start = int(round(s / step)) * step
        if not 21 <= note.pitch <= 108 or start >= total:
            continue
        start = max(0, start)
        length = max(step, int(round((e - s) / step)) * step)
        out.append((start, min(length, total - start), note.pitch, note.velocity))
    out.sort()
    return out, step


def harmony_map(analysis: Analysis, total: int) -> tuple[list[frozenset[int]], frozenset[int]]:
    """For every sixteenth, the pitch classes of the chord sounding there; and the key's scale."""
    chord_at: list[frozenset[int]] = [frozenset()] * total
    for event in analysis.chords:
        if event.chord is None:
            continue
        tones = frozenset((event.chord.root + i) % 12 for i in QUALITIES[event.chord.quality].intervals)
        lo = max(0, int(round(event.beat * DIVISIONS)))
        hi = total
        chord_at[lo:hi] = [tones] * (hi - lo)  # until the next chord overwrites it
    scale = frozenset((analysis.key.tonic + d) % 12 for d in (_MINOR if analysis.key.minor else _MAJOR))
    return chord_at, scale


def find_melody(notes: list[Quantised], low: int = 52, high: int = 91, leap_cost: float = 0.5, floor: float = 0.3, top_bonus: float = 2.2) -> list[Quantised]:
    """Pick the tune: one note at a time, strong, on top, and moving smoothly.

    Each candidate note gets a worth (how strong and how long it is, and
    whether it is the top of what sounds with it); a dynamic-programming pass
    then finds the single line through the song with the most worth, paying
    for big leaps. Notes not worth their place are left out, so the line
    rests when the singer does instead of grabbing accompaniment.
    """
    cand = [n for n in notes if low <= n[2] <= high]
    if not cand:
        return []
    cand.sort(key=lambda n: (n[0], n[2]))
    starts = np.array([n[0] for n in cand])
    ends = np.array([n[0] + n[1] for n in cand])
    pitch = np.array([n[2] for n in cand], dtype=float)
    amp = np.array([n[3] for n in cand])

    worth = amp**2 * np.sqrt(np.minimum(ends - starts, 8) / 4.0)
    worth *= np.exp(-0.5 * ((pitch - 69) / 13.0) ** 2) * 0.6 + 0.4  # tunes live around A4, not in the bass
    for i in range(len(cand)):
        # A note that sits under a clearly sounding higher one is harmony, not tune.
        j0 = np.searchsorted(starts, starts[i] - 16)
        j1 = np.searchsorted(starts, starts[i], side="right")
        above = (pitch[j0:j1] > pitch[i]) & (ends[j0:j1] > starts[i]) & (amp[j0:j1] >= 0.8 * amp[i]) & (starts[j0:j1] >= starts[i] - 1)
        if not above.any():
            worth[i] *= top_bonus
    typical = float(np.percentile(worth, 85))
    gain = worth - floor * typical

    best = gain.copy()
    back = np.full(len(cand), -1)
    for j in range(len(cand)):
        lo = max(0, j - 48)
        prev = np.arange(lo, j)
        if not len(prev):
            continue
        ok = starts[prev] < starts[j]  # one note at a time
        if not ok.any():
            continue
        prev = prev[ok]
        leap = np.abs(pitch[prev] - pitch[j])
        rest = np.maximum(starts[j] - ends[prev], 0)
        # Leaps cost more the bigger they are, but a phrase break forgives them.
        cost = leap_cost * typical * np.maximum(leap - 2, 0) / 5.0 * np.exp(-rest / 16.0)
        total = best[prev] - cost
        k = int(total.argmax())
        if total[k] > 0 and total[k] + gain[j] > best[j]:
            best[j] = total[k] + gain[j]
            back[j] = prev[k]
    j = int(best.argmax())
    if best[j] <= 0:
        return []
    line = []
    while j >= 0:
        line.append(j)
        j = int(back[j])
    line.reverse()
    out: list[Quantised] = []
    for k, i in enumerate(line):
        start, length, p, a = cand[i]
        if k + 1 < len(line):
            length = min(length, cand[line[k + 1]][0] - start)  # the next note cuts this one off
        if length > 0:
            out.append((start, length, p, a))
    return out


def _is_overtone(note: Quantised, others: list[Quantised]) -> bool:
    """A faint note an octave or a twelfth above a stronger one struck with it."""
    start, length, pitch, strength = note
    if strength >= 0.45:
        return False
    for o_start, o_length, o_pitch, o_strength in others:
        if (pitch - o_pitch) in _OVERTONES and abs(o_start - start) <= 1 and start + length <= o_start + o_length + 1 and strength < 0.8 * o_strength:
            return True
    return False


SPAN = 14  # semitones one hand can cover comfortably (a ninth)


def arrange(analysis: Analysis, total: int, max_notes: int = 5, faithful: bool | None = None) -> Arrangement:
    """Lay the detected notes out for two hands, plus the tune on its own.

    With the trained note filter the detected notes can be trusted, so the
    score is *faithful*: every note is written, in the hand that can reach it.
    Without the filter (fixed thresholds), fall back to the cautious
    arrangement that only keeps what agrees with the chords.
    """
    notes, step = quantise(analysis, total)
    if not notes:
        return Arrangement([], [], [], step)
    if faithful is None:
        faithful = any(n.confidence < 1.0 for n in analysis.notes)
    chord_at, scale = harmony_map(analysis, total)
    found = find_melody(notes)
    tune = _tidy_tune(found, chord_at, scale, total)
    # A low vocal line is written an octave up on a lead sheet, where it reads comfortably in treble clef.
    shift = _treble_shift(tune)
    lead = [(s, l, p + shift, a) for s, l, p, a in tune]

    if faithful:
        right_notes, left_notes = _two_hands(notes)
        return Arrangement(
            right=voice(right_notes, total, max_notes, "top"),
            left=voice(left_notes, total, max_notes, "bottom"),
            melody=voice(lead, total, 1, "top"),
            grid=step,
        )

    tune_keys = {(n[0], n[2]) for n in found}  # whatever happens to them below, these are not accompaniment
    tune_at = {}  # sixteenth -> melody pitch (as written) sounding there
    for start, length, pitch, _ in lead:
        for pos in range(start, start + length):
            tune_at[pos] = pitch
    by_start: dict[int, list[Quantised]] = {}
    for note in notes:
        by_start.setdefault(note[0], []).append(note)
    right: list[Quantised] = list(lead)
    left: list[Quantised] = []
    for note in notes:
        start, length, pitch, strength = note
        if (start, pitch) in tune_keys:
            continue
        pc = pitch % 12
        chord = chord_at[start]
        in_chord = pc in chord if chord else pc in scale
        if pitch >= SPLIT:
            # Accompaniment in the right hand: only what agrees with the chord,
            # nothing above the tune, nothing that is just an overtone ringing.
            if not in_chord or pitch >= tune_at.get(start, 128):
                continue
            if _is_overtone(note, by_start[start]) or pitch > 88:
                continue
            right.append(note)
        else:
            # The bass may walk between chords, so strong scale notes stay too.
            if in_chord or (pc in scale and strength >= 0.5):
                snapped = min((start + 1) // 2 * 2, total - 2)
                left.append((snapped, max(2, (length + 1) // 2 * 2), pitch, strength))
    return Arrangement(
        right=voice(right, total, min(max_notes, 4), "top"),
        left=voice(left, total, min(max_notes, 3), "bottom"),
        melody=voice(lead, total, 1, "top"),
        grid=step,
    )


def _two_hands(notes: list[Quantised]) -> tuple[list[Quantised], list[Quantised]]:
    """Give every note to a hand that can reach it.

    Notes above middle C go to the right hand and the rest to the left; when
    the notes struck together in one hand span more than a ninth, the stray
    ones are passed to the other hand if it can take them.
    """
    by_start: dict[int, list[Quantised]] = {}
    for note in notes:
        by_start.setdefault(note[0], []).append(note)
    right: list[Quantised] = []
    left: list[Quantised] = []
    for start in sorted(by_start):
        group = sorted(by_start[start], key=lambda n: n[2])
        high = [n for n in group if n[2] >= SPLIT]
        low = [n for n in group if n[2] < SPLIT]
        # Too wide for the right hand: its lowest notes go to the left, if that keeps the left playable.
        while len(high) > 1 and high[-1][2] - high[0][2] > SPAN:
            stray = high.pop(0)
            if not low or stray[2] - low[0][2] <= SPAN:
                low.append(stray)
        while len(low) > 1 and low[-1][2] - low[0][2] > SPAN:
            stray = low.pop()
            if not high or high[-1][2] - stray[2] <= SPAN:
                high.insert(0, stray)
        right += high
        left += low
    return right, left


def _tidy_tune(tune: list[Quantised], chord_at: list[frozenset[int]], scale: frozenset[int], total: int) -> list[Quantised]:
    """Drop slips of the voice and let each note ring until the next one."""
    kept = []
    for note in tune:
        start, length, pitch, _ = note
        pc = pitch % 12
        # A brief note outside both the key and the chord is a slide or a
        # mis-hearing far more often than a real chromatic passing note.
        if length <= 2 and pc not in scale and pc not in chord_at[min(start, total - 1)]:
            continue
        kept.append(note)
    out = []
    for i, (start, length, pitch, strength) in enumerate(kept):
        nxt = kept[i + 1][0] if i + 1 < len(kept) else total
        gap = nxt - (start + length)
        if 0 < gap <= 4:
            length += gap  # singers hold notes; a rest shorter than a beat is not a rest
        out.append((start, min(length, nxt - start), pitch, strength))
    return out


def _treble_shift(tune: list[Quantised]) -> int:
    """Low (typically male) vocal lines are conventionally written an octave up on a lead sheet."""
    if not tune:
        return 0
    median = float(np.median([n[2] for n in tune]))
    return 12 if median < 60 else 0
