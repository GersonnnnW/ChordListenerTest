"""Guitar playing techniques: hammer-ons, pull-offs and slides.

Read from the audio for notes in a single line (solos, riffs, melodies). When a
note follows the previous one and the pitch moves *continuously* from one to
the other on a single vibrating string (followed frame by frame in the fine
pitch curve Basic Pitch computes: three steps per semitone, every 11.6 ms), the
second note was not picked again:

  * it glides through the frets in between, or jumps more than four frets: a slide;
  * otherwise it is a hammer-on (up) or pull-off (down).

Checked against GuitarSet, whose six-string pickup gives each string's true
pitch: 78% of the pairs marked this way really are joined on one string without
a new pick (6% were picked; the rest are unclear in the reference). Only about
one in five joined pairs is found, so the marks are sparse but mostly right.

Bends are not detected: from an ordinary recording the attempt was right only
one time in five, so they are left out rather than guessed. The tab still writes
a bend when one is entered in the score document (technique "b").
"""

from __future__ import annotations

import numpy as np

from .pitch import FPS, Note

CONTOUR_BASE = 21  # MIDI note of the first contour bin
CONTOUR_STEPS = 3  # bins per semitone
LEGATO_GAP = 0.05  # seconds of silence at most between joined notes
ALONE = 0.05  # seconds: no other note may start this close for a note to count as "in a line"
GLIDE_FRAMES = 2  # frames between the two pitches that make a slide rather than a hammer-on
HAMMER_MAX = 4  # semitones: wider joined moves are slides
CODES = ("h", "p", "/", "\\", "b")  # hammer-on, pull-off, slide up, slide down, bent up to


def follow(contour: np.ndarray, pitch: float, first: int, last: int, reach: int = 3, keep: float = 0.35) -> np.ndarray:
    """Follow one note's pitch frame by frame (semitones, nan once it is lost).

    Each frame may move at most `reach` bins (a third of a semitone each) from
    the last, so the track glides with a bend or slide but cannot jump to
    another string that happens to ring louder."""
    first, last = max(0, first), min(len(contour), last)
    out = np.full(max(0, last - first), np.nan)
    if last <= first:
        return out
    k = int(round((pitch - CONTOUR_BASE) * CONTOUR_STEPS))
    lo, hi = max(0, k - 1), min(contour.shape[1], k + 2)
    k = lo + int(contour[first, lo:hi].argmax())
    start = float(contour[first, k])
    if start <= 0.2:
        return out
    for i, frame in enumerate(range(first, last)):
        lo, hi = max(0, k - reach), min(contour.shape[1], k + reach + 1)
        j = lo + int(contour[frame, lo:hi].argmax())
        if contour[frame, j] < keep * start:
            break
        k = j
        left, mid, right = contour[frame, max(j - 1, 0)], contour[frame, j], contour[frame, min(j + 1, contour.shape[1] - 1)]
        denom = left - 2 * mid + right
        shift = 0.5 * (left - right) / denom if abs(denom) > 1e-9 else 0.0
        out[i] = (j + float(np.clip(shift, -0.5, 0.5))) / CONTOUR_STEPS + CONTOUR_BASE
    return out


def detect(notes: list[Note], contour: np.ndarray | None) -> None:
    """Mark hammer-ons, pull-offs and slides on the notes (in place): `technique` on the second note."""
    for note in notes:
        note.technique = ""
    if contour is None or len(notes) < 2:
        return
    order = sorted(range(len(notes)), key=lambda i: notes[i].start)
    starts = np.array([notes[i].start for i in order])
    lo = np.searchsorted(starts, starts - ALONE)
    hi = np.searchsorted(starts, starts + ALONE)
    alone = (hi - lo) == 1
    for k in range(1, len(order)):
        a, b = notes[order[k - 1]], notes[order[k]]
        interval = b.pitch - a.pitch
        if not (alone[k] and alone[k - 1]) or not 1 <= abs(interval) <= 12 or b.start - a.end > LEGATO_GAP:
            continue
        stop = int(round(b.start * FPS))
        track = follow(contour, a.pitch, max(int(round(a.start * FPS)) + 1, stop - 15), stop + 5, reach=4)
        if not (np.abs(track - b.pitch) <= 0.45).any():
            continue  # the string never reached the new pitch: picked, or another string
        low, high = min(a.pitch, b.pitch), max(a.pitch, b.pitch)
        between = int(np.sum((track > low + 0.4) & (track < high - 0.4)))
        if between >= GLIDE_FRAMES or abs(interval) > HAMMER_MAX:
            b.technique = "/" if interval > 0 else "\\"
        else:
            b.technique = "h" if interval > 0 else "p"
