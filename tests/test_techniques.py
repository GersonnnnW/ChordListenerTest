"""Hammer-ons, pull-offs and slides from the pitch curve."""

import numpy as np

from chordlistener import techniques
from chordlistener.pitch import FPS, Note

BIN = techniques.CONTOUR_STEPS


def _contour(path):
    """A pitch curve: path[frame] = pitch (fractional semitones) or None."""
    out = np.zeros((len(path), 264), np.float32)
    for frame, pitch in enumerate(path):
        if pitch is not None:
            out[frame, int(round((pitch - techniques.CONTOUR_BASE) * BIN))] = 0.9
    return out


def _pair(path, a_pitch, b_pitch, split=30):
    a = Note(0.0, split / FPS, a_pitch, 0.8)
    b = Note(split / FPS, len(path) / FPS, b_pitch, 0.8)
    techniques.detect([a, b], _contour(path))
    return b.technique


def test_a_glide_is_a_slide():
    glide = [60.0] * 26 + [60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5] + [64.0] * 27
    assert _pair(glide, 60, 64) == "/"
    assert _pair(glide[::-1], 64, 60) == "\\"


def test_a_jump_on_one_string_is_a_hammer_on_or_pull_off():
    jump = [60.0] * 30 + [61.0] * 30
    assert _pair(jump, 60, 61) == "h"
    assert _pair(jump[::-1], 61, 60) == "p"


def test_a_new_note_elsewhere_is_picked():
    apart = [60.0] * 28 + [None] * 2 + [67.0] * 30
    assert _pair(apart, 60, 67) == ""


def test_notes_struck_together_are_left_alone():
    a, b, c = Note(0.0, 0.3, 60, 0.8), Note(0.3, 0.6, 61, 0.8), Note(0.31, 0.6, 64, 0.8)
    techniques.detect([a, b, c], _contour([60.0] * 26 + [61.0] * 26))
    assert b.technique == "" and c.technique == ""
