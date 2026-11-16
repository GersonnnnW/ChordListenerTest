"""Every output format, from one Analysis."""

from __future__ import annotations

from . import document
from .document import ScoreOptions, ShowOptions, build_score
from .midi import MidiTrack, write_midi
from .musicxml import to_musicxml
from .pipeline import Analysis


def musicxml(analysis: Analysis, options: ScoreOptions | None = None) -> str:
    return to_musicxml(build_score(analysis, options))


def musicxml_from_document(doc: dict, arrangement: str = "piano", transpose: int = 0, show: ShowOptions | None = None) -> str:
    """Sheet music for a (validated) score document, including any corrections people made."""
    return to_musicxml(document.score_from_document(doc, arrangement, transpose, show))


def midi(analysis: Analysis, transpose: int = 0) -> bytes:
    """Performance MIDI: the detected notes with their original timing, plus chord markers."""
    key = analysis.key.transposed(transpose)
    notes = [
        (n.start, n.end, n.pitch + transpose, 30 + int(n.velocity * 97))
        for n in analysis.notes
        if 0 <= n.pitch + transpose <= 127
    ]
    markers = [(c.start, c.chord.transposed(transpose).symbol(key)) for c in analysis.chords if c.chord]
    track = MidiTrack(notes, program=0, channel=0, name="Transcription", markers=markers)
    return write_midi([track], bpm=analysis.bpm, time_signature=(analysis.grid.per_bar, 4), key_fifths=key.fifths, minor=key.minor)


def chord_chart(analysis: Analysis, transpose: int = 0, title: str = "", artist: str = "", bars_per_line: int = 4) -> str:
    """Plain-text chord chart, one cell per bar."""
    doc = document.document_from_analysis(analysis, title, artist)
    return document.chord_chart(doc, transpose, bars_per_line)
