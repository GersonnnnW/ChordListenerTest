"""Every output format, from one Analysis."""

from __future__ import annotations

from .midi import MidiTrack, write_midi
from .musicxml import to_musicxml
from .pipeline import Analysis
from .score import ScoreOptions, build_score


def musicxml(analysis: Analysis, options: ScoreOptions | None = None) -> str:
    return to_musicxml(build_score(analysis, options))


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
    key = analysis.key.transposed(transpose)
    grid = analysis.grid
    per_bar = grid.per_bar
    n_bars = max(1, (len(grid.times) - 1) // per_bar)
    slots: list[list[str]] = [["."] * per_bar for _ in range(n_bars)]
    carried = ""
    starts: dict[int, str] = {}
    for event in analysis.chords:
        starts[int(round(event.beat))] = event.chord.transposed(transpose).symbol(key) if event.chord else "N.C."
    cells = []
    for index in range(n_bars):
        for beat in range(per_bar):
            if index * per_bar + beat in starts:
                slots[index][beat] = starts[index * per_bar + beat]
        tokens = slots[index]
        if tokens[0] == ".":
            tokens[0] = carried or "N.C."
        changes = [t for t in tokens if t != "."]
        carried = changes[-1]
        cells.append(tokens[0] if len(changes) == 1 else " ".join(tokens))
    width = max(len(c) for c in cells)
    header = [f"{title}" + (f" - {artist}" if artist else "")] if title else []
    header.append(f"Key: {key.name} | Tempo: {analysis.bpm:.0f} bpm | Time: {per_bar}/4")
    lines = header + [""]
    for i in range(0, n_bars, bars_per_line):
        row = cells[i : i + bars_per_line]
        lines.append("| " + " | ".join(c.ljust(width) for c in row) + " |")
    return "\n".join(lines) + "\n"
