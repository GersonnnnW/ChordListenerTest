"""The editable score document.

An analysis is turned into a plain JSON document once; from then on that
document is what gets displayed, edited by people, saved, and rendered to
MusicXML. Keeping it plain data means corrections made in the browser round
trip without the engine having to understand "edits".

    {
      "version": 1,
      "title": "...", "artist": "...",
      "key": {"tonic": 0, "minor": false},        # tonic as a pitch class, C = 0
      "bpm": 73.0,
      "beats_per_bar": 4,
      "beats": [0.51, 1.33, ...],                 # seconds; bars * beats_per_bar + 1 entries
      "chords": [[0, "C"], [4, "G"], ...],        # [beat index, symbol as written]
      "lyrics": {"0": "When I find myself", ...}, # bar index -> words sung in that bar
      "arrangements": {
        "piano": [[[start, length, [pitches]], ...], [...]],   # one list per staff
        "lead":  [[[start, length, [pitches]], ...]]
      }
    }

Note positions and lengths are in sixteenth notes from the start of the piece;
pitches are MIDI numbers at the original (untransposed) pitch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipeline import Analysis
from .score import DIVISIONS, Harmony, Measure, Score, lay_out, melody, slashes, underlay, voice
from .theory import Key, parse_chord_symbol, spell, spell_pc

ARRANGEMENTS = {"piano": 2, "lead": 1}  # name -> number of staves
MAX_BARS = 2000
_NO_CHORD = {"n.c.", "nc", "n.c", "-", "–"}


class DocumentError(ValueError):
    """The document is malformed. The message is safe to show to the person who sent it."""


@dataclass
class ShowOptions:
    notes: bool = True
    chords: bool = True
    lyrics: bool = True
    tempo: bool = True
    key: bool = True


@dataclass
class ScoreOptions:
    arrangement: str = "piano"  # piano (two staves) | lead (melody line)
    transpose: int = 0
    title: str = ""
    artist: str = ""
    max_notes_per_chord: int = 4
    include_chords: bool = True


# --------------------------------------------------------------------------
# Analysis -> document
# --------------------------------------------------------------------------


def document_from_analysis(analysis: Analysis, title: str = "", artist: str = "", max_notes_per_chord: int = 4) -> dict:
    grid = analysis.grid
    bar = grid.per_bar * DIVISIONS
    n_bars = max(1, (len(grid.times) - 1) // grid.per_bar)
    total = n_bars * bar

    quantised: list[tuple[int, int, int, float]] = []
    if analysis.notes:
        starts = grid.position(np.array([n.start for n in analysis.notes])) * DIVISIONS
        ends = grid.position(np.array([n.end for n in analysis.notes])) * DIVISIONS
        for note, s, e in zip(analysis.notes, starts, ends):
            start = int(round(s))
            if not 21 <= note.pitch <= 108 or start >= total:
                continue
            start = max(0, start)
            length = max(1, int(round(e)) - start)
            quantised.append((start, min(length, total - start), note.pitch, note.velocity))

    chords: list[list] = []
    previous = None
    for event in analysis.chords:
        beat = int(round(event.beat))
        symbol = event.chord.symbol(analysis.key) if event.chord else "N.C."
        if symbol != previous and 0 <= beat < n_bars * grid.per_bar:
            chords.append([beat, symbol])
        previous = symbol

    def staff(events) -> list[list]:
        return [[start, length, list(pitches)] for start, length, pitches in events]

    upper = voice([n for n in quantised if n[2] >= 60], total, max_notes_per_chord, "top")
    lower = voice([n for n in quantised if n[2] < 60], total, max_notes_per_chord, "bottom")
    line = voice(melody(quantised), total, 1, "top")
    return {
        "version": 1,
        "title": title,
        "artist": artist,
        "key": {"tonic": analysis.key.tonic, "minor": analysis.key.minor},
        "bpm": round(analysis.bpm, 1),
        "beats_per_bar": grid.per_bar,
        "beats": [round(float(t), 3) for t in grid.times[: n_bars * grid.per_bar + 1]],
        "chords": chords,
        "lyrics": {},
        "arrangements": {"piano": [staff(upper), staff(lower)], "lead": [staff(line)]},
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _text(value, limit: int, what: str) -> str:
    if not isinstance(value, str):
        raise DocumentError(f"{what} must be text")
    value = " ".join(value.split())
    if len(value) > limit:
        raise DocumentError(f"{what} is longer than {limit} characters")
    return value


def _integer(value, low: int, high: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise DocumentError(f"{what} must be a whole number from {low} to {high}")
    return value


def normalize_document(raw) -> dict:
    """Check a document from an untrusted source and return a clean copy."""
    if not isinstance(raw, dict):
        raise DocumentError("The score must be a JSON object")
    per_bar = raw.get("beats_per_bar")
    if per_bar not in (3, 4):
        raise DocumentError("beats_per_bar must be 3 or 4")
    beats = raw.get("beats")
    if not isinstance(beats, list) or len(beats) < per_bar + 1 or (len(beats) - 1) % per_bar:
        raise DocumentError("beats must list every beat plus the final bar line")
    n_bars = (len(beats) - 1) // per_bar
    if n_bars > MAX_BARS:
        raise DocumentError(f"The score is longer than {MAX_BARS} bars")
    clean_beats = []
    for value in beats:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not -60 <= value <= 7200:
            raise DocumentError("beats must be times in seconds")
        if clean_beats and value <= clean_beats[-1]:
            raise DocumentError("beats must be in increasing order")
        clean_beats.append(round(float(value), 3))
    total = n_bars * per_bar * DIVISIONS

    key = raw.get("key")
    if not isinstance(key, dict) or not isinstance(key.get("minor"), bool):
        raise DocumentError("key must give a tonic and whether it is minor")
    tonic = _integer(key.get("tonic"), 0, 11, "key.tonic")
    bpm = raw.get("bpm")
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not 20 <= bpm <= 400:
        raise DocumentError("bpm must be a number from 20 to 400")

    chords_in = raw.get("chords", [])
    if not isinstance(chords_in, list):
        raise DocumentError("chords must be a list")
    by_beat: dict[int, str] = {}
    for item in chords_in:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise DocumentError("each chord must be [beat, symbol]")
        beat = _integer(item[0], 0, n_bars * per_bar - 1, "chord beat")
        symbol = _text(item[1], 24, "A chord symbol")
        if symbol:
            by_beat[beat] = symbol

    lyrics_in = raw.get("lyrics", {})
    if not isinstance(lyrics_in, dict):
        raise DocumentError("lyrics must map bar numbers to text")
    lyrics = {}
    for bar_key, text in lyrics_in.items():
        if not isinstance(bar_key, str) or not bar_key.isdigit() or int(bar_key) >= n_bars:
            raise DocumentError("lyrics refer to a bar that does not exist")
        text = _text(text, 200, "A lyric line")
        if text:
            lyrics[str(int(bar_key))] = text

    arrangements_in = raw.get("arrangements")
    if not isinstance(arrangements_in, dict):
        raise DocumentError("arrangements is missing")
    arrangements = {}
    for name, n_staves in ARRANGEMENTS.items():
        staves_in = arrangements_in.get(name)
        if not isinstance(staves_in, list) or len(staves_in) != n_staves:
            raise DocumentError(f"The {name} arrangement must have {n_staves} staff/staves")
        staves = []
        for events_in in staves_in:
            if not isinstance(events_in, list) or len(events_in) > total:
                raise DocumentError("A staff must be a list of notes")
            events = []
            for event in events_in:
                if not isinstance(event, (list, tuple)) or len(event) != 3 or not isinstance(event[2], (list, tuple)):
                    raise DocumentError("each note must be [start, length, [pitches]]")
                start = _integer(event[0], 0, total - 1, "A note's start")
                length = _integer(event[1], 1, total, "A note's length")
                pitches = sorted({_integer(p, 21, 108, "A pitch") for p in event[2]})
                if not 1 <= len(pitches) <= 8:
                    raise DocumentError("A chord needs between 1 and 8 notes")
                events.append([start, length, pitches])
            events.sort(key=lambda e: e[0])
            for i, event in enumerate(events):
                limit = events[i + 1][0] if i + 1 < len(events) else total
                if event[0] == limit:
                    raise DocumentError("Two notes on one staff start at the same moment; combine them into a chord")
                event[1] = min(event[1], limit - event[0])  # a note ends where the next begins
            staves.append(events)
        arrangements[name] = staves

    return {
        "version": 1,
        "title": _text(raw.get("title", ""), 200, "The title"),
        "artist": _text(raw.get("artist", ""), 200, "The artist"),
        "key": {"tonic": tonic, "minor": key["minor"]},
        "bpm": round(float(bpm), 1),
        "beats_per_bar": per_bar,
        "beats": clean_beats,
        "chords": [[beat, by_beat[beat]] for beat in sorted(by_beat)],
        "lyrics": lyrics,
        "arrangements": arrangements,
    }


# --------------------------------------------------------------------------
# Document -> what to show
# --------------------------------------------------------------------------


def _key(doc: dict, transpose: int) -> Key:
    return Key(doc["key"]["tonic"], doc["key"]["minor"]).transposed(transpose)


def _chord_text(symbol: str, key: Key, transpose: int) -> tuple[str, list[str], str | None]:
    """How a stored chord symbol reads after transposing: (symbol, note names, root name)."""
    if symbol.lower() in _NO_CHORD:
        return "N.C.", [], None
    parsed = parse_chord_symbol(symbol)
    if parsed is None:
        return symbol, [], None  # not understood: show it exactly as written
    chord = parsed.chord.transposed(transpose)
    # Untransposed, keep the spelling the person chose (A# stays A#).
    shown = symbol if transpose == 0 else chord.symbol(key)
    return shown, [spell_pc(pc, key) for pc in chord.pitch_classes()], spell_pc(chord.root, key)


def document_view(doc: dict, transpose: int = 0) -> dict:
    """Everything a client needs to display the chords of a (clean) document."""
    key = _key(doc, transpose)
    per_bar = doc["beats_per_bar"]
    beats = doc["beats"]
    chords = []
    for i, (beat, symbol) in enumerate(doc["chords"]):
        shown, notes, root = _chord_text(symbol, key, transpose)
        end_beat = doc["chords"][i + 1][0] if i + 1 < len(doc["chords"]) else len(beats) - 1
        chords.append(
            {
                "start": beats[beat],
                "end": beats[end_beat],
                "chord": shown,
                "root": root,
                "notes": notes,
                "bar": beat // per_bar + 1,
                "beat": beat % per_bar + 1,
            }
        )
    return {
        "key": {"name": key.name, "tonic": spell_pc(key.tonic, key), "mode": "minor" if key.minor else "major"},
        "tempo_bpm": doc["bpm"],
        "time_signature": f"{per_bar}/4",
        "bars": (len(beats) - 1) // per_bar,
        "duration": beats[-1],
        "beats": beats,
        "chords": chords,
        "has_lyrics": bool(doc["lyrics"]),
        "transpose": transpose,
    }


def score_from_document(doc: dict, arrangement: str = "piano", transpose: int = 0, show: ShowOptions | None = None) -> Score:
    """Lay a (clean) document out as notation."""
    show = show or ShowOptions()
    if arrangement not in ARRANGEMENTS:
        raise DocumentError(f"arrangement must be one of: {', '.join(ARRANGEMENTS)}")
    key = _key(doc, transpose)
    per_bar = doc["beats_per_bar"]
    bar = per_bar * DIVISIONS
    n_bars = (len(doc["beats"]) - 1) // per_bar

    harmonies: list[list[Harmony]] = [[] for _ in range(n_bars)]
    if show.chords:
        for beat, symbol in doc["chords"]:
            position = beat * DIVISIONS
            parsed = parse_chord_symbol(symbol)
            shown = _chord_text(symbol, key, transpose)[0]
            harmony = Harmony(position % bar, shown)
            if parsed is not None:
                chord = parsed.chord.transposed(transpose)
                harmony.quality = chord.quality
                if transpose == 0:
                    harmony.root, harmony.bass = parsed.root, parsed.bass
                else:
                    harmony.root = spell(chord.root, key)
                    harmony.bass = spell(chord.bass, key) if chord.bass is not None else None
            harmonies[position // bar].append(harmony)
    cuts = [{h.offset for h in row} for row in harmonies]
    no_cuts: list[set[int]] = [set() for _ in range(n_bars)]

    if show.notes:
        staves = []
        for index, events in enumerate(doc["arrangements"][arrangement]):
            shifted = [(s, l, tuple(p + transpose for p in ps if 21 <= p + transpose <= 108)) for s, l, ps in events]
            staves.append(lay_out([e for e in shifted if e[2]], n_bars, bar, cuts if index == 0 else no_cuts))
        clefs = ["G", "F"] if arrangement == "piano" else ["G"]
    else:
        staves = [slashes(n_bars, per_bar)]
        clefs = ["G"]

    measures = [Measure([staff[i] for staff in staves], harmonies[i]) for i in range(n_bars)]
    if show.lyrics:
        for bar_key, text in doc["lyrics"].items():
            measure = measures[int(bar_key)]
            if not underlay(measure.staves[0], text):
                measure.words = text

    details = []
    if show.key:
        details.append(f"Key: {key.name}")
    if show.tempo:
        details.append(f"Tempo: {round(doc['bpm'])} BPM")
    return Score(
        measures=measures,
        key=key,
        beats_per_bar=per_bar,
        bpm=doc["bpm"] if show.tempo else None,
        title=doc["title"],
        artist=doc["artist"],
        subtitle="  |  ".join(details),
        n_staves=len(staves),
        clefs=clefs,
    )


def build_score(analysis: Analysis, options: ScoreOptions | None = None) -> Score:
    """Straight from an analysis to notation, with no human edits."""
    options = options or ScoreOptions()
    doc = document_from_analysis(analysis, options.title, options.artist, options.max_notes_per_chord)
    show = ShowOptions(chords=options.include_chords)
    return score_from_document(doc, options.arrangement, options.transpose, show)


def chord_chart(doc: dict, transpose: int = 0, bars_per_line: int = 4, show: ShowOptions | None = None) -> str:
    """Plain-text chord chart, one cell per bar, with any lyrics under each row."""
    show = show or ShowOptions()
    view = document_view(doc, transpose)
    per_bar = doc["beats_per_bar"]
    n_bars = view["bars"]
    starts = {(c["bar"] - 1) * per_bar + c["beat"] - 1: c["chord"] for c in view["chords"]}
    cells = []
    carried = ""
    for index in range(n_bars):
        tokens = [starts.get(index * per_bar + beat, ".") for beat in range(per_bar)]
        if tokens[0] == ".":
            tokens[0] = carried or "N.C."
        changes = [t for t in tokens if t != "."]
        carried = changes[-1]
        cells.append(tokens[0] if len(changes) == 1 else " ".join(tokens))
    width = max(len(c) for c in cells)
    lines = []
    if doc["title"]:
        lines.append(doc["title"] + (f" - {doc['artist']}" if doc["artist"] else ""))
    facts = []
    if show.key:
        facts.append(f"Key: {view['key']['name']}")
    if show.tempo:
        facts.append(f"Tempo: {round(doc['bpm'])} bpm")
    facts.append(f"Time: {per_bar}/4")
    lines += [" | ".join(facts), ""]
    for i in range(0, n_bars, bars_per_line):
        if show.chords:
            lines.append("| " + " | ".join(c.ljust(width) for c in cells[i : i + bars_per_line]) + " |")
        words = [doc["lyrics"].get(str(b), "") for b in range(i, min(i + bars_per_line, n_bars))]
        if show.lyrics and any(words):
            lines.append("  " + " / ".join(w for w in words if w))
    return "\n".join(lines) + "\n"
