"""MusicXML 4.0 output. Opens in MuseScore, Sibelius, Finale, Dorico, etc."""

from __future__ import annotations

from xml.sax.saxutils import escape

from . import __version__
from .score import DIVISIONS, NOTE_TYPES, Element, Score
from .theory import QUALITIES, Chord, Key, spell, spell_midi

_ACCIDENTALS = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}
_SHARP_ORDER = "FCGDAEB"


def _key_alterations(fifths: int) -> dict[str, int]:
    alter = {step: 0 for step in "CDEFGAB"}
    for step in _SHARP_ORDER[: max(fifths, 0)]:
        alter[step] = 1
    for step in _SHARP_ORDER[::-1][: max(-fifths, 0)]:
        alter[step] = -1
    return alter


def _harmony(chord: Chord, key: Key, offset: int) -> str:
    quality = QUALITIES[chord.quality]
    step, alter = spell(chord.root, key)
    parts = [f'<harmony print-frame="no"><root><root-step>{step}</root-step>']
    if alter:
        parts.append(f"<root-alter>{alter}</root-alter>")
    parts.append(f'</root><kind text="{escape(quality.suffix)}">{quality.musicxml}</kind>')
    if chord.bass is not None and chord.bass != chord.root:
        bass_step, bass_alter = spell(chord.bass, key)
        parts.append(f"<bass><bass-step>{bass_step}</bass-step>")
        if bass_alter:
            parts.append(f"<bass-alter>{bass_alter}</bass-alter>")
        parts.append("</bass>")
    if offset:
        parts.append(f"<offset>{offset}</offset>")
    parts.append("<staff>1</staff></harmony>")
    return "".join(parts)


def _note(element: Element, staff: int, voice: int, key: Key, state: dict, bar: int) -> list[str]:
    if not element.pitches:
        if element.start == 0 and element.length == bar:
            return [f'<note><rest measure="yes"/><duration>{bar}</duration><voice>{voice}</voice><staff>{staff}</staff></note>']
        kind, dots = NOTE_TYPES[element.length]
        return [f"<note><rest/><duration>{element.length}</duration><voice>{voice}</voice><type>{kind}</type>{'<dot/>' * dots}<staff>{staff}</staff></note>"]
    kind, dots = NOTE_TYPES[element.length]
    out = []
    for i, pitch in enumerate(element.pitches):
        step, alter, octave = spell_midi(pitch, key)
        parts = ["<note>"]
        if i:
            parts.append("<chord/>")
        parts.append(f"<pitch><step>{step}</step>")
        if alter:
            parts.append(f"<alter>{alter}</alter>")
        parts.append(f"<octave>{octave}</octave></pitch><duration>{element.length}</duration>")
        if element.tie_in:
            parts.append('<tie type="stop"/>')
        if element.tie_out:
            parts.append('<tie type="start"/>')
        parts.append(f"<voice>{voice}</voice><type>{kind}</type>{'<dot/>' * dots}")
        # Print an accidental whenever the pitch departs from what the bar so far implies.
        slot = (staff, step, octave)
        if state.get(slot, state["key"][step]) != alter:
            if not element.tie_in:
                parts.append(f"<accidental>{_ACCIDENTALS[alter]}</accidental>")
            state[slot] = alter
        parts.append(f"<staff>{staff}</staff>")
        if element.tie_in or element.tie_out:
            parts.append("<notations>")
            if element.tie_in:
                parts.append('<tied type="stop"/>')
            if element.tie_out:
                parts.append('<tied type="start"/>')
            parts.append("</notations>")
        parts.append("</note>")
        out.append("".join(parts))
    return out


def to_musicxml(score: Score) -> str:
    key = score.key
    bar = score.beats_per_bar * DIVISIONS
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="4.0">',
    ]
    if score.title:
        lines.append(f"<work><work-title>{escape(score.title)}</work-title></work>")
    lines.append("<identification>")
    if score.artist:
        lines.append(f'<creator type="composer">{escape(score.artist)}</creator>')
    lines.append(f"<encoding><software>ChordListener {__version__}</software></encoding></identification>")
    name = "Piano" if score.n_staves == 2 else "Melody"
    lines.append(f'<part-list><score-part id="P1"><part-name>{name}</part-name></score-part></part-list>')
    lines.append('<part id="P1">')

    for number, measure in enumerate(score.measures, start=1):
        lines.append(f'<measure number="{number}">')
        if number == 1:
            attrs = [f"<attributes><divisions>{DIVISIONS}</divisions>"]
            attrs.append(f"<key><fifths>{key.fifths}</fifths><mode>{'minor' if key.minor else 'major'}</mode></key>")
            attrs.append(f"<time><beats>{score.beats_per_bar}</beats><beat-type>4</beat-type></time>")
            if score.n_staves > 1:
                attrs.append(f"<staves>{score.n_staves}</staves>")
            for i, sign in enumerate(score.clefs, start=1):
                attrs.append(f'<clef number="{i}"><sign>{sign}</sign><line>{2 if sign == "G" else 4}</line></clef>')
            attrs.append("</attributes>")
            lines.append("".join(attrs))
            tempo = int(round(score.bpm))
            lines.append(
                '<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                f'<per-minute>{tempo}</per-minute></metronome></direction-type><staff>1</staff><sound tempo="{tempo}"/></direction>'
            )
        state: dict = {"key": _key_alterations(key.fifths)}
        for index, elements in enumerate(measure.staves):
            staff = index + 1
            if index:
                lines.append(f"<backup><duration>{bar}</duration></backup>")
            pending = sorted(measure.harmonies, key=lambda h: h[0]) if index == 0 else []
            for element in elements:
                # A chord symbol belongs to the note sounding when it starts.
                while pending and pending[0][0] < element.start + element.length:
                    offset, chord = pending.pop(0)
                    lines.append(_harmony(chord, key, max(0, offset - element.start)))
                lines.extend(_note(element, staff, 1 if index == 0 else 5, key, state, bar))
        if number == len(score.measures):
            lines.append('<barline location="right"><bar-style>light-heavy</bar-style></barline>')
        lines.append("</measure>")
    lines.append("</part></score-partwise>")
    return "\n".join(lines) + "\n"
