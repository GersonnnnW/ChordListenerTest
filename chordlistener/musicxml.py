"""MusicXML 4.0 output. Opens in MuseScore, Sibelius, Finale, Dorico, etc."""

from __future__ import annotations

from xml.sax.saxutils import escape

from . import __version__
from .score import DIVISIONS, NOTE_TYPES, Element, Harmony, Score
from .theory import QUALITIES, Key, spell_midi

_TUNING_STEPS = {0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("E", -1), 4: ("E", 0), 5: ("F", 0), 6: ("F", 1), 7: ("G", 0), 8: ("A", -1), 9: ("A", 0), 10: ("B", -1), 11: ("B", 0)}

_ACCIDENTALS = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}
_SHARP_ORDER = "FCGDAEB"


def _key_alterations(fifths: int) -> dict[str, int]:
    alter = {step: 0 for step in "CDEFGAB"}
    for step in _SHARP_ORDER[: max(fifths, 0)]:
        alter[step] = 1
    for step in _SHARP_ORDER[::-1][: max(-fifths, 0)]:
        alter[step] = -1
    return alter


def _words(text: str, placement: str, offset: int = 0) -> str:
    shift = f"<offset>{offset}</offset>" if offset else ""
    return f'<direction placement="{placement}"><direction-type><words>{escape(text)}</words></direction-type>{shift}<staff>1</staff></direction>'


def _harmony(harmony: Harmony, offset: int) -> str:
    if harmony.root is None or harmony.quality is None:
        # Not a chord we can describe structurally ("N.C.", or free text someone typed).
        return _words(harmony.text, "above", offset)
    quality = QUALITIES[harmony.quality]
    step, alter = harmony.root
    parts = [f'<harmony print-frame="no"><root><root-step>{step}</root-step>']
    if alter:
        parts.append(f"<root-alter>{alter}</root-alter>")
    parts.append(f'</root><kind text="{escape(quality.suffix)}">{quality.musicxml}</kind>')
    if harmony.bass is not None and harmony.bass != harmony.root:
        bass_step, bass_alter = harmony.bass
        parts.append(f"<bass><bass-step>{bass_step}</bass-step>")
        if bass_alter:
            parts.append(f"<bass-alter>{bass_alter}</bass-alter>")
        parts.append("</bass>")
    for value, degree_alter, kind in quality.degrees:
        parts.append(f"<degree><degree-value>{value}</degree-value><degree-alter>{degree_alter}</degree-alter><degree-type>{kind}</degree-type></degree>")
    if offset:
        parts.append(f"<offset>{offset}</offset>")
    parts.append("<staff>1</staff></harmony>")
    return "".join(parts)


def _lyric(text: str) -> str:
    return f'<lyric number="1"><syllabic>single</syllabic><text>{escape(text)}</text></lyric>' if text else ""


def _technical(element: Element, i: int) -> str:
    if not element.frets:
        return ""
    string, fret = element.frets[i]
    return f"<technical><string>{string}</string><fret>{fret}</fret></technical>"


def _note(element: Element, staff: int, voice: int, key: Key, state: dict, bar: int) -> list[str]:
    if element.slash:
        kind, dots = NOTE_TYPES[element.length]
        return [
            f"<note><pitch><step>B</step><octave>4</octave></pitch><duration>{element.length}</duration><voice>{voice}</voice>"
            f"<type>{kind}</type>{'<dot/>' * dots}<stem>none</stem><notehead>slash</notehead><staff>{staff}</staff>{_lyric(element.lyric)}</note>"
        ]
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
            if not element.tie_in and not element.frets:
                parts.append(f"<accidental>{_ACCIDENTALS[alter]}</accidental>")
            state[slot] = alter
        parts.append(f"<staff>{staff}</staff>")
        if element.tie_in or element.tie_out or element.frets:
            parts.append("<notations>")
            if element.tie_in:
                parts.append('<tied type="stop"/>')
            if element.tie_out:
                parts.append('<tied type="start"/>')
            parts.append(_technical(element, i))
            parts.append("</notations>")
        if i == 0:
            parts.append(_lyric(element.lyric))
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
    # Renderers print the work title as the heading and the movement title under it.
    lines.append(f"<work><work-title>{escape(score.title or 'Untitled')}</work-title></work>")
    if score.subtitle:
        lines.append(f"<movement-title>{escape(score.subtitle)}</movement-title>")
    lines.append("<identification>")
    if score.artist:
        lines.append(f'<creator type="composer">{escape(score.artist)}</creator>')
    lines.append(f"<encoding><software>ChordListener {__version__}</software></encoding></identification>")
    name = score.part_name or ("Piano" if score.n_staves == 2 else "Melody")
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
                attrs.append(f'<clef number="{i}"><sign>{sign}</sign><line>{ {"G": 2, "F": 4, "TAB": 5}[sign]}</line></clef>')
            for i, sign in enumerate(score.clefs, start=1):
                if sign == "TAB" and score.tab_tuning:
                    attrs.append(f'<staff-details number="{i}"><staff-lines>{len(score.tab_tuning)}</staff-lines>')
                    for line, pitch in enumerate(score.tab_tuning, start=1):
                        step, alter = _TUNING_STEPS[pitch % 12]
                        alter_xml = f"<tuning-alter>{alter}</tuning-alter>" if alter else ""
                        attrs.append(f'<staff-tuning line="{line}"><tuning-step>{step}</tuning-step>{alter_xml}<tuning-octave>{pitch // 12 - 1}</tuning-octave></staff-tuning>')
                    attrs.append("</staff-details>")
            if score.written_octave_up:
                attrs.append("<transpose><diatonic>0</diatonic><chromatic>0</chromatic><octave-change>-1</octave-change></transpose>")
            attrs.append("</attributes>")
            lines.append("".join(attrs))
            if score.bpm is not None:
                tempo = int(round(score.bpm))
                lines.append(
                    '<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                    f'<per-minute>{tempo}</per-minute></metronome></direction-type><staff>1</staff><sound tempo="{tempo}"/></direction>'
                )
        if measure.words:
            lines.append(_words(measure.words, "below"))
        state: dict = {"key": _key_alterations(key.fifths)}
        for index, elements in enumerate(measure.staves):
            staff = index + 1
            if index:
                lines.append(f"<backup><duration>{bar}</duration></backup>")
            pending = sorted(measure.harmonies, key=lambda h: h.offset) if index == 0 else []
            for element in elements:
                # A chord symbol belongs to the note sounding when it starts.
                while pending and pending[0].offset < element.start + element.length:
                    harmony = pending.pop(0)
                    lines.append(_harmony(harmony, max(0, harmony.offset - element.start)))
                lines.extend(_note(element, staff, 1 if index == 0 else 5, key, state, bar))
        if number == len(score.measures):
            lines.append('<barline location="right"><bar-style>light-heavy</bar-style></barline>')
        lines.append("</measure>")
    lines.append("</part></score-partwise>")
    return "\n".join(lines) + "\n"
