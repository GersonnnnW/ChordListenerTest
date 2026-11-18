"""Bring people's own sheet music and chord sheets into a song.

An import becomes a score document (see document.py) for a song that has
already been analysed. The analysis supplies the clock: where every beat of
the recording falls. The import supplies the music:

  * MusicXML (.musicxml, .xml, or compressed .mxl) from MuseScore, Sibelius,
    Finale, Dorico...: notes, chord symbols, lyrics, key and tempo. Bar 1 of
    the file is placed on the bar of the recording chosen by the person
    (`start_bar`), and every quarter note on the next beat of the recording.
  * A chord sheet, as text:
      - with bar lines, "| C | G | Am F |": each bar is a bar of the recording,
        from `start_bar` on; "%" or an empty bar repeats the one before;
      - without bar lines (chords over lyrics, ChordPro "[C]words", or just
        "C G Am F"): the chords, in order, are lined up with what ChordListener
        heard, so each one starts where the recording most likely changes to it.
    A chord sheet replaces the chords (and lyrics, when it has them); the notes
    stay as ChordListener heard them.
  * A ChordListener score document (the JSON download), as is.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from xml.etree import ElementTree

import numpy as np

from .document import DocumentError, normalize_document
from .theory import QUALITIES, parse_chord_symbol

MAX_IMPORT_BYTES = 5 * 1024 * 1024
_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_KINDS = {q.musicxml: q.suffix for q in QUALITIES.values() if q.id not in ("add9", "7sus4")}
_KINDS.update({"major-minor": "7", "augmented-seventh": "7", "suspended-fourth": "sus4", "dominant-seventh": "7", "minor-major": "m", "major-11th": "maj7", "minor-11th": "m7", "major-13th": "maj7", "minor-13th": "m7", "none": None})


class ImportError_(DocumentError):
    """The file or text could not be read. The message is safe to show."""


def import_any(base: dict, data: bytes | str, filename: str = "", start_bar: int = 1) -> dict:
    """A new document for the song of `base` (a clean document), from what was uploaded or pasted."""
    if isinstance(data, bytes):
        if len(data) > MAX_IMPORT_BYTES:
            raise ImportError_("The file is larger than 5 MB")
        if data[:2] == b"PK":
            return import_musicxml(base, _unzip_mxl(data), start_bar)
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
    else:
        text = data
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            return normalize_document(json.loads(stripped))
        except ValueError as exc:
            raise ImportError_("The file looks like JSON but could not be read") from exc
    if stripped.startswith("<"):
        return import_musicxml(base, text.encode("utf-8"), start_bar)
    return import_chords(base, text, start_bar)


# ---------------------------------------------------------------- MusicXML


def _unzip_mxl(data: bytes) -> bytes:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        names = archive.namelist()
        target = None
        if "META-INF/container.xml" in names:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            root = container.find(".//{*}rootfile")
            target = root.get("full-path") if root is not None else None
        target = target or next((n for n in names if n.endswith((".musicxml", ".xml")) and not n.startswith("META-INF")), None)
        if target is None or archive.getinfo(target).file_size > 20 * MAX_IMPORT_BYTES:
            raise ImportError_("The .mxl file has no score inside")
        return archive.read(target)
    except zipfile.BadZipFile as exc:
        raise ImportError_("The file is neither MusicXML nor a readable .mxl archive") from exc


def _harmony_symbol(harmony) -> str | None:
    step = harmony.findtext("root/root-step")
    if not step:
        return "N.C." if harmony.find("kind") is not None and harmony.findtext("kind") == "none" else None
    alter = int(float(harmony.findtext("root/root-alter") or 0))
    kind_el = harmony.find("kind")
    kind = (kind_el.text or "major").strip() if kind_el is not None else "major"
    suffix = _KINDS.get(kind, "")
    if suffix is None:
        return "N.C."
    text = kind_el.get("text") if kind_el is not None else None
    name = step + ("#" * alter if alter > 0 else "b" * -alter)
    symbol = name + (text if text and parse_chord_symbol(name + text) else suffix)
    bass = harmony.findtext("bass/bass-step")
    if bass:
        b_alter = int(float(harmony.findtext("bass/bass-alter") or 0))
        symbol += "/" + bass + ("#" * b_alter if b_alter > 0 else "b" * -b_alter)
    return symbol if parse_chord_symbol(symbol) else name + suffix


def import_musicxml(base: dict, data: bytes, start_bar: int = 1) -> dict:
    """A document from MusicXML: notes, chords and lyrics on the recording's beats."""
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise ImportError_("The MusicXML could not be read") from exc
    if root.tag.endswith("score-timewise"):
        raise ImportError_("Timewise MusicXML is not supported; export it as partwise (the usual kind)")
    parts = root.findall("part")
    if not parts:
        raise ImportError_("The MusicXML has no music in it")

    per_bar, divisions, fifths, minor, bpm = None, 1, None, False, None
    notes: list[tuple[int, int, int, int]] = []  # (start, length, pitch, part) in sixteenths from bar 1
    chords: list[tuple[int, str]] = []
    words: dict[int, list[str]] = {}
    measure_starts: list[int] = []
    for p, part in enumerate(parts):
        position = 0
        ties: dict[tuple[int, int], int] = {}  # (pitch, voice) -> index in notes of the note being tied
        for m, measure in enumerate(part.findall("measure")):
            if p == 0:
                measure_starts.append(position)
            start = position
            cursor = 0
            last_start = 0
            longest = 0
            for el in measure:
                tag = el.tag
                if tag == "attributes":
                    divisions = int(float(el.findtext("divisions") or divisions))
                    beats, kind = el.findtext("time/beats"), el.findtext("time/beat-type")
                    if beats and kind and per_bar is None:
                        if kind != "4" or beats not in ("3", "4"):
                            raise ImportError_(f"The score is in {beats}/{kind}; only 3/4 and 4/4 can be lined up with the recording")
                        per_bar = int(beats)
                    if el.find("key/fifths") is not None and fifths is None:
                        fifths = int(el.findtext("key/fifths"))
                        minor = (el.findtext("key/mode") or "major") == "minor"
                elif tag == "direction":
                    tempo = el.find("sound")
                    if tempo is not None and tempo.get("tempo") and bpm is None:
                        bpm = float(tempo.get("tempo"))
                elif tag == "sound" and el.get("tempo") and bpm is None:
                    bpm = float(el.get("tempo"))
                elif tag == "harmony" and p == 0:
                    symbol = _harmony_symbol(el)
                    offset = int(float(el.findtext("offset") or 0))
                    if symbol:
                        chords.append((start + (cursor + offset) * 4 // divisions, symbol))
                elif tag == "backup":
                    cursor -= int(float(el.findtext("duration") or 0))
                elif tag == "forward":
                    cursor += int(float(el.findtext("duration") or 0))
                elif tag == "note":
                    if el.find("grace") is not None or el.find("cue") is not None:
                        continue
                    duration = int(float(el.findtext("duration") or 0))
                    at = last_start if el.find("chord") is not None else cursor
                    if el.find("chord") is None:
                        last_start = cursor
                        cursor += duration
                    longest = max(longest, cursor)
                    if el.find("rest") is not None or el.find("pitch") is None:
                        continue
                    step = el.findtext("pitch/step")
                    pitch = 12 * (int(el.findtext("pitch/octave")) + 1) + _STEPS[step] + int(float(el.findtext("pitch/alter") or 0))
                    s16 = start + at * 4 // divisions
                    l16 = max(1, duration * 4 // divisions)
                    voice = int(el.findtext("voice") or 1)
                    tie_types = {t.get("type") for t in el.findall("tie")}
                    if "stop" in tie_types and (pitch, voice) in ties:
                        k = ties[(pitch, voice)]
                        s, l, pp, pt = notes[k]
                        notes[k] = (s, s16 + l16 - s, pp, pt)
                        if "start" not in tie_types:
                            del ties[(pitch, voice)]
                    else:
                        notes.append((s16, l16, pitch, p))
                        if "start" in tie_types:
                            ties[(pitch, voice)] = len(notes) - 1
                    if p == 0:
                        text = " ".join(t.text or "" for t in el.findall("lyric/text")[:1]).strip()
                        if text:
                            syllabic = el.findtext("lyric/syllabic") or "single"
                            words.setdefault(s16, []).append(text + ("-" if syllabic in ("begin", "middle") else ""))
            position = start + max(longest, cursor) * 4 // divisions

    per_bar = per_bar or base["beats_per_bar"]
    bar = per_bar * 4
    # A pickup (a short first bar) ends where the first full bar of the recording begins.
    shift = 0
    if len(measure_starts) > 1 and 0 < measure_starts[1] < bar:
        shift = bar - measure_starts[1]
    offset = (max(1, start_bar) - 1) * bar + shift
    last = max([s + l for s, l, _, _ in notes] + [c for c, _ in chords] + [0]) + offset
    n_bars = max(1, -(-last // bar))
    beats = _beats(base, n_bars * per_bar + 1)

    tonic = (fifths * 7 + (9 if minor else 0)) % 12 if fifths is not None else base["key"]["tonic"]
    melody = _top_line([(s + offset, l, pp) for s, l, pp, pt in notes if pt == 0])
    everything = [(s + offset, l, pp) for s, l, pp, _ in notes]
    right = _staff([n for n in everything if n[2] >= 60])
    left = _staff([n for n in everything if n[2] < 60])
    lyrics: dict[str, str] = {}
    for s, parts_ in sorted(words.items()):
        index = (s + offset) // bar
        if 0 <= index < n_bars:
            text = lyrics.get(str(index), "")
            for piece in parts_:
                text = (text[:-1] if text.endswith("-") else (text + " " if text else "")) + piece
            lyrics[str(index)] = text
    doc = {
        "version": 1,
        "title": base["title"],
        "artist": base["artist"],
        "key": {"tonic": tonic, "minor": minor if fifths is not None else base["key"]["minor"]},
        "bpm": round(bpm, 1) if bpm and 20 <= bpm <= 400 else base["bpm"],
        "beats_per_bar": per_bar,
        "beats": beats,
        "chords": [[(c + offset) // 4, symbol] for c, symbol in chords if 0 <= (c + offset) // 4 < n_bars * per_bar],
        "lyrics": {k: v.rstrip("-")[:200] for k, v in lyrics.items()},
        "arrangements": {"piano": [right, left], "lead": [_staff(melody)]},
    }
    return normalize_document(doc)


def _beats(base: dict, count: int) -> list[float]:
    """The recording's beat times, continued at its last pace if the import is longer."""
    beats = list(base["beats"][:count])
    step = (beats[-1] - beats[-2]) if len(beats) > 1 else 0.5
    while len(beats) < count:
        beats.append(round(beats[-1] + step, 3))
    return beats


def _top_line(notes: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    top: dict[int, tuple[int, int, int]] = {}
    for s, l, p in notes:
        if s not in top or p > top[s][2]:
            top[s] = (s, l, p)
    return [top[s] for s in sorted(top)]


def _staff(notes: list[tuple[int, int, int]]) -> list[list]:
    """Notes struck together become one chord; each ends where the next begins."""
    together: dict[int, tuple[int, set[int]]] = {}
    for s, l, p in notes:
        if s < 0 or not 21 <= p <= 108:
            continue
        length, ps = together.get(s, (l, set()))
        together[s] = (max(length, l), ps | {p})
    starts = sorted(together)
    out = []
    for i, s in enumerate(starts):
        length, ps = together[s]
        if i + 1 < len(starts):
            length = min(length, starts[i + 1] - s)
        out.append([s, max(1, length), sorted(ps)[:8]])
    return out


# ---------------------------------------------------------------- chord sheets

_CHORD_TOKEN = re.compile(r"^[A-G][#b♯♭]?[^\s/|\[\]]*(?:/[A-G][#b♯♭]?)?$")
_NO_CHORD = {"n.c.", "nc", "n.c", "x"}


def _is_chord(token: str) -> bool:
    token = token.replace("♯", "#").replace("♭", "b")
    return token.lower() in _NO_CHORD or (bool(_CHORD_TOKEN.match(token)) and parse_chord_symbol(token) is not None)


def _clean(token: str) -> str:
    token = token.replace("♯", "#").replace("♭", "b")
    return "N.C." if token.lower() in _NO_CHORD else token


def parse_chord_sheet(text: str) -> tuple[list[list[str]] | None, list[tuple[str, str]]]:
    """(bars of chords if the sheet has bar lines, else None; [(chord, words sung from it)])."""
    lines = [line.rstrip() for line in text.splitlines()]
    if any("|" in line for line in lines):
        bars: list[list[str]] = []
        for line in lines:
            if "|" not in line:
                continue
            cells = line.strip().strip("|").split("|")
            for cell in cells:
                tokens = cell.split()
                if not tokens or tokens == ["%"]:
                    bars.append(list(bars[-1]) if bars else ["N.C."])
                    continue
                chords = [_clean(t) for t in tokens if _is_chord(t)]
                bars.append(chords or ["N.C."])
        if bars:
            return bars, []
    sequence: list[tuple[str, str]] = []
    pending: list[str] | None = None  # a line of chords waiting for its lyric line
    for line in lines + [""]:
        if "[" in line and re.search(r"\[([^\]]+)\]", line):  # ChordPro: [C]words [G]more words
            pieces = re.split(r"\[([^\]]+)\]", line)
            for chord, words in zip(pieces[1::2], pieces[2::2]):
                if _is_chord(chord.strip()):
                    sequence.append((_clean(chord.strip()), words.strip()))
            continue
        tokens = line.split()
        if tokens and all(_is_chord(t) for t in tokens if t not in ("-", "–")):
            if pending:
                sequence.extend((c, "") for c in pending)
            pending = [_clean(t) for t in tokens if t not in ("-", "–")]
            continue
        if pending is not None:
            # A lyric line under a chord line: its words are shared out among the chords.
            words = line.strip()
            if words and pending:
                share = _split_words(words, len(pending))
                sequence.extend(zip(pending, share))
            else:
                sequence.extend((c, "") for c in pending)
            pending = None
    return None, [(c, w) for c, w in sequence if c]


def _split_words(words: str, parts: int) -> list[str]:
    tokens = words.split()
    out, size = [], max(1, -(-len(tokens) // parts))
    for i in range(parts):
        out.append(" ".join(tokens[i * size : (i + 1) * size]))
    return out


def import_chords(base: dict, text: str, start_bar: int = 1) -> dict:
    """A document with the chords (and words) of a chord sheet, on the recording's beats."""
    bars, sequence = parse_chord_sheet(text)
    per_bar = base["beats_per_bar"]
    n_beats = len(base["beats"]) - 1
    doc = json.loads(json.dumps(base))
    if bars:
        first = (max(1, start_bar) - 1) * per_bar
        chords = []
        for i, cells in enumerate(bars):
            beat0 = first + i * per_bar
            if beat0 >= n_beats:
                break
            step = per_bar / len(cells)
            for k, chord in enumerate(cells):
                chords.append([beat0 + int(round(k * step)), chord])
        doc["chords"] = chords
        return normalize_document(doc)
    if not sequence:
        raise ImportError_("No chords were found in the text")
    starts = align(base, [c for c, _ in sequence], first_beat=(max(1, start_bar) - 1) * per_bar)
    doc["chords"] = [[b, c] for b, (c, _) in zip(starts, sequence)]
    if any(w for _, w in sequence):
        lyrics: dict[str, str] = {}
        for b, (_, words) in zip(starts, sequence):
            if words:
                key = str(b // per_bar)
                lyrics[key] = (lyrics.get(key, "") + " " + words).strip()[:200]
        doc["lyrics"] = lyrics
    return normalize_document(doc)


def _beat_labels(base: dict) -> list:
    """The chord ChordListener heard on every beat: (root, minor) or None."""
    n = len(base["beats"]) - 1
    out = [None] * n
    marks = base["chords"]
    for i, (beat, symbol) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n
        parsed = parse_chord_symbol(symbol)
        value = None
        if parsed is not None:
            intervals = QUALITIES[parsed.chord.quality].intervals
            value = (parsed.chord.root, 3 in intervals and 4 not in intervals)
        for b in range(beat, min(end, n)):
            out[b] = value
    return out


def align(base: dict, sequence: list[str], first_beat: int = 0) -> list[int]:
    """Where each chord of the sequence starts (a beat index), lined up with the chords heard.

    Every chord takes at least one beat, in order; a chord costs nothing on a
    beat where the same chord was heard, a little where only the root agrees,
    and starting a chord off a bar line costs a little too."""
    heard = _beat_labels(base)[first_beat:]
    per_bar = base["beats_per_bar"]
    n, m = len(heard), len(sequence)
    if m > n:
        raise ImportError_("The chord sheet has more chords than the recording has beats; give it bar lines (| C | G |)")
    wanted = []
    for symbol in sequence:
        parsed = parse_chord_symbol(symbol)
        if parsed is None:
            wanted.append(None)
        else:
            intervals = QUALITIES[parsed.chord.quality].intervals
            wanted.append((parsed.chord.root, 3 in intervals and 4 not in intervals))
    cost = np.zeros((m, n))
    for k, w in enumerate(wanted):
        for b, h in enumerate(heard):
            if w is None or h is None:
                cost[k, b] = 0.5
            elif w == h:
                cost[k, b] = 0.0
            elif w[0] == h[0]:
                cost[k, b] = 0.4
            else:
                cost[k, b] = 1.0
    start_cost = np.array([0.0 if (first_beat + b) % per_bar == 0 else (0.15 if (first_beat + b) % per_bar == per_bar // 2 else 0.35) for b in range(n)])
    # D[k, b]: best cost with chord k sounding on beat b (chords 0..k placed, beats 0..b covered).
    inf = 1e18
    D = np.full((m, n), inf)
    came = np.zeros((m, n), dtype=bool)  # True: chord k started on beat b
    D[0, 0] = cost[0, 0]
    for b in range(1, n):
        D[0, b] = D[0, b - 1] + cost[0, b]
    for k in range(1, m):
        for b in range(k, n):
            stay = D[k, b - 1]
            start = D[k - 1, b - 1] + start_cost[b]
            if start < stay:  # on a tie the chord started earlier: a sheet usually follows the song from its start
                D[k, b], came[k, b] = start + cost[k, b], True
            else:
                D[k, b] = stay + cost[k, b]
    starts = [0] * m
    b = n - 1
    for k in range(m - 1, 0, -1):
        while not came[k, b]:
            b -= 1
        starts[k] = b
        b -= 1
    return [first_beat + s for s in starts]


# ---------------------------------------------------------------- another recording of the same song


def fit(version: dict, base: dict, max_shift: int = 16) -> tuple[dict, dict]:
    """Put a version made on one recording onto another recording of the same song.

    Recordings differ in timing (a longer intro, a different tempo) and
    sometimes in key, so the version is moved by whole bars and transposed by
    the amount that makes its chords agree best with what this recording
    sounds like; its notes, chords, lyrics and marks then sit on this
    recording's beats. Returns (document, {"shift_bars", "transpose", "match"}),
    where match is the share of beats whose chords agree (0-1)."""
    per_bar = base["beats_per_bar"]
    if version["beats_per_bar"] != per_bar:
        return version, {"shift_bars": 0, "transpose": 0, "match": None}
    mine = _beat_labels(version)
    heard = _beat_labels(base)
    n_mine, n_heard = len(mine), len(heard)
    labelled = max(1, sum(m is not None for m in mine))
    best = (-1.0, 0, 0, 0.0)
    for shift in range(-max_shift, max_shift + 1):
        for transpose in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, 6):
            agree = total = 0
            for b in range(n_mine):
                target = b + shift * per_bar
                if not 0 <= target < n_heard or mine[b] is None or heard[target] is None:
                    continue
                total += 1
                root, minor = mine[b]
                agree += heard[target] == ((root + transpose) % 12, minor)
            if total >= 2 * per_bar:
                # Chords of the version that land outside the recording count as misses, so a
                # repeating progression is not lined up with half of it hanging off the end;
                # remaining ties go to the smallest change.
                score = agree / labelled - 0.002 * (abs(shift) + abs(transpose))
                if score > best[0]:
                    best = (score, shift, transpose, agree / total)
    score, shift, transpose, match = best
    if score < 0:
        return version, {"shift_bars": 0, "transpose": 0, "match": None}
    return _move(version, base, shift, transpose), {"shift_bars": shift, "transpose": transpose, "match": round(match, 3)}


def _move(version: dict, base: dict, shift: int, transpose: int) -> dict:
    from .theory import Key

    per_bar = base["beats_per_bar"]
    n_beats = len(base["beats"]) - 1
    total = n_beats * 4
    beats_shift, sixteenths = shift * per_bar, shift * per_bar * 4
    key = Key(version["key"]["tonic"], version["key"]["minor"]).transposed(transpose)

    def chord(symbol: str) -> str:
        parsed = parse_chord_symbol(symbol)
        return parsed.chord.transposed(transpose).symbol(key) if parsed and transpose else symbol

    def staff(events):
        out = []
        for start, length, pitches in events:
            s = start + sixteenths
            ps = [p + transpose for p in pitches if 21 <= p + transpose <= 108]
            if 0 <= s < total and ps:
                out.append([s, min(length, total - s), ps])
        return out

    doc = {
        "version": 1,
        "title": version["title"],
        "artist": version["artist"],
        "key": {"tonic": key.tonic, "minor": key.minor},
        "bpm": base["bpm"],
        "beats_per_bar": per_bar,
        "beats": base["beats"],
        "chords": [[b + beats_shift, chord(c)] for b, c in version["chords"] if 0 <= b + beats_shift < n_beats],
        "lyrics": {str(int(k) + shift): v for k, v in version["lyrics"].items() if 0 <= int(k) + shift < n_beats // per_bar},
        "techniques": [[s + sixteenths, p + transpose, kind] for s, p, kind in version.get("techniques", []) if 0 <= s + sixteenths < total and 21 <= p + transpose <= 108],
        "arrangements": {name: [staff(events) for events in staves] for name, staves in version["arrangements"].items()},
    }
    return normalize_document(doc)
