import copy
import xml.etree.ElementTree as ET

import pytest

from chordlistener import export
from chordlistener.document import DocumentError, ShowOptions, chord_chart, document_from_analysis, document_view, normalize_document, score_from_document
from chordlistener.theory import Key, parse_chord_symbol


@pytest.fixture()
def doc(analysis):
    return normalize_document(document_from_analysis(analysis, "Song", "Band"))


def parse(xml):
    return ET.fromstring(xml.split("\n", 2)[2])


@pytest.mark.parametrize(
    "text, expected",
    [("C", "C"), ("Am", "Am"), ("F#m7", "F#m7"), ("Bbmaj7/D", "Bbmaj7/D"), ("C-", "Cm"), ("Gsus", "Gsus4"), ("E°", "Edim"),
     ("CM7", "Cmaj7"), ("c#m", "C#m"), ("D/F#", "D/F#"), ("G7sus4", "G7sus4"), ("Cadd9", "Cadd9"), ("E5", "E5"), ("B♭", "Bb")],
)
def test_chord_symbols_people_type(text, expected):
    assert parse_chord_symbol(text).chord.symbol(Key(0, False)) == expected


@pytest.mark.parametrize("text", ["", "H7", "Cfoo", "N.C.", "hello", "C/", "7"])
def test_things_that_are_not_chords(text):
    assert parse_chord_symbol(text) is None


def test_document_from_analysis(doc):
    assert doc["key"] == {"tonic": 0, "minor": False}
    assert doc["beats_per_bar"] == 4
    assert (len(doc["beats"]) - 1) % 4 == 0
    assert [c[1] for c in doc["chords"]][:4] == ["C", "G", "Am", "F"]
    assert len(doc["arrangements"]["piano"]) == 2 and len(doc["arrangements"]["lead"]) == 1
    assert normalize_document(copy.deepcopy(doc)) == doc  # already clean, so unchanged


def test_normalize_tidies_and_bounds_notes(doc):
    staff = doc["arrangements"]["lead"][0]
    staff[:] = [[8, 4, [72, 60, 72]], [0, 100, [64]]]  # out of order, duplicate pitch, overlong
    doc["lyrics"] = {"0": "  hello   world ", "1": "   "}
    doc["chords"].append([1, "  F#m "])
    clean = normalize_document(doc)
    assert clean["arrangements"]["lead"][0] == [[0, 8, [64]], [8, 4, [60, 72]]]
    assert clean["lyrics"] == {"0": "hello world"}
    assert [1, "F#m"] in clean["chords"]


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.update(beats_per_bar=5),
        lambda d: d.update(beats=d["beats"][:-1]),
        lambda d: d.update(beats=[0.0] * len(d["beats"])),
        lambda d: d.update(bpm=5),
        lambda d: d["key"].update(tonic=12),
        lambda d: d["chords"].append([10_000, "C"]),
        lambda d: d["chords"].append([0, "x" * 30]),
        lambda d: d["lyrics"].update({"9999": "la"}),
        lambda d: d["lyrics"].update({"0": "la " * 100}),
        lambda d: d["arrangements"]["lead"][0].append([0, 1, [200]]),
        lambda d: d["arrangements"]["lead"][0].append([0, 1, []]),
        lambda d: d["arrangements"]["lead"][0].append(["0", 1, [60]]),
        lambda d: d["arrangements"]["lead"][0].extend([[2, 1, [60]], [2, 1, [64]]]),
        lambda d: d["arrangements"].pop("piano"),
        lambda d: d.update(title=7),
    ],
)
def test_normalize_rejects_broken_documents(doc, damage):
    damage(doc)
    with pytest.raises(DocumentError):
        normalize_document(doc)


def test_normalize_rejects_non_objects():
    for junk in (None, [], "score", 3):
        with pytest.raises(DocumentError):
            normalize_document(junk)


def test_view_follows_edits_and_transposition(doc):
    doc["chords"][0][1] = "A#m7"
    doc["chords"][1][1] = "play freely"
    view = document_view(doc)
    assert view["chords"][0]["chord"] == "A#m7"  # the spelling someone chose is kept
    assert view["chords"][0]["notes"] and view["chords"][0]["end"] == view["chords"][1]["start"]
    assert view["chords"][1] == {**view["chords"][1], "chord": "play freely", "notes": [], "root": None}
    up = document_view(doc, 2)
    assert up["key"]["name"] == "D major"
    assert up["chords"][0]["chord"] == "Cm7"
    assert up["chords"][1]["chord"] == "play freely"  # not a chord: left alone


def test_what_is_shown(doc):
    doc["lyrics"] = {"0": "one two three four five six seven eight nine ten", "1": "la"}
    full = score_from_document(doc)
    assert full.n_staves == 2 and full.bpm is not None
    assert full.subtitle == f"Key: C major  |  Tempo: {round(doc['bpm'])} BPM"
    sung = [e.lyric for e in full.measures[0].staves[0] if e.lyric]
    assert " ".join(sung) == doc["lyrics"]["0"]  # nothing is lost when words outnumber notes

    sheet = score_from_document(doc, show=ShowOptions(notes=False, tempo=False, key=False))
    assert sheet.n_staves == 1 and sheet.bpm is None and sheet.subtitle == ""
    assert all(e.slash and e.length == 4 for e in sheet.measures[0].staves[0])
    assert sheet.measures[0].harmonies

    bare = score_from_document(doc, show=ShowOptions(chords=False, lyrics=False))
    assert not any(m.harmonies for m in bare.measures)
    assert not any(e.lyric for m in bare.measures for e in m.staves[0])


def test_lyrics_in_a_bar_without_notes_are_still_printed(doc):
    doc["arrangements"]["lead"][0] = [e for e in doc["arrangements"]["lead"][0] if e[0] >= 32]  # two empty bars
    doc["lyrics"] = {"0": "spoken intro"}
    score = score_from_document(normalize_document(doc), "lead")
    assert score.measures[0].words == "spoken intro"
    xml = export.musicxml_from_document(normalize_document(doc), "lead")
    assert "<words>spoken intro</words>" in xml


def test_musicxml_from_an_edited_document(doc):
    doc["title"] = 'Rock & <Roll> "live"'
    doc["lyrics"] = {"0": "Tom & Jerry <3"}
    doc["chords"][0][1] = "Cadd9"
    doc["chords"][1][1] = "N.C."
    doc["chords"][2][1] = "<b>loud</b>"
    root = parse(export.musicxml_from_document(normalize_document(doc)))
    assert root.find("work/work-title").text == 'Rock & <Roll> "live"'
    assert root.find("movement-title").text.startswith("Key: C major")
    lyrics = [t.text for t in root.iter("text")]
    assert lyrics[:2] == ["Tom", "&"]
    first = root.find("part/measure/harmony")
    assert first.find("kind").text == "major" and first.find("degree/degree-value").text == "9"
    words = [w.text for w in root.iter("words")]
    assert "N.C." in words and "<b>loud</b>" in words  # text, never markup
    for measure in root.findall("part/measure"):  # still exactly full bars
        total = sum(int(n.find("duration").text) for n in measure.findall("note") if n.find("chord") is None)
        assert total - sum(int(b.find("duration").text) for b in measure.findall("backup")) == 16


def test_chord_sheet_musicxml(doc):
    root = parse(export.musicxml_from_document(doc, show=ShowOptions(notes=False, tempo=False)))
    first = root.find("part/measure")
    assert [n.find("notehead").text for n in first.findall("note")] == ["slash"] * 4
    assert first.find("attributes/staves") is None
    assert root.find(".//metronome") is None


def test_transposed_chords_are_respelled(doc):
    root = parse(export.musicxml_from_document(doc, transpose=1))  # C major -> Db major
    assert root.find("part/measure/attributes/key/fifths").text == "-5"
    first = root.find("part/measure/harmony/root")
    assert (first.find("root-step").text, first.find("root-alter").text) == ("D", "-1")


def test_chart_with_lyrics(doc):
    doc["lyrics"] = {"0": "first words", "1": "more words", "5": "later"}
    lines = chord_chart(doc).splitlines()
    assert lines[0] == "Song - Band"
    assert lines[3].startswith("| C ") and lines[4] == "  first words / more words"
    assert "  later" in lines
    quiet = chord_chart(doc, show=ShowOptions(lyrics=False, key=False))
    assert "first words" not in quiet and "Key:" not in quiet
