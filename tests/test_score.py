import struct
import xml.etree.ElementTree as ET

import pytest

from chordlistener import export
from chordlistener.document import ScoreOptions, build_score
from chordlistener.score import split_duration


@pytest.mark.parametrize("bar", [12, 16])
@pytest.mark.parametrize("rest", [False, True])
def test_split_duration_is_exact_and_notatable(bar, rest):
    allowed = {16, 12, 8, 6, 4, 3, 2, 1}
    for pos in range(bar):
        for length in range(1, bar - pos + 1):
            parts = split_duration(pos, length, bar, rest)
            assert sum(parts) == length
            assert set(parts) <= allowed


def test_common_rhythms_are_written_simply():
    assert split_duration(0, 16, 16) == [16]
    assert split_duration(0, 6, 16) == [6]  # dotted quarter on the beat
    assert split_duration(2, 4, 16) == [4]  # syncopated quarter
    assert split_duration(3, 5, 16) == [1, 4]  # tied over the beat, not a mystery value
    assert split_duration(0, 12, 12) == [12]  # full bar of 3/4


@pytest.mark.parametrize("arrangement, staves", [("piano", 2), ("lead", 1)])
def test_every_bar_is_exactly_full(analysis, arrangement, staves):
    score = build_score(analysis, ScoreOptions(arrangement=arrangement))
    bar = score.beats_per_bar * 4
    assert score.n_staves == staves
    assert len(score.measures) >= 12
    for measure in score.measures:
        for staff in measure.staves:
            position = 0
            for element in staff:
                assert element.start == position
                position += element.length
            assert position == bar


@pytest.mark.parametrize("arrangement", ["piano", "lead"])
def test_musicxml_is_well_formed(analysis, arrangement):
    xml = export.musicxml(analysis, ScoreOptions(arrangement=arrangement, title="A & B <test>", artist="Someone"))
    root = ET.fromstring(xml.split("\n", 2)[2])  # skip the XML declaration and DOCTYPE
    assert root.find("work/work-title").text == "A & B <test>"
    measures = root.findall("part/measure")
    assert len(measures) >= 12
    assert measures[0].find("attributes/key/fifths").text == "0"
    assert measures[0].find("attributes/time/beats").text == "4"
    # Notes minus backups must fill each bar exactly once per staff.
    staves = 2 if arrangement == "piano" else 1
    for measure in measures:
        total = 0
        for child in measure:
            if child.tag == "note" and child.find("chord") is None:
                total += int(child.find("duration").text)
            elif child.tag == "backup":
                total -= int(child.find("duration").text)
        assert total == 16, measure.get("number")
        assert len(measure.findall("backup")) == staves - 1
    symbols = [h.find("root/root-step").text + ("m" if h.find("kind").text == "minor" else "") for h in root.iter("harmony")]
    assert symbols[:4] == ["C", "G", "Am", "F"]


def test_transposed_score_changes_key_signature(analysis):
    xml = export.musicxml(analysis, ScoreOptions(transpose=-2))
    assert "<fifths>-2</fifths>" in xml  # Bb major


def test_midi_file_structure(analysis):
    data = export.midi(analysis)
    assert data[:4] == b"MThd"
    length, fmt, tracks, ppq = struct.unpack(">IHHH", data[4:14])
    assert (length, fmt, tracks, ppq) == (6, 1, 2, 480)
    assert data.count(b"MTrk") == 2
    assert data.endswith(b"\xff\x2f\x00")


def test_chord_chart(analysis):
    text = export.chord_chart(analysis, title="Song", artist="Band")
    lines = text.splitlines()
    assert lines[0] == "Song - Band"
    assert "Key: C major" in lines[1] and "4/4" in lines[1]
    assert [cell.strip() for cell in lines[3].strip("|").split("|")] == ["C", "G", "Am", "F"]
