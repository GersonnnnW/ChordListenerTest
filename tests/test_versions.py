"""Shared versions: files, reviews, ranking, import formats and export."""

import json

import pytest

from chordlistener import imports
from chordlistener.server.versions import VersionError, VersionStore, rating
from tools import export_versions


def _base():
    beats = [round(i * 0.5, 3) for i in range(33)]  # 8 bars of 4/4
    return {
        "version": 1, "title": "Song", "artist": "Band", "key": {"tonic": 0, "minor": False}, "bpm": 120.0,
        "beats_per_bar": 4, "beats": beats, "lyrics": {}, "techniques": [],
        "chords": [[0, "C"], [4, "G"], [8, "Am"], [12, "F"], [16, "C"], [20, "G"], [24, "Am"], [28, "F"]],
        "arrangements": {"piano": [[[0, 4, [60, 64, 67]]], []], "lead": [[[0, 4, [67]]]]},
    }


def test_rating_trusts_many_reviews_over_few():
    assert rating(0, 0) == 0
    assert rating(45, 5) > rating(2, 0) > rating(1, 1)
    assert rating(10, 0) > 0.75


def test_store_keeps_versions_and_one_vote_each(tmp_path):
    store = VersionStore(str(tmp_path))
    record = store.add("abc123", _base(), "Ann", "chords", "from the songbook")
    assert [r["id"] for r in store.list("abc123")] == [record["id"]]
    store.vote("abc123", record["id"], "ip:1", 1)
    store.vote("abc123", record["id"], "ip:1", 1)
    store.vote("abc123", record["id"], "ip:2", -1)
    summary = store.summary(store.get("abc123", record["id"]))
    assert (summary["votes_up"], summary["votes_down"]) == (1, 1)
    store.vote("abc123", "auto", "ip:3", -1)
    assert list(store.auto("abc123")["votes"].values()) == [-1]
    with pytest.raises(VersionError):
        store.get("abc123", "../../etc/passwd")
    with pytest.raises(VersionError):
        store.list("../x")


def test_chord_sheets_of_every_kind():
    base = _base()
    assert imports.import_any(base, "| C | G | Am F | % |", start_bar=2)["chords"][:5] == [[4, "C"], [8, "G"], [12, "Am"], [14, "F"], [16, "Am"]]
    assert [c for _, c in imports.import_any(base, "C G Am F")["chords"]] == ["C", "G", "Am", "F"]
    assert imports.import_any(base, "C G Am F")["chords"][:4] == [[0, "C"], [4, "G"], [8, "Am"], [12, "F"]]
    pro = imports.import_any(base, "[C]When I find my[G]self in [Am]times of [F]trouble")
    assert pro["lyrics"]["0"].startswith("When I find")
    over = imports.import_any(base, "C        G\nWhen I find myself\nAm       F\nin times of trouble")
    assert [c for _, c in over["chords"]] == ["C", "G", "Am", "F"]
    with pytest.raises(imports.ImportError_):
        imports.import_any(base, "no chords in here at all")
    with pytest.raises(imports.ImportError_):
        imports.import_any(base, b"PK\x03\x04 not really a zip")


def test_musicxml_with_a_pickup_and_ties():
    xml = """<?xml version="1.0"?><score-partwise version="4.0"><part-list><score-part id="P1"><part-name>x</part-name></score-part></part-list>
    <part id="P1">
      <measure number="0" implicit="yes"><attributes><divisions>1</divisions><key><fifths>1</fifths><mode>major</mode></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
        <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><lyric><syllabic>single</syllabic><text>Oh</text></lyric></note></measure>
      <measure number="1"><harmony><root><root-step>G</root-step></root><kind>major</kind></harmony>
        <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><tie type="start"/><voice>1</voice></note></measure>
      <measure number="2"><harmony><root><root-step>D</root-step></root><kind>dominant</kind></harmony>
        <note><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><tie type="stop"/><voice>1</voice></note>
        <note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note></measure>
    </part></score-partwise>"""
    doc = imports.import_musicxml(_base(), xml.encode())
    assert doc["key"] == {"tonic": 7, "minor": False}
    lead = doc["arrangements"]["lead"][0]
    assert lead[0] == [12, 4, [62]]  # the pickup is the last beat of bar 1
    assert lead[1] == [16, 24, [67]]  # tied over the bar line: one note of six beats
    assert [c for _, c in doc["chords"]] == ["G", "D7"] and doc["chords"][0][0] == 4
    assert doc["lyrics"] == {"0": "Oh"}


def test_export_for_training(tmp_path):
    store = VersionStore(str(tmp_path / "versions"))
    good = store.add("songhash", _base(), "Ann", "chords")
    store.add("songhash", _base(), "Bob", "chords")
    for k in range(6):
        store.vote("songhash", good["id"], f"ip:{k}", 1)
    store.vote("songhash", "auto", "ip:9", -1)
    rows = list(export_versions.export(str(tmp_path), 0.5))
    versions = [r for r in rows if not r.get("automatic")]
    assert [r["author"] for r in versions] == ["Ann"]
    assert versions[0]["chords"][0] == [0.0, 2.0, "C"]
    assert any(r.get("automatic") and r["votes_down"] == 1 for r in rows)
    json.dumps(rows)
