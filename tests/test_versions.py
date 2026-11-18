"""Shared versions: files, reviews, ranking, import formats and export."""

import json

import pytest

from chordlistener import imports
from chordlistener.server.versions import VersionError, VersionStore, song_key, stars_summary
from tools import export_versions


def _base():
    beats = [round(i * 0.5, 3) for i in range(33)]  # 8 bars of 4/4
    return {
        "version": 1, "title": "Song", "artist": "Band", "key": {"tonic": 0, "minor": False}, "bpm": 120.0,
        "beats_per_bar": 4, "beats": beats, "lyrics": {}, "techniques": [],
        "chords": [[0, "C"], [4, "G"], [8, "Am"], [12, "F"], [16, "C"], [20, "G"], [24, "Am"], [28, "F"]],
        "arrangements": {"piano": [[[0, 4, [60, 64, 67]]], []], "lead": [[[0, 4, [67]]]]},
    }


def test_songs_are_matched_by_name():
    same = {song_key(t, a, "h1")[0] for t, a in [
        ("Bohemian Rhapsody (Official Video)", "Queen"),
        ("bohemian rhapsody [Remastered 2011]", "QUEEN"),
        ("Bohemian Rhapsody - Remastered 2011", "Queen"),
        ("Bohemian Rhapsody", "Queen feat. Nobody"),
    ]}
    assert same == {"queen--bohemian-rhapsody"}
    assert song_key("Let It Be", "The Beatles", "h")[0] == song_key("Let it be", "Beatles", "x")[0]
    # Without an artist, only the same audio is the same song.
    assert song_key("audio", "", "abc")[0] == "rec-abc" and song_key("audio", "", "abc")[1] is False


def test_stars_rank_many_ratings_over_few():
    one_five = stars_summary({"a": 5})
    many_fours = stars_summary({str(k): 4 for k in range(20)})
    assert one_five["stars"] == 5 and many_fours["stars"] == 4
    assert many_fours["score"] > one_five["score"] > stars_summary({})["score"] == 3.0


def test_store_keeps_versions_and_one_rating_each(tmp_path):
    store = VersionStore(str(tmp_path))
    record = store.add("abc123", "rec1", _base(), "Ann", "chords", "from the songbook")
    assert [r["id"] for r in store.list("abc123")] == [record["id"]]
    store.rate("abc123", record["id"], "ip:1", 5)
    store.rate("abc123", record["id"], "ip:1", 4)
    store.rate("abc123", record["id"], "ip:2", 2)
    summary = store.summary(store.get("abc123", record["id"]))
    assert (summary["ratings"], summary["stars"]) == (2, 3.0)
    store.rate("abc123", "auto", "ip:3", 1, recording="rec1")
    assert list(store.auto("abc123", "rec1")["ratings"].values()) == [1]
    assert store.auto("abc123", "rec2")["ratings"] == {}  # each recording's automatic version is rated on its own
    with pytest.raises(VersionError):
        store.rate("abc123", record["id"], "ip:4", 6)
    with pytest.raises(VersionError):
        store.get("abc123", "../../etc/passwd")
    with pytest.raises(VersionError):
        store.list("../x")


def test_a_version_fits_another_recording():
    base = _base()
    version = dict(_base(), chords=[[0, "A"], [4, "E"], [8, "F#m"], [12, "D"]], key={"tonic": 9, "minor": False}, lyrics={"0": "hello"})
    version["arrangements"] = {"piano": [[[0, 4, [69]]], []], "lead": [[[0, 4, [69]]]]}
    later = dict(base, chords=[[0, "N.C."], [8, "C"], [12, "G"], [16, "Am"], [20, "F"], [24, "C"], [28, "G"]])
    doc, info = imports.fit(imports.normalize_document(version), imports.normalize_document(later))
    assert info["shift_bars"] == 2 and info["transpose"] == 3 and info["match"] > 0.6
    assert doc["chords"][:4] == [[8, "C"], [12, "G"], [16, "Am"], [20, "F"]]
    assert doc["lyrics"] == {"2": "hello"} and doc["arrangements"]["lead"][0][0] == [32, 4, [72]]
    assert doc["beats"] == later["beats"] and doc["key"] == {"tonic": 0, "minor": False}


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
    good = store.add("band--song", "rec1", _base(), "Ann", "chords")
    store.add("band--song", "rec1", _base(), "Bob", "chords")
    for k in range(4):
        store.rate("band--song", good["id"], f"ip:{k}", 5)
    store.rate("band--song", "auto", "ip:9", 2, recording="rec1")
    rows = list(export_versions.export(str(tmp_path)))
    versions = [r for r in rows if not r.get("automatic")]
    assert [r["author"] for r in versions] == ["Ann"] and versions[0]["recording"] == "rec1"
    assert versions[0]["chords"][0] == [0.0, 2.0, "C"]
    assert any(r.get("automatic") and r["stars"] == 2 for r in rows)
    json.dumps(rows)
