import json

from chordlistener import Analysis
from tools.evaluate import chord_accuracy, note_f1


def test_finds_the_chords(song, analysis):
    root, exact = chord_accuracy(song, analysis)
    assert root > 0.9
    assert exact > 0.85


def test_finds_key_tempo_and_meter(song, analysis):
    assert analysis.key.name == "C major"
    assert analysis.grid.per_bar == 4
    assert abs(analysis.bpm / song.bpm - 1) < 0.05


def test_finds_the_notes(song, analysis):
    assert note_f1(song, analysis) > 0.6


def test_bar_one_starts_with_the_music(song, analysis):
    first = analysis.to_dict()["chords"][0]
    assert first["chord"] == "C"
    assert (first["bar"], first["beat"]) == (1, 1)
    assert abs(first["start"] - song.chords[0][0]) < 0.1


def test_record_round_trip(analysis):
    # The server only ever serves results rebuilt from the stored record.
    record = json.loads(json.dumps(analysis.to_record()))
    restored = Analysis.from_record(record)
    assert restored.to_record() == record
    assert [c["chord"] for c in restored.to_dict()["chords"]] == [c["chord"] for c in analysis.to_dict()["chords"]]
    assert len(restored.notes) == len(analysis.notes)
    assert restored.key == analysis.key and restored.grid.per_bar == analysis.grid.per_bar


def test_transposed_output(analysis):
    plain = analysis.to_dict()
    up = analysis.to_dict(transpose=2)
    assert up["key"]["name"] == "D major"
    assert [c["chord"] for c in up["chords"]][:4] == ["D", "A", "Bm", "G"]
    assert up["notes"][0]["pitch"] == plain["notes"][0]["pitch"] + 2
    assert "notes" not in analysis.to_dict(include_notes=False)


def test_silence_has_no_chords_or_notes():
    import numpy as np

    from chordlistener import analyze_samples

    result = analyze_samples(np.zeros(22050 * 4, dtype=np.float32))
    assert result.notes == []
    assert [c for c in result.chords if c.chord] == []
    assert result.to_dict()["chords"] == []
