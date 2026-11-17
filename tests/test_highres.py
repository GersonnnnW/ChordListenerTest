"""High-accuracy mode: large note detectors specialised for one instrument."""

import numpy as np
import pytest

from chordlistener import Analysis, AnalysisOptions, analyze, highres
from tools.evaluate import note_f1

needs_model = pytest.mark.skipif(not highres.available("piano"), reason="piano model not installed")


def test_resample_keeps_a_tone_in_tune():
    t = np.arange(22050) / 22050
    out = highres.resample(np.sin(2 * np.pi * 440 * t).astype(np.float32))
    assert len(out) == 16000
    peak = np.argmax(np.abs(np.fft.rfft(out)))
    assert abs(peak - 440) <= 1  # 1 Hz bins over one second


def _outputs(frames, notes):
    out = {k: np.zeros((frames, 88), np.float32) for k in ("frame", "onset", "offset", "velocity")}
    for start, end, key in notes:
        out["frame"][start:end, key] = 0.9
        out["onset"][start - 3 : start + 4, key] = [0.1, 0.3, 0.6, 0.9, 0.6, 0.3, 0.1]
        out["velocity"][start, key] = 0.7
    return out


def test_decode_reads_attacks_and_releases():
    notes = highres.decode(_outputs(400, [(50, 120, 39), (60, 200, 43), (150, 300, 39)]))
    got = [(round(n.start, 2), round(n.end, 2), n.pitch) for n in notes]
    assert got == [(0.5, 1.2, 60), (0.6, 2.0, 64), (1.5, 3.0, 60)]
    assert all(0 < n.confidence < 1 and n.velocity == pytest.approx(0.7) for n in notes)


def test_a_repeated_key_ends_the_note_before_it():
    notes = highres.decode(_outputs(300, [(50, 250, 40), (100, 250, 40)]))
    assert [(round(n.start, 2), round(n.end, 2)) for n in notes] == [(0.5, 0.99), (1.0, 2.5)]


def test_agreement_between_the_detectors():
    from chordlistener.pitch import Note

    standard = [Note(1.0, 1.5, 60, 0.8, 0.9), Note(2.0, 2.5, 64, 0.8, 0.9), Note(3.0, 3.5, 67, 0.8, 0.2)]
    assert highres.agreement(standard, [Note(1.05, 1.5, 60, 0.7, 0.9), Note(2.0, 2.4, 65, 0.7, 0.9)]) == 0.5
    assert highres.agreement(standard, []) == 0.0
    assert highres.agreement([], standard) == 0.0


def test_silence_has_no_notes():
    assert highres.decode(_outputs(300, [])) == []
    assert highres.decode(_outputs(5, [])) == []


@needs_model
def test_high_accuracy_analysis(song, wav_path, analysis):
    options = AnalysisOptions(accuracy="high")  # the earlier name for piano
    assert options.accuracy == "piano"
    keys = {AnalysisOptions(accuracy=a).cache_key() for a in ("standard", "piano", "guitar")}
    assert len(keys) == 3
    high = analyze(wav_path, options)
    assert high.accuracy == "piano" and analysis.accuracy == "standard" and analysis.note_detector == "standard"
    assert high.note_detector in ("piano", "standard")
    assert note_f1(song, high) > 0.6
    # Beats and chords still come from the standard model.
    assert [c.chord.symbol(high.key) for c in high.chords[:4] if c.chord] == ["C", "G", "Am", "F"]
    again = Analysis.from_record(high.to_record())
    assert again.accuracy == "piano" and again.note_detector == high.note_detector and len(again.notes) == len(high.notes)
    assert high.to_dict()["accuracy"] == "piano" and high.to_dict()["note_detector"] == high.note_detector


@needs_model
def test_piano_recordings_use_the_piano_model(monkeypatch, wav_path):
    # Whatever this recording is, with full agreement required the standard notes are kept, and with none the piano model's.
    monkeypatch.setitem(highres.AGREEMENT, "piano", 1.01)
    assert analyze(wav_path, AnalysisOptions(accuracy="piano")).note_detector == "standard"
    monkeypatch.setitem(highres.AGREEMENT, "piano", 0.0)
    assert analyze(wav_path, AnalysisOptions(accuracy="piano")).note_detector == "piano"


def test_old_records_read_as_piano():
    record = Analysis.from_record({**_minimal_record(), "accuracy": ["high", "piano"]})
    assert record.accuracy == "piano" and record.note_detector == "piano"
    assert Analysis.from_record(_minimal_record()).accuracy == "standard"


def _minimal_record():
    return {"duration": 1.0, "key": [0, False, 0.5], "beats": [0.0, 0.5], "per_bar": 4, "first_downbeat": 0, "chords": [], "notes": []}
