import numpy as np
import pytest

from chordlistener import analyze_samples
from chordlistener import arrange as A
from chordlistener import guitar
from chordlistener.audio import load_audio, looks_like_video, sniff_format
from chordlistener.document import document_from_analysis
from chordlistener.lyrics import Lyrics, can_place, guess_title_artist, parse_lrc, place
from chordlistener.theory import Chord, parse_chord_symbol
from chordlistener.tuning import estimate_tuning, retune
from tools.evaluate import chord_accuracy


# ---- Tuning ----------------------------------------------------------------


@pytest.mark.parametrize("cents", [0, -35, 40])
def test_tuning_is_measured(wav_path, cents):
    detuned, _ = retune(load_audio(wav_path), -cents)
    measured, confidence = estimate_tuning(detuned)
    assert abs(measured - cents) < 5 and confidence > 0.5


def test_off_pitch_recordings_are_corrected(wav_path, song):
    samples = load_audio(wav_path)
    flat, _ = retune(samples, 40)  # 40 cents flat: nearly a quarter-tone
    result = analyze_samples(flat)
    assert result.tuning_corrected and abs(result.tuning_cents + 40) < 5
    assert result.to_dict()["tuning"]["reference_hz"] == pytest.approx(430, abs=2)
    assert abs(result.duration - len(flat) / 22050) < 0.01  # times are back on the original clock
    stretch = len(flat) / len(samples)

    class Slowed:
        chords = [(a * stretch, b * stretch, root, quality) for a, b, root, quality in song.chords]

    assert chord_accuracy(Slowed, result)[0] > 0.9


def test_in_tune_recordings_are_left_alone(analysis):
    assert not analysis.tuning_corrected and abs(analysis.tuning_cents) < 10


def test_silence_has_no_tuning():
    assert estimate_tuning(np.zeros(22050 * 3, dtype=np.float32)) == (0.0, 0.0)


# ---- Arrangement -----------------------------------------------------------


def test_grid_is_eighths_unless_sixteenths_are_real():
    eighths = np.arange(0, 64, 2) + np.random.default_rng(0).normal(0, 0.12, 32)
    assert A.choose_grid(eighths) == 2
    assert A.choose_grid(np.arange(0, 64, 1.0)) == 1


def test_melody_prefers_a_smooth_line_on_top():
    # A tune stepping around A4 over repeated chord tones, plus one stray very high blip.
    tune = [(i * 2, 2, 69 + step, 0.7) for i, step in enumerate([0, 2, 4, 2, 0, -1, 0, 2])]
    chords = [(i * 4, 4, pitch, 0.6) for i in range(4) for pitch in (57, 60, 64)]
    stray = [(6, 1, 93, 0.4)]
    line = A.find_melody(sorted(tune + chords + stray))
    assert [n[2] for n in line] == [n[2] for n in tune]


def test_melody_rests_when_nothing_is_worth_singing():
    assert A.find_melody([]) == []


def test_right_hand_keeps_the_tune_on_top(analysis):
    doc = document_from_analysis(analysis)
    right, left = doc["arrangements"]["piano"]
    lead = doc["arrangements"]["lead"][0]
    assert lead and right and left
    tops = {start: max(pitches) for start, _, pitches in right}
    on_top = [tops.get(start) == pitches[0] for start, _, pitches in lead if start in tops]
    assert sum(on_top) / len(on_top) > 0.9
    assert all(max(pitches) < 60 for _, _, pitches in left)
    assert all(start % 2 == 0 for start, _, _ in left)  # the bass moves in eighths at most


def test_low_tunes_are_written_an_octave_up():
    low = [(i * 4, 4, 50 + (i % 3), 0.7) for i in range(8)]
    assert A._treble_shift(low) == 12 and A._treble_shift([(0, 4, 72, 0.7)]) == 0


# ---- Guitar ----------------------------------------------------------------


def chords(*symbols):
    return [(parse_chord_symbol(s).chord, 4) for s in symbols]


@pytest.mark.parametrize(
    "sounding, capo, shapes",
    [(("Bbm", "Ab", "Db"), 1, ["Am", "G", "C"]), (("B", "F#", "G#m", "E"), 4, ["G", "D", "Em", "C"]), (("F#m7", "A", "Esus4", "B7sus4"), 2, ["Em7", "G", "Dsus4", "A7sus4"])],
)
def test_capo_suggestions_match_how_songs_are_played(sounding, capo, shapes):
    tip = guitar.suggest(chords(*sounding))
    assert tip["capo"] == capo and tip["transpose"] == -capo
    assert [c.transposed(-capo).symbol() for c, _ in chords(*sounding)] == shapes


def test_easy_chords_need_no_capo():
    assert guitar.suggest(chords("C", "G", "Am", "F")) is None
    assert guitar.suggest(chords("G", "D", "Em", "C")) is None
    assert guitar.suggest([]) is None
    assert guitar.ease(Chord(0, "maj")) > guitar.ease(Chord(1, "maj"))


# ---- Lyrics ----------------------------------------------------------------


def test_parse_lrc():
    lines = parse_lrc("[00:12.50] Hello there\n[ar: Someone]\n[01:02.5][02:02] Twice sung\n\n[00:05] ")
    assert lines == [(12.5, "Hello there"), (62.5, "Twice sung"), (122.0, "Twice sung")]


@pytest.mark.parametrize(
    "filename, expected",
    [("Queen - Bohemian Rhapsody (Official Video).mp4", ("Bohemian Rhapsody", "Queen")), ("03 Let It Be.mp3", ("Let It Be", "")),
     ("Adele_-_Someone_Like_You_[Lyrics].wav", ("Someone Like You", "Adele")), ("holiday.mov", ("holiday", ""))],
)
def test_title_and_artist_from_a_filename(filename, expected):
    assert guess_title_artist("", "", filename) == expected


def test_given_title_and_artist_win():
    assert guess_title_artist("Real Title", "Real Artist", "junk - name.mp3") == ("Real Title", "Real Artist")


def test_lines_land_in_the_bars_where_they_are_sung():
    beats = [i * 0.5 for i in range(4 * 8 + 1)]  # 8 bars of 2 seconds
    bars = place([(0.1, "one two three four"), (6.2, "five six")], beats, 4)
    assert bars["0"].startswith("one") and " ".join(bars[k] for k in sorted(bars, key=int)) == "one two three four five six"
    assert "3" in bars and bars["3"].startswith("five")
    assert place([(None, "untimed")], beats, 4) == {}


def test_timed_lyrics_only_fit_the_matching_recording():
    timed = Lyrics([(1.0, "a"), (2.0, "b")], duration=200.0)
    assert can_place(timed, 201.5) and not can_place(timed, 30.0)
    assert not can_place(Lyrics([(None, "a")], duration=200.0), 200.0)


# ---- Video containers ------------------------------------------------------


def test_video_formats_are_recognised():
    assert sniff_format(b"RIFF\x00\x00\x00\x00AVI LIST") == "avi"
    assert sniff_format(bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")) == "asf"
    assert sniff_format(b"FLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00") == "flv"
    transport = bytearray(600)
    transport[0] = transport[188] = transport[376] = 0x47
    assert sniff_format(bytes(transport)) == "mpegts"
    assert sniff_format(b"Good morning, this is a text file.") is None
    assert looks_like_video("clip.MOV", "", b"") and looks_like_video("x", "video/mp4", b"")
    assert not looks_like_video("song.mp3", "audio/mpeg", b"ID3\x04" + bytes(12))
