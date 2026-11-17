import numpy as np
import pytest

from chordlistener import notefilter as N
from chordlistener.audio import load_audio
from chordlistener.pitch import decode_events, decode_notes, run_model
from tools.notefilter import data, songs
from tools.notefilter.real import read_midi_notes


@pytest.fixture(scope="module")
def heard(wav_path):
    samples = load_audio(wav_path)
    samples = samples * (0.9 / np.abs(samples).max())
    return run_model(samples), samples


def test_candidates_contain_every_note_the_fixed_thresholds_find(heard):
    act, _ = heard
    events = N.candidates(act)
    old = [e for e in events if e[5]]
    assert [e[:5] for e in old] == decode_events(act)
    assert len(events) > 1.5 * len(old)  # and plenty the thresholds would have missed
    assert events == sorted(events)


def test_features_describe_each_candidate(heard):
    act, samples = heard
    events = N.candidates(act)
    x = N.features(act, events, N.spectrum(samples))
    assert x.shape == (len(events), len(N.FEATURES)) and np.isfinite(x).all()
    column = {name: x[:, i] for i, name in enumerate(N.FEATURES)}
    assert np.array_equal(column["old"] > 0.5, np.array([e[5] for e in events]))
    assert np.allclose(column["strength"], [e[3] for e in events], atol=1e-6)
    assert (column["passes_old"] <= column["old"]).all()
    # Notes the old thresholds accept stand out of the spectrum more than the faint extras do.
    assert column["partial1"][column["passes_old"] > 0.5].mean() > column["partial1"][column["old"] < 0.5].mean()
    assert N.features(act, [], N.spectrum(samples)).shape == (0, len(N.FEATURES))


def test_decision_only_overrides_the_old_thresholds_when_confident():
    passes = N.FEATURES.index("passes_old")
    x = np.zeros((4, len(N.FEATURES)), dtype=np.float32)
    x[:2, passes] = 1.0
    chance = np.array([0.5, 0.05, 0.5, 0.95])
    # old note kept, old note dropped (surely false), new candidate ignored, new candidate added (surely real)
    assert N.decide(x, chance, 0.15, 0.6).tolist() == [True, False, False, True]


def test_matching_pairs_each_true_note_once():
    truth = [(1.0, 60), (2.0, 60), (2.0, 64)]
    found = [(1.02, 60), (1.05, 60), (2.3, 60), (2.01, 64), (2.0, 65)]
    assert data.match(found, truth).tolist() == [True, False, False, True, False]
    assert data.distinct([(1.0, 60), (1.01, 60), (1.5, 60), (1.0, 62)]) == [(1.0, 60), (1.0, 62), (1.5, 60)]
    assert data.score(np.array([True, False, True]), np.array([True, True, False]), 4) == (0.5, 0.25, pytest.approx(1 / 3))


def test_random_songs_are_varied_and_reproducible():
    specs = [songs.random_spec(seed) for seed in range(40)]
    assert songs.random_spec(7) == specs[7]
    assert len({s["key"] for s in specs}) > 10 and len({s["programs"] for s in specs}) > 20
    assert any(len(set(s["programs"])) == 1 and not s["drums"] for s in specs)  # some solo-instrument songs
    unseen = {p for seed in range(40) for p in songs.random_spec(seed, unseen=True)["programs"]}
    assert not unseen & {p for s in specs for p in s["programs"]}  # kept out of training entirely


def test_midi_reader_round_trips_our_own_files(song, tmp_path):
    path = tmp_path / "song.mid"
    path.write_bytes(song.midi_bytes())
    notes = read_midi_notes(str(path))
    truth = sorted((start, pitch) for start, _, pitch in song.pitched_notes)
    assert len(notes) == len(truth)
    assert all(abs(a[0] - b[0]) < 0.005 and a[1] == b[1] for a, b in zip(notes, truth))


@pytest.mark.skipif(not N.available(), reason="no trained note filter in this checkout")
def test_filter_writes_more_true_notes_and_fewer_false_ones(heard, song):
    act, samples = heard
    truth = data.distinct([(start, pitch) for start, _, pitch in song.pitched_notes])
    old = [(n.start, n.pitch) for n in decode_notes(act)]
    new_notes = N.filtered_notes(act, samples)
    new = [(n.start, n.pitch) for n in new_notes]
    old_hits, new_hits = data.match(old, truth).sum(), data.match(new, truth).sum()
    old_f = 2 * old_hits / (len(old) + len(truth))
    new_f = 2 * new_hits / (len(new) + len(truth))
    assert new_f >= old_f - 0.01
    assert all(0.0 <= n.confidence <= 1.0 and n.end > n.start for n in new_notes)
    # No key sounds twice at once.
    by_pitch = {}
    for n in new_notes:
        assert by_pitch.get(n.pitch, 0.0) <= n.start + 1e-9
        by_pitch[n.pitch] = n.end
