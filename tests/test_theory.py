import pytest

from chordlistener.theory import Chord, Key, note_name, spell_midi, spell_pc


@pytest.mark.parametrize(
    "tonic, minor, fifths, name",
    [(0, False, 0, "C major"), (7, False, 1, "G major"), (5, False, -1, "F major"), (10, False, -2, "Bb major"),
     (6, False, 6, "F# major"), (1, False, -5, "Db major"), (9, True, 0, "A minor"), (4, True, 1, "E minor"),
     (0, True, -3, "C minor"), (6, True, 3, "F# minor")],
)
def test_key_signatures(tonic, minor, fifths, name):
    key = Key(tonic, minor)
    assert key.fifths == fifths
    assert key.name == name


def test_spelling_follows_the_key():
    assert spell_pc(10, Key(5, False)) == "Bb"  # F major
    assert spell_pc(6, Key(7, False)) == "F#"  # G major
    assert spell_pc(8, Key(9, True)) == "G#"  # leading tone of A minor, not Ab
    assert spell_pc(3, Key(0, True)) == "Eb"  # C minor
    assert spell_pc(5, Key(6, False)) == "E#"  # F# major


def test_octaves_across_enharmonic_boundaries():
    assert note_name(60) == "C4"
    assert note_name(69) == "A4"
    assert spell_midi(60, Key(1, False))[2] == 4  # still C4 in Db major
    assert note_name(59, Key(6, False)) == "B3"


def test_chord_symbols_and_transposition():
    key = Key(0, False)
    assert Chord(9, "min").symbol(key) == "Am"
    assert Chord(7, "7").symbol(key) == "G7"
    assert Chord(0, "maj", bass=4).symbol(key) == "C/E"
    assert Chord(9, "min").transposed(3).symbol(Key(3, False)) == "Cm"
    assert sorted(Chord(0, "maj7").pitch_classes()) == [0, 4, 7, 11]
