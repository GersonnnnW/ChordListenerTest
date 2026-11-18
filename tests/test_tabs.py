"""Guitar tablature."""

from chordlistener import tabs
from chordlistener.document import document_from_analysis, normalize_document, score_from_document, tab_text
from chordlistener.musicxml import to_musicxml

STANDARD = tabs.TUNINGS["standard"]


def test_every_fingering_is_playable():
    places = tabs.fingerings((64,), STANDARD)
    assert ((5, 0),) in places and ((4, 5),) in places and len(places) == 5
    for fingering in tabs.fingerings((57, 60, 64), STANDARD):
        strings = [s for s, _ in fingering]
        assert strings == sorted(strings) and len(set(strings)) == 3  # higher notes on higher strings
        fretted = [f for _, f in fingering if f > 0]
        assert not fretted or max(fretted) - min(fretted) <= 4


def test_a_scale_moves_the_hand_little():
    # A minor pentatonic up and down.
    run = [57, 60, 62, 64, 67, 69, 72, 74, 76, 74, 72, 69, 67, 64, 62, 60, 57]
    chosen = tabs.place([(i * 0.5, (p,)) for i, p in enumerate(run)], STANDARD)
    frets = [f for (_, f), in chosen]
    assert all(abs(a - b) <= 4 for a, b in zip(frets, frets[1:]))
    assert chosen[0] == chosen[-1]  # the same note the same way, up and down
    for (string, fret), pitch in zip([c[0] for c in chosen], run):
        assert STANDARD[string] + fret == pitch


def test_strummed_bars_are_left_empty():
    bar = 16
    strum = [(i * 4, 4, (40, 47, 52, 56, 59, 64)) for i in range(4)]
    riff = [(bar + i * 2, 2, (p,)) for i, p in enumerate([57, 60, 62, 64, 62, 60, 57, 55])]
    tab = tabs.tab_from_events(strum + riff, 2, bar)
    assert tab.bars[0].strummed and not tab.bars[0].notes
    assert not tab.bars[1].strummed and len(tab.bars[1].notes) == 8
    assert tab.tuning == "standard"


def test_low_d_means_drop_d():
    tab = tabs.tab_from_events([(0, 4, (38,)), (4, 4, (38,)), (8, 4, (45,))], 1, 16)
    assert tab.tuning == "drop_d"
    assert tab.bars[0].notes[0].places == ((0, 0),)


def test_out_of_range_notes_are_moved_by_octaves():
    assert tabs.playable((28, 100), STANDARD) == (40, 76)


def test_text_tab():
    tab = tabs.tab_from_events([(0, 4, (64,)), (4, 4, (67,))], 1, 16)
    text = tabs.tab_text(tab, [(0, "C")], "Song")
    lines = text.splitlines()
    assert lines[0] == "Song" and "C" in lines[2]
    strings = [line for line in lines if line[:2] in ("e|", "B|", "G|", "D|", "A|", "E|")]
    assert len(strings) == 6 and len({len(line) for line in strings}) == 1


def test_guitar_views_of_a_score(analysis):
    doc = normalize_document(document_from_analysis(analysis, "Song"))
    for view in ("guitar", "guitar_melody"):
        xml = to_musicxml(score_from_document(doc, view))
        assert "<sign>TAB</sign>" in xml and "<staff-lines>6</staff-lines>" in xml and "<octave-change>-1</octave-change>" in xml
        assert "<fret>" in xml and "<part-name>Guitar</part-name>" in xml
    assert "e|" in tab_text(doc) and "e|" in tab_text(doc, arrangement="guitar_melody")
