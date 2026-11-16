import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chordlistener import analyze  # noqa: E402
from tools import songgen  # noqa: E402

PROGRESSION = [("C", 4), ("G", 4), ("Am", 4), ("F", 4)]


@pytest.fixture(scope="session")
def song():
    return songgen.make_song(PROGRESSION, bpm=100, key="C", style="piano", repeats=3, seed=7)


@pytest.fixture(scope="session")
def wav_path(song, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("audio") / "song.wav")
    songgen.write_wav(path, songgen.render_numpy(song, seed=7))
    return path


@pytest.fixture(scope="session")
def analysis(wav_path):
    return analyze(wav_path)
