import pytest

from chordlistener.audio import AudioError, load_audio, sniff_format


def test_sniffs_common_containers():
    assert sniff_format(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "wav"
    assert sniff_format(b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00") == "mp3"
    assert sniff_format(b"fLaC\x00\x00\x00\x22\x00\x00\x00\x00") == "flac"
    assert sniff_format(b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00") == "mp4"
    assert sniff_format(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00") == "ogg"


def test_rejects_playlists_and_text(tmp_path):
    # ffmpeg would follow the references in a playlist; it must never get that far.
    playlist = tmp_path / "evil.m3u8"
    playlist.write_text("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:10.0,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n")
    assert sniff_format(playlist.read_bytes()[:16]) is None
    with pytest.raises(AudioError):
        load_audio(str(playlist))


def test_decodes_wav(wav_path, song):
    samples = load_audio(wav_path)
    assert samples.dtype.kind == "f"
    assert abs(len(samples) / 22050 - (song.duration + 1)) < 0.1
    assert len(load_audio(wav_path, max_seconds=2.0)) == 44100
