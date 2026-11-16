"""Audio decoding. Everything downstream works on mono float32 at 22.05 kHz."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 22050


class AudioError(ValueError):
    """The input could not be decoded as audio."""


def sniff_format(head: bytes) -> str | None:
    """Identify an audio container from its first bytes.

    Uploads are untrusted, and ffmpeg will happily open playlist-style formats
    that reference other files or URLs. Only real media containers get through.
    """
    if len(head) < 12:
        return None
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:3] == b"ID3":
        return "mp3"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[0] == 0xFF and (head[1] & 0xF6) == 0xF0:
        return "aac"
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"
    return None


def load_audio(path: str, max_seconds: float | None = None) -> np.ndarray:
    """Decode any supported audio file to mono float32 at SAMPLE_RATE."""
    path = os.path.abspath(path)
    with open(path, "rb") as fh:
        head = fh.read(16)
    if sniff_format(head) is None:
        raise AudioError("Unsupported or unrecognised audio format")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-protocol_whitelist", "file", "-i", path, "-vn", "-sn", "-dn"]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Audio decoding timed out") from exc
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if proc.returncode != 0 and samples.size == 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError("Could not decode audio" + (f": {detail[-1][:200]}" if detail else ""))
    if samples.size < SAMPLE_RATE // 2:
        raise AudioError("Audio is too short to analyse (need at least half a second)")
    return np.nan_to_num(samples, copy=True)
