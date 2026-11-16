"""Minimal Standard MIDI File writer (format 1)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class MidiTrack:
    notes: list[tuple[float, float, int, int]]  # (start_s, end_s, pitch, velocity 1..127)
    program: int = 0
    channel: int = 0
    name: str = ""
    markers: list[tuple[float, str]] = field(default_factory=list)


def _varlen(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _meta(kind: int, data: bytes) -> bytes:
    return bytes([0xFF, kind]) + _varlen(len(data)) + data


def _chunk(events: list[tuple[int, int, bytes]]) -> bytes:
    events.sort(key=lambda e: (e[0], e[1]))
    body = bytearray()
    now = 0
    for tick, _, data in events:
        body += _varlen(tick - now) + data
        now = tick
    body += _varlen(0) + _meta(0x2F, b"")
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def write_midi(
    tracks: list[MidiTrack],
    bpm: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
    key_fifths: int = 0,
    minor: bool = False,
    ppq: int = 480,
) -> bytes:
    """Serialise tracks. Times are in seconds and keep their original timing."""
    bpm = min(max(bpm, 20.0), 400.0)
    ticks = lambda seconds: max(0, int(round(seconds * bpm / 60.0 * ppq)))  # noqa: E731

    num, den = time_signature
    conductor = [
        (0, 0, _meta(0x51, struct.pack(">I", int(round(60_000_000 / bpm)))[1:])),
        (0, 0, _meta(0x58, bytes([num, den.bit_length() - 1, 24, 8]))),
        (0, 0, _meta(0x59, struct.pack(">bB", max(-7, min(7, key_fifths)), 1 if minor else 0))),
    ]
    chunks = [_chunk(conductor)]
    for track in tracks:
        ch = track.channel & 0x0F
        events: list[tuple[int, int, bytes]] = []
        if track.name:
            events.append((0, 0, _meta(0x03, track.name.encode("utf-8"))))
        events.append((0, 0, bytes([0xC0 | ch, track.program & 0x7F])))
        for when, text in track.markers:
            events.append((ticks(when), 0, _meta(0x06, text.encode("utf-8"))))
        for start, end, pitch, velocity in track.notes:
            t0 = ticks(start)
            t1 = max(t0 + 1, ticks(end))
            pitch = max(0, min(127, int(pitch)))
            events.append((t0, 2, bytes([0x90 | ch, pitch, max(1, min(127, int(velocity)))])))
            events.append((t1, 1, bytes([0x80 | ch, pitch, 0])))
        chunks.append(_chunk(events))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), ppq)
    return header + b"".join(chunks)
