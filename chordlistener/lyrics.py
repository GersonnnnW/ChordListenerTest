"""Putting lyrics under the right bars.

Lyrics come from a provider as lines, ideally each with the time it is sung
(LRC-style "synced" lyrics). Given the beat grid of a recording, timed lines
can be dropped into the bars where they are sung.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_STAMP = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_JUNK = re.compile(
    r"\s*[\(\[][^\)\]]*(official|video|audio|lyric|remaster|live|hd|hq|4k|visuali[sz]er|explicit|version|edit|feat\.?|ft\.?)[^\)\]]*[\)\]]",
    re.IGNORECASE,
)


@dataclass
class Lyrics:
    lines: list[tuple[float | None, str]]  # (seconds from the start of the full track, text)
    source: str = ""
    title: str = ""
    artist: str = ""
    duration: float | None = None  # length of the recording the timings belong to
    link: str = ""

    @property
    def synced(self) -> bool:
        return bool(self.lines) and all(t is not None for t, _ in self.lines)

    def to_record(self) -> dict:
        return {"lines": [[t, text] for t, text in self.lines], "source": self.source, "title": self.title, "artist": self.artist, "duration": self.duration, "link": self.link}

    @classmethod
    def from_record(cls, rec: dict) -> "Lyrics":
        return cls([(t, text) for t, text in rec.get("lines", [])], rec.get("source", ""), rec.get("title", ""), rec.get("artist", ""), rec.get("duration"), rec.get("link", ""))


def parse_lrc(text: str) -> list[tuple[float | None, str]]:
    """Read LRC text ("[01:02.50] words") into (seconds, words) lines."""
    lines: list[tuple[float | None, str]] = []
    for raw in text.splitlines():
        stamps = list(_STAMP.finditer(raw))
        words = _STAMP.sub("", raw).strip()
        if not stamps or not words:
            continue
        for stamp in stamps:  # one line of text can be sung at several times
            minutes, seconds, fraction = stamp.groups()
            lines.append((int(minutes) * 60 + int(seconds) + (float("0." + fraction) if fraction else 0.0), words))
    lines.sort(key=lambda line: line[0])
    return lines


def parse_plain(text: str) -> list[tuple[float | None, str]]:
    return [(None, line.strip()) for line in text.splitlines() if line.strip()]


def guess_title_artist(title: str, artist: str, filename: str = "") -> tuple[str, str]:
    """Work out what to search for from what little an upload tells us.

    "Queen - Bohemian Rhapsody (Official Video).mp4" -> ("Bohemian Rhapsody", "Queen")
    """
    title = (title or re.sub(r"\.[A-Za-z0-9]{2,5}$", "", filename)).replace("_", " ")
    title = _JUNK.sub("", title)
    title = re.sub(r"^\s*\d{1,2}[\s.\-]+(?=\D)", "", title)  # leading track number
    if not artist and re.search(r"\s[-–—]\s", title):
        artist, title = re.split(r"\s[-–—]\s", title, maxsplit=1)
    return " ".join(title.split()), " ".join(artist.split())


def can_place(lyrics: Lyrics, audio_seconds: float, tolerance: float = 4.0) -> bool:
    """Timed lyrics line up with a recording only if it is the same full-length track."""
    return lyrics.synced and lyrics.duration is not None and abs(lyrics.duration - audio_seconds) <= tolerance


def place(lines: list[tuple[float | None, str]], beats: list[float], beats_per_bar: int) -> dict[str, str]:
    """Spread timed lines over the bars they are sung in. Returns {bar index: words}."""
    timed = [(t, text) for t, text in lines if t is not None]
    n_bars = (len(beats) - 1) // beats_per_bar
    if not timed or n_bars < 1:
        return {}
    bar_starts = np.array(beats[:: beats_per_bar][: n_bars + 1])
    bars: dict[int, list[str]] = {}
    for i, (start, text) in enumerate(timed):
        words = text.split()
        if not words:
            continue
        following = timed[i + 1][0] if i + 1 < len(timed) else start + 8.0
        # A line is sung at roughly two words a second and then the singer
        # breathes; it does not stretch all the way to the next line.
        span = min(max(following - start, 0.5), 0.55 * len(words) + 0.5)
        for k, word in enumerate(words):
            when = start + span * k / len(words)
            bar = int(np.searchsorted(bar_starts, when + 0.15, side="right")) - 1
            if 0 <= bar < n_bars:
                bars.setdefault(bar, []).append(word)
    return {str(bar): " ".join(words)[:200] for bar, words in sorted(bars.items())}
