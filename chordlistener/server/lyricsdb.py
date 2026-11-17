"""Where lyrics come from.

A lyrics provider answers one question: "what are the words to this song?".
The built-in provider is LRCLIB, a free community database that also carries
line timings. Song lyrics are copyrighted: a commercial service should use a
provider it has a licence with (Musixmatch, LyricFind, ...) by implementing
the same one-method interface.
"""

from __future__ import annotations

import re

import httpx

from ..lyrics import Lyrics, parse_lrc, parse_plain


class LyricsProvider:
    name = "none"

    async def find(self, title: str, artist: str, duration: float | None = None) -> Lyrics | None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


def _simplify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _same_title(wanted: str, found: str) -> bool:
    """Equal, or one is the other plus a little extra ("let it be" / "let it be remastered 2009")."""
    if not wanted or not found:
        return False
    if wanted == found:
        return True
    short, long = sorted((wanted, found), key=len)
    return long.startswith(short + " ") and len(short) >= 0.5 * len(long)


class LrclibProvider(LyricsProvider):
    name = "lrclib"
    _API = "https://lrclib.net/api"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(timeout=6, follow_redirects=False, headers={"User-Agent": "ChordListener/0.1"})

    async def find(self, title: str, artist: str, duration: float | None = None) -> Lyrics | None:
        if not title:
            return None
        params = {"track_name": title}
        if artist:
            params["artist_name"] = artist
        try:
            response = await self._client.get(f"{self._API}/search", params=params)
            response.raise_for_status()
            results = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(results, list):
            return None
        want_title, want_artist = _simplify(title), _simplify(artist)
        best, best_score = None, 0.0
        for item in results[:25]:
            if not isinstance(item, dict) or item.get("instrumental") or not (item.get("syncedLyrics") or item.get("plainLyrics")):
                continue
            name, by = _simplify(str(item.get("trackName", ""))), _simplify(str(item.get("artistName", "")))
            length = item.get("duration")
            close = bool(duration) and isinstance(length, (int, float)) and abs(length - duration) <= 3
            if not _same_title(want_title, name):
                continue
            if want_artist:
                if want_artist not in by and by not in want_artist:
                    continue
            elif not (name == want_title and close):
                # Going by a file name alone, many songs share a title. Only the
                # exact title on a recording of exactly this length is safe;
                # showing the wrong song's words is worse than showing none.
                continue
            score = 1.0 + (1.0 if name == want_title else 0.0) + (0.5 if item.get("syncedLyrics") else 0.0)
            if duration and isinstance(length, (int, float)):
                # The same song exists in many cuts; the one whose length matches is the one with usable timings.
                score += 2.0 if close else (1.0 if abs(length - duration) <= 6 else -0.5)
            if score > best_score:
                best, best_score = item, score
        if best is None:
            return None
        synced = parse_lrc(best.get("syncedLyrics") or "")
        lines = synced or parse_plain(best.get("plainLyrics") or "")
        if not lines:
            return None
        length = best.get("duration")
        return Lyrics(
            lines=[(t, text[:300]) for t, text in lines[:400]],
            source=self.name,
            title=str(best.get("trackName", ""))[:200],
            artist=str(best.get("artistName", ""))[:200],
            duration=float(length) if isinstance(length, (int, float)) else None,
            link="https://lrclib.net",
        )

    async def close(self) -> None:
        await self._client.aclose()


def make_lyrics_provider(name: str) -> LyricsProvider | None:
    if name == "lrclib":
        return LrclibProvider()
    return None
