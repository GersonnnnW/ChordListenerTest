"""Find a song by name and fetch audio for it.

A catalog provider answers two questions: "which tracks match this text?" and
"give me audio for this track". The built-in provider uses Apple's public
iTunes Search API, which exposes a ~30 second preview clip per track. A
business with its own licensed catalogue plugs in by implementing the same
two methods (see docs/API.md, "Bring your own catalogue").
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import httpx


class CatalogError(Exception):
    pass


@dataclass
class Track:
    provider: str
    id: str
    title: str
    artist: str
    album: str = ""
    artwork_url: str = ""
    preview_url: str = ""
    link: str = ""
    duration: float | None = None  # full track length in seconds
    clip_seconds: float | None = None  # how much audio the provider hands over

    def public(self) -> dict:
        return asdict(self)


class CatalogProvider:
    name = "none"

    async def search(self, query: str, limit: int = 8) -> list[Track]:
        raise NotImplementedError

    async def lookup(self, track_id: str) -> Track | None:
        raise NotImplementedError

    async def download(self, track: Track, dest_path: str, max_bytes: int) -> str:
        """Write the track's audio to dest_path; return its SHA-256."""
        raise NotImplementedError


class ITunesProvider(CatalogProvider):
    name = "itunes"
    _API = "https://itunes.apple.com"
    # Audio is only ever fetched from Apple's own hosts, whatever the API returns.
    _AUDIO_HOSTS = (".itunes.apple.com", ".mzstatic.com")

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(timeout=20, follow_redirects=False, headers={"User-Agent": "ChordListener/0.1"})

    @staticmethod
    def _track(item: dict) -> Track | None:
        if item.get("kind") != "song" or not item.get("previewUrl"):
            return None
        millis = item.get("trackTimeMillis")
        return Track(
            provider="itunes",
            id=str(item["trackId"]),
            title=item.get("trackName", ""),
            artist=item.get("artistName", ""),
            album=item.get("collectionName", ""),
            artwork_url=item.get("artworkUrl100", ""),
            preview_url=item["previewUrl"],
            link=item.get("trackViewUrl", ""),
            duration=millis / 1000 if millis else None,
            clip_seconds=30.0,
        )

    async def _get(self, path: str, params: dict) -> list[dict]:
        try:
            response = await self._client.get(self._API + path, params=params)
            response.raise_for_status()
            return response.json().get("results", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise CatalogError("The song catalogue is unavailable right now") from exc

    async def search(self, query: str, limit: int = 8) -> list[Track]:
        items = await self._get("/search", {"term": query, "media": "music", "entity": "song", "limit": limit})
        return [t for t in map(self._track, items) if t]

    async def lookup(self, track_id: str) -> Track | None:
        if not track_id.isdigit():
            return None
        items = await self._get("/lookup", {"id": track_id, "entity": "song"})
        tracks = [t for t in map(self._track, items) if t and t.id == track_id]
        return tracks[0] if tracks else None

    async def download(self, track: Track, dest_path: str, max_bytes: int) -> str:
        url = urlparse(track.preview_url)
        host = "." + (url.hostname or "")
        if url.scheme != "https" or not host.endswith(self._AUDIO_HOSTS):
            raise CatalogError("Preview audio is not available for this track")
        digest = hashlib.sha256()
        size = 0
        try:
            async with self._client.stream("GET", track.preview_url) as response:
                response.raise_for_status()
                with open(dest_path, "wb") as out:
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise CatalogError("Preview audio is too large")
                        digest.update(chunk)
                        out.write(chunk)
        except httpx.HTTPError as exc:
            raise CatalogError("Could not fetch the preview audio") from exc
        return digest.hexdigest()

    async def close(self) -> None:
        await self._client.aclose()


def make_provider(name: str) -> CatalogProvider | None:
    if name == "itunes":
        return ITunesProvider()
    return None
