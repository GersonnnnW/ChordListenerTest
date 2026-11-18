"""Versions of a song that people share, and their star ratings. Stored as files.

A song is identified by its name: artist and title, tidied up so that
"Queen - Bohemian Rhapsody (Official Video)" and "queen – bohemian rhapsody
[Remastered 2011]" are the same song. Without both an artist and a title, a
song falls back to the exact audio (its hash), since "audio.mp3" names nothing.
Each song has a folder:

    DATA_DIR/versions/<song>/
        <version id>.json       one shared version: who made it, how, from which recording, its document, its ratings
        auto-<recording>.json   the ratings of ChordListener's automatic version of one recording

Plain files on purpose: easy to inspect, back up and later move into a
database. Writes go through a temporary file and a rename, so a crash never
leaves half a file behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
import unicodedata

AUTO = "auto"
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
PRIOR_STARS, PRIOR_WEIGHT = 3.0, 2.0  # a new version starts as if two people gave it three stars


class VersionError(ValueError):
    """Bad request about versions. The message is safe to show."""


def _tidy(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text)  # (Official Video), [Remastered 2011]
    text = re.sub(r"\b(feat|ft|featuring)\b\.?.*$", " ", text)
    text = re.sub(r"\s[-–]\s.*\b(remaster\w*|live|version|edit|mix|mono|stereo|demo|acoustic)\b.*$", " ", text)
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"^the ", "", text)


def song_key(title: str, artist: str, recording: str) -> tuple[str, bool]:
    """(folder name, whether it is by name). By name needs both an artist and a title."""
    t, a = _tidy(title), _tidy(artist)
    if t and a:
        key = (a.replace(" ", "-")[:40] + "--" + t.replace(" ", "-")[:50]).strip("-")
        if _SAFE.match(key):
            return key, True
    return "rec-" + recording[:64], False


class VersionStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------ files

    def _folder(self, song: str) -> str:
        if not _SAFE.match(song):
            raise VersionError("Unknown song")
        return os.path.join(self.root, song)

    def _path(self, song: str, vid: str) -> str:
        if not _SAFE.match(vid):
            raise VersionError("No such version")
        return os.path.join(self._folder(song), vid + ".json")

    def _read(self, path: str) -> dict | None:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None

    def _write(self, path: str, data: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{secrets.token_hex(4)}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)

    # ------------------------------------------------------------ versions

    def add(self, song: str, recording: str, document: dict, author: str, how: str, note: str = "", title: str = "", artist: str = "") -> dict:
        vid = "v_" + secrets.token_hex(8)
        record = {
            "id": vid,
            "song": song,
            "recording": recording,  # the audio it was made on: its beats are that recording's beats
            "author": author.strip()[:60] or "Anonymous",
            "how": how,  # musicxml | chords | json | edited
            "note": note.strip()[:300],
            "title": title[:200],
            "artist": artist[:200],
            "created_at": time.time(),
            "ratings": {},
            "document": document,
        }
        with self._lock:
            self._write(self._path(song, vid), record)
        return record

    def get(self, song: str, vid: str) -> dict:
        record = None if vid == AUTO else self._read(self._path(song, vid))
        if record is None:
            raise VersionError("No such version")
        return record

    def list(self, song: str) -> list[dict]:
        folder = self._folder(song)
        out = []
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if name.endswith(".json") and name.startswith("v_"):
                    record = self._read(os.path.join(folder, name))
                    if record:
                        out.append(record)
        return out

    def summary(self, record: dict) -> dict:
        return {
            "id": record["id"],
            "author": record.get("author", ""),
            "how": record.get("how", ""),
            "note": record.get("note", ""),
            "created_at": record.get("created_at", 0),
            **stars_summary(record.get("ratings", {})),
        }

    # ------------------------------------------------------------ ratings

    def _auto_path(self, song: str, recording: str) -> str:
        return self._path(song, f"auto-{recording[:64]}")

    def auto(self, song: str, recording: str) -> dict:
        return self._read(self._auto_path(song, recording)) or {"id": AUTO, "song": song, "recording": recording, "ratings": {}}

    def rate(self, song: str, vid: str, voter: str, stars: int, recording: str = "") -> dict:
        """Record one person's rating (1-5 stars, 0 takes it back); one rating each."""
        if stars not in range(6):
            raise VersionError("stars must be 1 to 5, or 0 to take a rating back")
        voter_key = hashlib.sha256(voter.encode()).hexdigest()[:20]
        with self._lock:
            if vid == AUTO:
                path, record = self._auto_path(song, recording), self.auto(song, recording)
            else:
                path = self._path(song, vid)
                record = self._read(path)
                if record is None:
                    raise VersionError("No such version")
            ratings = record.setdefault("ratings", {})
            if stars:
                ratings[voter_key] = stars
            else:
                ratings.pop(voter_key, None)
            self._write(path, record)
        return record

    def mine(self, record: dict, voter: str) -> int:
        return record.get("ratings", {}).get(hashlib.sha256(voter.encode()).hexdigest()[:20], 0)


def stars_summary(ratings: dict) -> dict:
    values = [v for v in ratings.values() if 1 <= v <= 5]
    n = len(values)
    average = round(sum(values) / n, 2) if n else 0.0
    # Ranked by a cautious average: pulled towards three stars until enough people have rated.
    score = round((PRIOR_STARS * PRIOR_WEIGHT + sum(values)) / (PRIOR_WEIGHT + n), 3)
    return {"ratings": n, "stars": average, "score": score}
