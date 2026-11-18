"""Versions of a song that people share, and their reviews. Stored as files.

A song is identified by the hash of its audio (the same file, or the same
catalogue clip, gives the same song for everyone). Each song has a folder:

    DATA_DIR/versions/<song>/
        <version id>.json   one shared version: who made it, how, its score document, its votes
        auto.json           the votes on ChordListener's own automatic version

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

AUTO = "auto"
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class VersionError(ValueError):
    """Bad request about versions. The message is safe to show."""


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

    def add(self, song: str, document: dict, author: str, how: str, note: str = "", title: str = "", artist: str = "") -> dict:
        vid = "v_" + secrets.token_urlsafe(9).replace("-", "a").replace("_", "b")
        record = {
            "id": vid,
            "song": song,
            "author": author.strip()[:60] or "Anonymous",
            "how": how,  # musicxml | chords | json | edited
            "note": note.strip()[:300],
            "title": title[:200],
            "artist": artist[:200],
            "created_at": time.time(),
            "votes": {},
            "document": document,
        }
        with self._lock:
            self._write(self._path(song, vid), record)
        return record

    def get(self, song: str, vid: str) -> dict:
        record = self._read(self._path(song, vid)) if vid != AUTO else None
        if record is None:
            raise VersionError("No such version")
        return record

    def list(self, song: str) -> list[dict]:
        folder = self._folder(song)
        out = []
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if name.endswith(".json") and name != AUTO + ".json":
                    record = self._read(os.path.join(folder, name))
                    if record:
                        out.append(record)
        return out

    def summary(self, record: dict) -> dict:
        up, down = tally(record.get("votes", {}))
        return {
            "id": record["id"],
            "author": record.get("author", ""),
            "how": record.get("how", ""),
            "note": record.get("note", ""),
            "created_at": record.get("created_at", 0),
            "votes_up": up,
            "votes_down": down,
            "score": rating(up, down),
        }

    # ------------------------------------------------------------ reviews

    def auto(self, song: str) -> dict:
        return self._read(os.path.join(self._folder(song), AUTO + ".json")) or {"id": AUTO, "song": song, "votes": {}}

    def vote(self, song: str, vid: str, voter: str, value: int) -> dict:
        """Record one person's verdict (+1 accurate, -1 not accurate, 0 take it back); one vote each."""
        if value not in (-1, 0, 1):
            raise VersionError("vote must be up, down or none")
        voter_key = hashlib.sha256(voter.encode()).hexdigest()[:20]
        with self._lock:
            if vid == AUTO:
                path = os.path.join(self._folder(song), AUTO + ".json")
                record = self.auto(song)
            else:
                path = self._path(song, vid)
                record = self._read(path)
                if record is None:
                    raise VersionError("No such version")
            votes = record.setdefault("votes", {})
            if value:
                votes[voter_key] = value
            else:
                votes.pop(voter_key, None)
            self._write(path, record)
        return record

    def voted(self, record: dict, voter: str) -> int:
        return record.get("votes", {}).get(hashlib.sha256(voter.encode()).hexdigest()[:20], 0)


def tally(votes: dict) -> tuple[int, int]:
    up = sum(1 for v in votes.values() if v > 0)
    return up, sum(1 for v in votes.values() if v < 0)


def rating(up: int, down: int) -> float:
    """How sure we can be that a version is accurate: the lower bound of the
    share of positive reviews (Wilson, 90%), so two thumbs up do not outrank
    forty-five out of fifty."""
    n = up + down
    if n == 0:
        return 0.0
    z = 1.645
    p = up / n
    return round((p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / (1 + z * z / n), 4)
