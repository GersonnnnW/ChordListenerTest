"""Read single files out of a large zip on a web server, without downloading all of it.

zipfile only needs seek() and read(); HTTP range requests provide both.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request


class RemoteFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos = url, block, 0
        for attempt in range(8):
            try:
                request = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(request, timeout=60) as response:
                    self.size = int(response.headers["Content-Length"])
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 7:
                    raise
                time.sleep(min(60, 2**attempt))  # busy servers answer 503 now and then
        self._cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _range(self, start: int, end: int) -> bytes:
        """Bytes start..end, retried with growing pauses: big archives on busy servers fail now and then."""
        for attempt in range(8):
            try:
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                if len(data) == end - start:
                    return data
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 7:
                    raise
            time.sleep(min(60, 2**attempt))
        raise OSError(f"could not read bytes {start}-{end} of {self.url}")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n > 4 * self.block:
            data = self._range(self.pos, self.pos + n)
        else:
            out = bytearray()
            pos = self.pos
            while len(out) < n:
                index = pos // self.block
                if index not in self._cache:
                    if len(self._cache) > 64:
                        self._cache.clear()
                    self._cache[index] = self._range(index * self.block, min(self.size, (index + 1) * self.block))
                chunk = self._cache[index]
                offset = pos - index * self.block
                take = chunk[offset : offset + n - len(out)]
                out += take
                pos += len(take)
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
