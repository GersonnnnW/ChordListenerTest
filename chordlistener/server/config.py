"""Server settings, all from environment variables (see README)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    data_dir: str = field(default_factory=lambda: os.environ.get("CL_DATA_DIR", "./data"))
    # Largest upload accepted, and longest audio analysed (longer files are cut).
    max_upload_mb: int = field(default_factory=lambda: _int("CL_MAX_UPLOAD_MB", 60))
    max_seconds: int = field(default_factory=lambda: _int("CL_MAX_SECONDS", 900))
    # Analyses running at once. Each one already uses several cores.
    workers: int = field(default_factory=lambda: _int("CL_WORKERS", 2))
    # Whether people without an API key (i.e. the website) may analyse audio.
    public_access: bool = field(default_factory=lambda: _bool("CL_PUBLIC_ACCESS", True))
    public_per_hour: int = field(default_factory=lambda: _int("CL_PUBLIC_PER_HOUR", 30))
    # Song lookup by name: "itunes" or "none".
    catalog: str = field(default_factory=lambda: os.environ.get("CL_CATALOG", "itunes"))
    # Browser origins allowed to call the API. "*" suits a keyed public API.
    cors_origins: str = field(default_factory=lambda: os.environ.get("CL_CORS_ORIGINS", "*"))
    # Set when running behind a reverse proxy that sets X-Forwarded-For.
    trust_proxy: bool = field(default_factory=lambda: _bool("CL_TRUST_PROXY", False))
    # How long results are kept before they are purged (0 = keep forever).
    retention_days: int = field(default_factory=lambda: _int("CL_RETENTION_DAYS", 30))
    wait_seconds: int = field(default_factory=lambda: _int("CL_WAIT_SECONDS", 120))

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "chordlistener.db")
