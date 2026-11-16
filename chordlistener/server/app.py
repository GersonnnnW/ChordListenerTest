"""HTTP API and website.

    uvicorn chordlistener.server.app:app

All analysis endpoints live under /v1. The website in ./web is a client of the
same API and needs nothing the API does not expose.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import ENGINE_VERSION, __version__, export
from ..audio import AudioError
from ..pipeline import Analysis, AnalysisOptions, analyze
from ..score import ScoreOptions
from ..theory import VOCABULARIES
from .catalog import CatalogError, CatalogProvider, Track, make_provider
from .config import Settings
from .store import Store

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
PUBLIC = "public"
TEMPO_SCALES = {"auto": 1.0, "half": 0.5, "double": 2.0}


class RateLimiter:
    """Sliding-window counter per caller. In-memory: one server process."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, who: str, limit: int, window: float) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[who]
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                return False, max(1, int(hits[0] + window - now) + 1)
            hits.append(now)
            if len(self._hits) > 50_000:
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]
            return True, 0


class BodyLimit:
    """Reject request bodies over the upload limit before they fill the disk."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            response = JSONResponse({"error": {"code": "file_too_large", "message": self._message()}}, status_code=413)
            return await response(scope, receive, send)
        received = 0

        async def limited():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(413, self._message())
            return message

        await self.app(scope, limited, send)

    def _message(self) -> str:
        return f"The file is larger than the {self.max_bytes // (1024 * 1024)} MB limit"


class Caller:
    def __init__(self, owner: str, key_row=None, ip: str = ""):
        self.owner = owner
        self.key = key_row
        self.ip = ip

    @property
    def is_public(self) -> bool:
        return self.key is None


def create_app(settings: Settings | None = None, catalog: CatalogProvider | None = None) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.db_path)
    store.fail_interrupted()
    if settings.retention_days > 0:
        store.purge(settings.retention_days)
    executor = ThreadPoolExecutor(max_workers=max(1, settings.workers), thread_name_prefix="analysis")
    limiter = RateLimiter()
    provider = catalog if catalog is not None else make_provider(settings.catalog)
    running: dict[tuple[str, str], asyncio.Future] = {}  # identical uploads in flight share one analysis
    tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        executor.shutdown(wait=False, cancel_futures=True)
        close = getattr(provider, "close", None)
        if close:
            await close()
        store.close()

    app = FastAPI(
        title="ChordListener API",
        version=__version__,
        description="Turn audio into chords, notes and sheet music.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.store = store
    app.state.settings = settings
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "X-API-Key", "Content-Type"],
            expose_headers=["Retry-After"],
        )
    app.add_middleware(BodyLimit, max_bytes=settings.max_upload_mb * 1024 * 1024 + 64 * 1024)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        codes = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "not_ready", 413: "file_too_large", 422: "unprocessable", 429: "rate_limited", 502: "catalog_unavailable"}
        body = {"error": {"code": codes.get(exc.status_code, "error"), "message": exc.detail}}
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    # ---- Who is calling ---------------------------------------------------

    def caller(request: Request) -> Caller:
        ip = request.client.host if request.client else "unknown"
        if settings.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for", "")
            ip = forwarded.split(",")[0].strip() or ip
        secret = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            secret = auth[7:].strip()
        if secret:
            row = store.find_key(secret)
            if row is None:
                raise HTTPException(401, "Invalid or revoked API key")
            return Caller(row["id"], row, ip)
        if not settings.public_access:
            raise HTTPException(401, "An API key is required. Send it as 'Authorization: Bearer <key>'.")
        return Caller(PUBLIC, None, ip)

    def charge(who: Caller) -> None:
        """Count one analysis request against the caller's rate limit and quota."""
        if who.is_public:
            ok, retry = limiter.allow("ip:" + who.ip, settings.public_per_hour, 3600)
        else:
            ok, retry = limiter.allow("key:" + who.owner, who.key["per_minute"], 60)
        if not ok:
            raise HTTPException(429, "Too many requests. Please slow down.", headers={"Retry-After": str(retry)})
        if not who.is_public and who.key["monthly_seconds"] is not None:
            if store.usage_since(who.owner, _month_start())["audio_seconds"] >= who.key["monthly_seconds"]:
                raise HTTPException(429, "This API key has used its monthly audio quota.")

    def owned(tid: str, who: Caller):
        row = store.get_transcription(tid)
        # Results of keyed requests are private to that key; the ID alone is
        # the capability for anonymous ones.
        if row is None or (row["owner"] != PUBLIC and row["owner"] != who.owner):
            raise HTTPException(404, "No such transcription")
        return row

    def load(row) -> Analysis:
        if row["status"] != "completed":
            raise HTTPException(409 if row["status"] == "processing" else 422, _error(row)[1] or "The transcription is not ready yet")
        record = store.get_analysis(row["content_hash"], row["options_key"], row["engine_version"])
        if record is None:
            raise HTTPException(404, "The result has expired")
        return Analysis.from_record(record)

    def describe(row, transpose: int = 0, notes: bool = True) -> dict:
        meta = json.loads(row["meta"])
        body = {
            "id": row["id"],
            "status": row["status"],
            "created_at": _iso(row["created_at"]),
            "source": meta,
            "cached": bool(row["cached"]),
        }
        if row["status"] == "failed":
            code, message = _error(row)
            body["error"] = {"code": code, "message": message}
        elif row["status"] == "completed":
            body["result"] = load(row).to_dict(transpose=transpose, include_notes=notes)
            base = f"/v1/transcriptions/{row['id']}"
            body["links"] = {"self": base, "musicxml": base + "/musicxml", "midi": base + "/midi", "chart": base + "/chart"}
        return body

    # ---- Running an analysis ----------------------------------------------

    def run_blocking(path: str, content_hash: str, options: AnalysisOptions) -> float:
        analysis = analyze(path, options)
        store.put_analysis(content_hash, options.cache_key(), ENGINE_VERSION, analysis.to_record(), analysis.duration)
        return analysis.duration

    async def process(tid: str, who: Caller, path: str | None, content_hash: str, options: AnalysisOptions, meta: dict) -> None:
        """Analyse (or reuse a cached result) and mark the transcription finished."""
        key = (content_hash, options.cache_key())
        try:
            cached = store.get_analysis(content_hash, options.cache_key(), ENGINE_VERSION)
            if cached is not None:
                seconds, was_cached = cached["duration"], True
            else:
                if path is None:
                    raise RuntimeError("Audio missing for an uncached analysis")
                future = running.get(key)
                if future is None:
                    future = asyncio.get_running_loop().run_in_executor(executor, run_blocking, path, content_hash, options)
                    running[key] = future
                    future.add_done_callback(lambda _: running.pop(key, None))
                seconds, was_cached = await asyncio.shield(future), False
            meta["audio_seconds"] = round(seconds, 2)
            meta["options"] = {"vocabulary": options.vocabulary, "beats_per_bar": options.beats_per_bar, "tempo": next(k for k, v in TEMPO_SCALES.items() if v == options.tempo_scale)}
            store.finish_transcription(tid, content_hash, was_cached, meta)
            if not who.is_public:
                store.record_usage(who.owner, tid, seconds, was_cached)
        except AudioError as exc:
            store.fail_transcription(tid, f"invalid_audio|{exc}")
        except Exception:  # noqa: BLE001 - never leave a job stuck in "processing"
            store.fail_transcription(tid, "analysis_failed|The analysis failed unexpectedly")
        finally:
            if path:
                try:
                    os.unlink(path)  # audio is never kept after analysis
                except OSError:
                    pass

    async def start(who: Caller, path: str | None, content_hash: str, options: AnalysisOptions, meta: dict, wait: bool, transpose: int = 0) -> Response:
        tid = store.create_transcription(who.owner, meta, options.cache_key(), ENGINE_VERSION)
        task = asyncio.create_task(process(tid, who, path, content_hash, options, meta))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        if wait:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=settings.wait_seconds)
            except asyncio.TimeoutError:
                pass
        row = store.get_transcription(tid)
        status = {"completed": 200, "processing": 202, "failed": 422}[row["status"]]
        return JSONResponse(describe(row, transpose), status_code=status)

    def parse_options(vocabulary: str, beats_per_bar: int | None, tempo: str) -> AnalysisOptions:
        if vocabulary not in VOCABULARIES:
            raise HTTPException(400, f"vocabulary must be one of: {', '.join(VOCABULARIES)}")
        if beats_per_bar not in (None, 3, 4):
            raise HTTPException(400, "beats_per_bar must be 3 or 4")
        if tempo not in TEMPO_SCALES:
            raise HTTPException(400, f"tempo must be one of: {', '.join(TEMPO_SCALES)}")
        return AnalysisOptions(vocabulary=vocabulary, beats_per_bar=beats_per_bar, tempo_scale=TEMPO_SCALES[tempo], max_seconds=settings.max_seconds)

    # ---- Endpoints --------------------------------------------------------

    @app.get("/v1/health", tags=["Service"])
    async def health():
        return {"status": "ok", "version": __version__, "engine_version": ENGINE_VERSION, "catalog": provider.name if provider else None}

    @app.post("/v1/transcriptions", tags=["Transcriptions"], summary="Analyse an audio file or a catalogue track")
    async def create_transcription(
        who: Caller = Depends(caller),
        file: UploadFile | None = File(None, description="Audio file: MP3, WAV, M4A, FLAC, OGG, AIFF or WebM"),
        track_id: str | None = Form(None, description="Catalogue track ID from /v1/songs/search (instead of a file)"),
        vocabulary: str = Form("standard", description="Chord detail: basic (major/minor), standard (adds 7ths, sus, dim, aug) or full"),
        beats_per_bar: int | None = Form(None, description="Force 3 or 4 beats per bar instead of detecting it"),
        tempo: str = Form("auto", description="auto, or half / double to count the beat at half or twice the detected speed"),
        title: str = Form("", max_length=200),
        artist: str = Form("", max_length=200),
        wait: bool = Query(True, description="Hold the request until the result is ready (otherwise poll the returned ID)"),
    ):
        options = parse_options(vocabulary, beats_per_bar, tempo)
        if (file is None) == (track_id is None):
            raise HTTPException(400, "Send exactly one of: file, track_id")
        charge(who)
        fd, path = tempfile.mkstemp(prefix="cl-", suffix=".audio")
        try:
            if file is not None:
                digest = hashlib.sha256()
                with os.fdopen(fd, "wb") as out:
                    while chunk := await file.read(1 << 20):
                        digest.update(chunk)
                        out.write(chunk)
                content_hash = digest.hexdigest()
                meta = {"type": "upload", "filename": (file.filename or "")[:200], "title": title, "artist": artist}
            else:
                os.close(fd)
                if provider is None:
                    raise HTTPException(400, "Song lookup is not enabled on this server")
                try:
                    track = await provider.lookup(track_id)
                    if track is None:
                        raise HTTPException(404, "No such track in the catalogue")
                    content_hash = store.get_track_hash(provider.name, track.id)
                    known = content_hash and store.get_analysis(content_hash, options.cache_key(), ENGINE_VERSION) is not None
                    if not known:
                        content_hash = await provider.download(track, path, 30 * 1024 * 1024)
                        store.put_track_hash(provider.name, track.id, content_hash)
                except CatalogError as exc:
                    raise HTTPException(502, str(exc)) from exc
                meta = {"type": "catalog", **_track_meta(track)}
        except BaseException:
            _unlink(path)
            raise
        return await start(who, path, content_hash, options, meta, wait)

    @app.get("/v1/transcriptions/{tid}", tags=["Transcriptions"], summary="Fetch a transcription")
    async def get_transcription(
        tid: str,
        who: Caller = Depends(caller),
        transpose: int = Query(0, ge=-11, le=11, description="Shift everything by this many semitones"),
        notes: bool = Query(True, description="Include the individual notes (can be large)"),
    ):
        return describe(owned(tid, who), transpose, notes)

    @app.get("/v1/transcriptions/{tid}/musicxml", tags=["Transcriptions"], summary="Sheet music as MusicXML")
    async def get_musicxml(
        tid: str,
        who: Caller = Depends(caller),
        arrangement: str = Query("piano", pattern="^(piano|lead)$", description="piano = two staves with all notes; lead = melody line with chord symbols"),
        transpose: int = Query(0, ge=-11, le=11),
        download: bool = False,
    ):
        row = owned(tid, who)
        meta = json.loads(row["meta"])
        xml = export.musicxml(load(row), ScoreOptions(arrangement=arrangement, transpose=transpose, title=_title(meta), artist=meta.get("artist", "")))
        return Response(xml, media_type="application/vnd.recordare.musicxml+xml", headers=_attachment(meta, f"{arrangement}.musicxml", download))

    @app.get("/v1/transcriptions/{tid}/midi", tags=["Transcriptions"], summary="Detected notes as a MIDI file")
    async def get_midi(tid: str, who: Caller = Depends(caller), transpose: int = Query(0, ge=-11, le=11)):
        row = owned(tid, who)
        return Response(export.midi(load(row), transpose), media_type="audio/midi", headers=_attachment(json.loads(row["meta"]), "mid", True))

    @app.get("/v1/transcriptions/{tid}/chart", tags=["Transcriptions"], summary="Plain-text chord chart")
    async def get_chart(tid: str, who: Caller = Depends(caller), transpose: int = Query(0, ge=-11, le=11), download: bool = False):
        row = owned(tid, who)
        meta = json.loads(row["meta"])
        text = export.chord_chart(load(row), transpose, _title(meta), meta.get("artist", ""))
        return PlainTextResponse(text, headers=_attachment(meta, "chords.txt", download))

    @app.delete("/v1/transcriptions/{tid}", status_code=204, tags=["Transcriptions"], summary="Delete a transcription")
    async def delete_transcription(tid: str, who: Caller = Depends(caller)):
        store.delete_transcription(owned(tid, who)["id"])
        return Response(status_code=204)

    @app.get("/v1/songs/search", tags=["Songs"], summary="Find a song by name")
    async def search_songs(q: str = Query(..., min_length=2, max_length=120), limit: int = Query(8, ge=1, le=25), who: Caller = Depends(caller)):
        if provider is None:
            raise HTTPException(400, "Song lookup is not enabled on this server")
        ok, retry = limiter.allow(("search:ip:" + who.ip) if who.is_public else ("search:key:" + who.owner), 120, 60)
        if not ok:
            raise HTTPException(429, "Too many searches. Please slow down.", headers={"Retry-After": str(retry)})
        try:
            tracks = await provider.search(q, limit)
        except CatalogError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"provider": provider.name, "results": [_track_meta(t) for t in tracks]}

    @app.get("/v1/usage", tags=["Service"], summary="Usage for the calling API key this month")
    async def usage(who: Caller = Depends(caller)):
        if who.is_public:
            raise HTTPException(401, "Usage is tracked per API key")
        return {"key": {"id": who.owner, "name": who.key["name"]}, "period_start": _iso(_month_start()), "monthly_seconds_limit": who.key["monthly_seconds"], **store.usage_since(who.owner, _month_start())}

    # ---- Website ----------------------------------------------------------

    page_headers = {
        "Cache-Control": "no-cache",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        # Scripts only from this server; artwork and preview audio only from the catalogue's CDN.
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self' data:; "
            "img-src 'self' data: https://*.mzstatic.com; media-src 'self' blob: https://*.apple.com https://*.mzstatic.com; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'"
        ),
    }

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"), headers=page_headers)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def _error(row) -> tuple[str, str]:
    code, _, message = (row["error"] or "").partition("|")
    return (code, message) if message else ("analysis_failed", code)


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _month_start() -> float:
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _track_meta(track: Track) -> dict:
    data = track.public()
    data["track_id"] = data.pop("id")
    return data


def _title(meta: dict) -> str:
    return meta.get("title") or os.path.splitext(meta.get("filename", ""))[0]


def _attachment(meta: dict, suffix: str, download: bool) -> dict:
    if not download:
        return {}
    stem = "".join(c if c.isalnum() or c in " -_" else "_" for c in (_title(meta) or "transcription")).strip()[:80] or "transcription"
    return {"Content-Disposition": f'attachment; filename="{stem}.{suffix}"'}


def __getattr__(name: str):
    # `uvicorn chordlistener.server.app:app` builds the app on first access, so
    # importing this module (e.g. in tests) does not open the database.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
