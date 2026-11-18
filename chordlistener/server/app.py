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
import secrets
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import ENGINE_VERSION, __version__, export
from ..audio import SNIFF_BYTES, AudioError, looks_like_video, to_mp3
from ..lyrics import Lyrics, can_place, guess_title_artist, place
from .. import highres
from ..pipeline import Analysis, AnalysisOptions, analyze
from ..document import VIEWS, DocumentError, ShowOptions, document_from_analysis, document_view, normalize_document
from ..document import tab_text as document_tab
from ..document import chord_chart as document_chart
from ..theory import VOCABULARIES
from .catalog import CatalogError, CatalogProvider, Track, make_provider
from .config import Settings
from .lyricsdb import LyricsProvider, make_lyrics_provider
from .store import Store, hash_key

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
PUBLIC = "public"
TEMPO_SCALES = {"auto": 1.0, "half": 0.5, "double": 2.0}
ACCURACIES = ("standard", "piano", "guitar", "high")  # "high" = piano, its earlier name
MAX_SCORE_BYTES = 4 * 1024 * 1024


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


def create_app(settings: Settings | None = None, catalog: CatalogProvider | None = None, lyrics: LyricsProvider | None = None) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.db_path)
    store.fail_interrupted()
    if settings.retention_days > 0:
        store.purge(settings.retention_days)
    executor = ThreadPoolExecutor(max_workers=max(1, settings.workers), thread_name_prefix="analysis")
    limiter = RateLimiter()
    provider = catalog if catalog is not None else make_provider(settings.catalog)
    lyrics_provider = lyrics if lyrics is not None else make_lyrics_provider(settings.lyrics)
    audio_dir = os.path.join(settings.data_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    def audio_path(tid: str) -> str:
        return os.path.join(audio_dir, tid.replace("/", "_").replace(".", "_") + ".mp3")

    def sweep_audio() -> None:
        """Delete converted soundtracks that have outlived their welcome."""
        limit = time.time() - settings.keep_audio_minutes * 60
        for name in os.listdir(audio_dir):
            full = os.path.join(audio_dir, name)
            try:
                if os.path.getmtime(full) < limit:
                    os.unlink(full)
            except OSError:
                pass

    sweep_audio()
    running: dict[tuple[str, str], asyncio.Future] = {}  # identical uploads in flight share one analysis
    tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        executor.shutdown(wait=False, cancel_futures=True)
        for closing in (provider, lyrics_provider):
            close = getattr(closing, "close", None)
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
            body["links"] = {"self": base, "score": base + "/score", "musicxml": base + "/musicxml", "midi": base + "/midi", "chart": base + "/chart", "tab": base + "/tab", "lyrics": base + "/lyrics"}
            if row["audio_until"] and row["audio_until"] > time.time() and os.path.exists(audio_path(row["id"])):
                body["links"]["audio"] = base + "/audio"
                body["audio_expires_at"] = _iso(row["audio_until"])
        return body

    # ---- Lyrics -----------------------------------------------------------

    def stored_lyrics(row) -> Lyrics | None:
        return Lyrics.from_record(json.loads(row["lyrics"])) if row["lyrics"] else None

    def lyrics_fit(row, found: Lyrics | None) -> bool:
        """Timed lyrics can go under the bars only when the audio is the whole track they were timed for."""
        seconds = json.loads(row["meta"]).get("audio_seconds") or 0
        return found is not None and can_place(found, seconds)

    def describe_lyrics(row, found: Lyrics | None, doc: dict | None = None) -> dict:
        if found is None:
            return {"found": False, "provider": lyrics_provider.name if lyrics_provider else None}
        aligned = lyrics_fit(row, found)
        out = {
            "found": True,
            "source": found.source,
            "link": found.link,
            "title": found.title,
            "artist": found.artist,
            "synced": found.synced,
            "aligned": aligned,
            "lines": [{"time": t, "text": text} for t, text in found.lines],
        }
        if aligned and doc is not None:
            out["bars"] = place(found.lines, doc["beats"], doc["beats_per_bar"])
        return out

    async def look_up_lyrics(meta: dict, seconds: float | None) -> Lyrics | None:
        if lyrics_provider is None:
            return None
        title, artist = guess_title_artist(meta.get("title", ""), meta.get("artist", ""), meta.get("filename", ""))
        if not title:
            return None
        # For a catalogue track the timings belong to the full song, not the excerpt we were given.
        length = meta.get("duration") if meta.get("type") == "catalog" else seconds
        try:
            return await asyncio.wait_for(lyrics_provider.find(title, artist, length), timeout=8)
        except Exception:  # noqa: BLE001 - lyrics are a bonus; never fail a transcription over them
            return None

    # ---- The editable score -----------------------------------------------

    def current_document(row, original: bool = False) -> tuple[dict, bool]:
        """The score as it stands: the saved corrections if there are any, else fresh from the analysis."""
        saved = None if original else store.get_score(row["id"])
        if saved is not None:
            return saved, True
        meta = json.loads(row["meta"])
        doc = document_from_analysis(load(row), _title(meta)[:200], meta.get("artist", "")[:200])
        found = stored_lyrics(row)
        if lyrics_fit(row, found):
            doc["lyrics"] = place(found.lines, doc["beats"], doc["beats_per_bar"])
        return normalize_document(doc), False

    def may_save(row, who: Caller, request: Request) -> bool:
        if row["owner"] != PUBLIC:
            return True  # `owned` has already checked the API key
        # Anyone with the link can view a keyless transcription, but only the
        # browser that created it holds the token to change it.
        token = request.headers.get("x-edit-token", "")
        return bool(token) and bool(row["edit_hash"]) and secrets.compare_digest(hash_key(token), row["edit_hash"])

    async def read_json(request: Request) -> dict:
        body = await request.body()
        if len(body) > MAX_SCORE_BYTES:
            raise HTTPException(413, "The score is too large")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise HTTPException(400, "The request body is not valid JSON") from exc

    def show_options(notes: bool, chords: bool, lyrics: bool, tempo: bool, key: bool, capo: int = 0) -> ShowOptions:
        return ShowOptions(notes=notes, chords=chords, lyrics=lyrics, tempo=tempo, key=key, capo=capo)

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
            meta["options"] = {"vocabulary": options.vocabulary, "beats_per_bar": options.beats_per_bar, "tempo": next(k for k, v in TEMPO_SCALES.items() if v == options.tempo_scale), "accuracy": options.accuracy}
            if meta.get("video") and path and settings.keep_audio_minutes > 0:
                # Browsers cannot play every video format, so hand back an MP3 of the soundtrack to play along with.
                try:
                    await asyncio.get_running_loop().run_in_executor(executor, to_mp3, path, audio_path(tid), settings.max_seconds)
                    store.set_audio_until(tid, time.time() + settings.keep_audio_minutes * 60)
                except (AudioError, OSError):
                    pass
            found = await look_up_lyrics(meta, seconds)
            store.set_lyrics(tid, found.to_record() if found else None)
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
                    os.unlink(path)  # the upload itself is never kept after analysis
                except OSError:
                    pass

    async def start(who: Caller, path: str | None, content_hash: str, options: AnalysisOptions, meta: dict, wait: bool, transpose: int = 0) -> Response:
        tid, edit_token = store.create_transcription(who.owner, meta, options.cache_key(), ENGINE_VERSION)
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
        body = describe(row, transpose)
        if who.is_public:
            body["edit_token"] = edit_token  # shown once; needed to save corrections without an API key
        return JSONResponse(body, status_code=status)

    specialists = highres.installed() if settings.high_accuracy else []

    def parse_options(vocabulary: str, beats_per_bar: int | None, tempo: str, accuracy: str = "standard") -> AnalysisOptions:
        if accuracy not in ACCURACIES:
            raise HTTPException(400, f"accuracy must be one of: {', '.join(ACCURACIES)}")
        if accuracy == "high":
            accuracy = "piano"
        if accuracy != "standard" and accuracy not in specialists:
            raise HTTPException(400, f"High-accuracy mode for {accuracy} is not available on this server")
        if vocabulary not in VOCABULARIES:
            raise HTTPException(400, f"vocabulary must be one of: {', '.join(VOCABULARIES)}")
        if beats_per_bar not in (None, 3, 4):
            raise HTTPException(400, "beats_per_bar must be 3 or 4")
        if tempo not in TEMPO_SCALES:
            raise HTTPException(400, f"tempo must be one of: {', '.join(TEMPO_SCALES)}")
        return AnalysisOptions(vocabulary=vocabulary, beats_per_bar=beats_per_bar, tempo_scale=TEMPO_SCALES[tempo], max_seconds=settings.max_seconds, accuracy=accuracy)

    async def save_upload(file: UploadFile, path: str, fd: int) -> tuple[str, bytes]:
        """Write an upload to `path`; return its SHA-256 and first bytes (to sniff the format)."""
        digest = hashlib.sha256()
        head = b""
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1 << 20):
                if len(head) < SNIFF_BYTES:
                    head += chunk[: SNIFF_BYTES - len(head)]
                digest.update(chunk)
                out.write(chunk)
        return digest.hexdigest(), head

    async def fetch_track(track_id: str, path: str, options: AnalysisOptions) -> tuple[str, Track]:
        """Download a catalogue track to `path` unless its result is already cached; return its hash."""
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
        return content_hash, track

    # ---- Endpoints --------------------------------------------------------

    @app.get("/v1/health", tags=["Service"])
    async def health():
        return {
            "status": "ok",
            "version": __version__,
            "engine_version": ENGINE_VERSION,
            "catalog": provider.name if provider else None,
            "lyrics": lyrics_provider.name if lyrics_provider else None,
            "max_upload_mb": settings.max_upload_mb,
            "high_accuracy": specialists,
        }

    @app.post("/v1/transcriptions", tags=["Transcriptions"], summary="Analyse an audio file or a catalogue track")
    async def create_transcription(
        who: Caller = Depends(caller),
        file: UploadFile | None = File(None, description="Audio (MP3, WAV, M4A, FLAC, OGG, AIFF) or video (MP4, MOV, MKV, WebM, AVI, WMV, FLV, MPEG): a video's soundtrack is used"),
        track_id: str | None = Form(None, description="Catalogue track ID from /v1/songs/search (instead of a file)"),
        vocabulary: str = Form("standard", description="Chord detail: basic (major/minor), standard (adds 7ths, sus, dim, aug) or full"),
        beats_per_bar: int | None = Form(None, description="Force 3 or 4 beats per bar instead of detecting it"),
        tempo: str = Form("auto", description="auto, or half / double to count the beat at half or twice the detected speed"),
        accuracy: str = Form("standard", description="standard; or piano / guitar: high-accuracy mode, where a large detector specialised for that instrument writes the notes when the recording is that instrument. Takes about ten times longer. (high = piano)"),
        title: str = Form("", max_length=200),
        artist: str = Form("", max_length=200),
        wait: bool = Query(True, description="Hold the request until the result is ready (otherwise poll the returned ID)"),
    ):
        options = parse_options(vocabulary, beats_per_bar, tempo, accuracy)
        if (file is None) == (track_id is None):
            raise HTTPException(400, "Send exactly one of: file, track_id")
        charge(who)
        fd, path = tempfile.mkstemp(prefix="cl-", suffix=".audio")
        try:
            if file is not None:
                content_hash, head = await save_upload(file, path, fd)
                filename = (file.filename or "")[:200]
                # "Queen - Bohemian Rhapsody (Official Video).mp4" -> title and artist
                clean_title, clean_artist = guess_title_artist(title, artist, filename)
                meta = {"type": "upload", "filename": filename, "title": clean_title[:200], "artist": clean_artist[:200]}
                if looks_like_video(filename, file.content_type or "", head):
                    meta["video"] = True
                sweep_audio()
            else:
                os.close(fd)
                content_hash, track = await fetch_track(track_id, path, options)
                meta = {"type": "catalog", **_track_meta(track)}
        except BaseException:
            _unlink(path)
            raise
        return await start(who, path, content_hash, options, meta, wait)

    @app.post("/v1/transcriptions/{tid}/reanalyze", tags=["Transcriptions"], summary="Analyse the same song again, e.g. in high-accuracy mode")
    async def reanalyze(
        tid: str,
        who: Caller = Depends(caller),
        accuracy: str = Form("piano", description="standard, piano or guitar (high = piano)"),
        file: UploadFile | None = File(None, description="The same file again. Needed only for an uploaded file whose result in this mode is not cached: uploads are never kept"),
        wait: bool = Query(True, description="Hold the request until the result is ready (otherwise poll the returned ID)"),
    ):
        """Starts a new transcription of the same audio with the same settings, in the accuracy mode asked for.

        A song found by name is fetched again from the catalogue. An uploaded file has to be sent
        again unless this mode's result is already cached. The new transcription has its own ID
        (and, without an API key, its own edit token); the original is left as it was.
        """
        row = owned(tid, who)
        if row["status"] != "completed":
            raise HTTPException(409, "The transcription is not finished yet")
        meta = json.loads(row["meta"])
        previous = meta.get("options") or {}
        options = parse_options(previous.get("vocabulary", "standard"), previous.get("beats_per_bar"), previous.get("tempo", "auto"), accuracy)
        meta = {k: v for k, v in meta.items() if k not in ("options", "audio_seconds")}
        content_hash = row["content_hash"]
        cached = store.get_analysis(content_hash, options.cache_key(), ENGINE_VERSION) is not None
        path = None
        if not cached and file is None and meta.get("type") != "catalog":
            raise HTTPException(400, "Uploaded files are not kept, so send the same file again to analyse it in this mode")
        charge(who)
        if not cached:
            fd, path = tempfile.mkstemp(prefix="cl-", suffix=".audio")
            try:
                if file is not None:
                    sent_hash, _ = await save_upload(file, path, fd)
                    if sent_hash != content_hash:
                        raise HTTPException(400, "That is not the same file as the original transcription")
                else:
                    os.close(fd)
                    content_hash, _ = await fetch_track(meta["track_id"], path, options)
            except BaseException:
                _unlink(path)
                raise
            if store.get_analysis(content_hash, options.cache_key(), ENGINE_VERSION) is not None:
                _unlink(path)
                path = None
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
        arrangement: str = Query("piano", pattern="^(piano|lead|guitar|guitar_melody)$", description="piano = two staves with all notes; lead = melody line with chord symbols; guitar = guitar notation with tablature of all the notes (strummed bars show chord shapes); guitar_melody = the same for the melody line"),
        transpose: int = Query(0, ge=-11, le=11),
        notes: bool = Query(True, description="false = chord sheet: rhythm slashes instead of notes"),
        chords: bool = Query(True, description="Print chord symbols"),
        lyrics: bool = Query(True, description="Print lyrics, if any have been added"),
        tempo: bool = Query(True, description="Print the tempo"),
        key: bool = Query(True, description="Print the key as text (the key signature is always there)"),
        capo: int = Query(0, ge=0, le=11, description="Print 'Capo N' in the heading (use with a negative transpose)"),
        original: bool = Query(False, description="Ignore saved corrections"),
        download: bool = False,
    ):
        row = owned(tid, who)
        doc, _ = current_document(row, original)
        xml = export.musicxml_from_document(doc, arrangement, transpose, show_options(notes, chords, lyrics, tempo, key, capo))
        return Response(xml, media_type="application/vnd.recordare.musicxml+xml", headers=_attachment(json.loads(row["meta"]), f"{arrangement}.musicxml", download))

    @app.get("/v1/transcriptions/{tid}/tab", tags=["Transcriptions"], summary="Guitar tablature as plain text")
    async def get_tab(
        tid: str,
        who: Caller = Depends(caller),
        source: str = Query("notes", pattern="^(notes|melody)$", description="notes = everything played (best for a guitar recording); melody = the tune only"),
        transpose: int = Query(0, ge=-11, le=11),
        chords: bool = Query(True, description="Print chord symbols above the tab"),
        capo: int = Query(0, ge=0, le=11, description="Print 'Capo N' (use with a negative transpose)"),
        original: bool = Query(False, description="Ignore saved corrections"),
        download: bool = False,
    ):
        """Each note's string and fret, chosen as a guitarist would play it. Strummed bars show the chord shape to hold; hammer-ons, pull-offs and slides are marked where the recording shows them."""
        row = owned(tid, who)
        doc, _ = current_document(row, original)
        text = document_tab(doc, transpose, "guitar_melody" if source == "melody" else "guitar", show_options(True, chords, False, False, False, capo))
        return PlainTextResponse(text, headers=_attachment(json.loads(row["meta"]), "tab.txt", download))

    @app.get("/v1/transcriptions/{tid}/midi", tags=["Transcriptions"], summary="Detected notes as a MIDI file")
    async def get_midi(tid: str, who: Caller = Depends(caller), transpose: int = Query(0, ge=-11, le=11)):
        row = owned(tid, who)
        return Response(export.midi(load(row), transpose), media_type="audio/midi", headers=_attachment(json.loads(row["meta"]), "mid", True))

    @app.get("/v1/transcriptions/{tid}/chart", tags=["Transcriptions"], summary="Plain-text chord chart")
    async def get_chart(
        tid: str,
        who: Caller = Depends(caller),
        transpose: int = Query(0, ge=-11, le=11),
        chords: bool = True,
        lyrics: bool = True,
        tempo: bool = True,
        key: bool = True,
        capo: int = Query(0, ge=0, le=11),
        original: bool = Query(False, description="Ignore saved corrections"),
        download: bool = False,
    ):
        row = owned(tid, who)
        doc, _ = current_document(row, original)
        text = document_chart(doc, transpose, show=show_options(True, chords, lyrics, tempo, key, capo))
        return PlainTextResponse(text, headers=_attachment(json.loads(row["meta"]), "chords.txt", download))

    @app.get("/v1/transcriptions/{tid}/score", tags=["Scores"], summary="The editable score")
    async def get_score(tid: str, request: Request, who: Caller = Depends(caller), transpose: int = Query(0, ge=-11, le=11), original: bool = False):
        row = owned(tid, who)
        doc, edited = current_document(row, original)
        return {"document": doc, "edited": edited, "can_save": may_save(row, who, request), "view": document_view(doc, transpose)}

    @app.put("/v1/transcriptions/{tid}/score", tags=["Scores"], summary="Save a corrected score")
    async def put_score(tid: str, request: Request, who: Caller = Depends(caller)):
        row = owned(tid, who)
        load(row)  # only finished transcriptions have a score
        if not may_save(row, who, request):
            raise HTTPException(403, "Only the person who created this transcription can save changes to it")
        try:
            doc = normalize_document(await read_json(request))
        except DocumentError as exc:
            raise HTTPException(400, str(exc)) from exc
        store.put_score(row["id"], doc)
        return {"document": doc, "edited": True, "can_save": True, "view": document_view(doc)}

    @app.delete("/v1/transcriptions/{tid}/score", status_code=204, tags=["Scores"], summary="Discard corrections and go back to the automatic transcription")
    async def delete_score(tid: str, request: Request, who: Caller = Depends(caller)):
        row = owned(tid, who)
        if not may_save(row, who, request):
            raise HTTPException(403, "Only the person who created this transcription can change it")
        store.delete_score(row["id"])
        return Response(status_code=204)

    @app.get("/v1/transcriptions/{tid}/lyrics", tags=["Lyrics"], summary="Lyrics found for this song")
    async def get_lyrics(tid: str, who: Caller = Depends(caller)):
        row = owned(tid, who)
        return describe_lyrics(row, stored_lyrics(row), current_document(row, original=True)[0])

    @app.post("/v1/transcriptions/{tid}/lyrics", tags=["Lyrics"], summary="Look lyrics up again under a corrected title or artist")
    async def find_lyrics(tid: str, request: Request, who: Caller = Depends(caller), title: str = Form(..., min_length=1, max_length=200), artist: str = Form("", max_length=200)):
        row = owned(tid, who)
        load(row)
        ok, retry = limiter.allow(("lyrics:ip:" + who.ip) if who.is_public else ("lyrics:key:" + who.owner), 30, 60)
        if not ok:
            raise HTTPException(429, "Too many lyric searches. Please slow down.", headers={"Retry-After": str(retry)})
        meta = {**json.loads(row["meta"]), "title": title, "artist": artist, "filename": ""}
        found = await look_up_lyrics(meta, meta.get("audio_seconds"))
        if found is not None and may_save(row, who, request):
            store.set_lyrics(row["id"], found.to_record())
        return describe_lyrics(row, found, current_document(row, original=True)[0])

    @app.get("/v1/transcriptions/{tid}/audio", tags=["Transcriptions"], summary="MP3 of an uploaded video's soundtrack (kept for a limited time)")
    async def get_audio(tid: str, request: Request, who: Caller = Depends(caller), token: str = Query("", description="Edit token, for players that cannot send headers")):
        row = owned(tid, who)
        allowed = may_save(row, who, request) or (bool(token) and bool(row["edit_hash"]) and secrets.compare_digest(hash_key(token), row["edit_hash"]))
        if not allowed:
            raise HTTPException(403, "Only the person who uploaded this can fetch its audio")
        path = audio_path(row["id"])
        if not row["audio_until"] or row["audio_until"] < time.time() or not os.path.exists(path):
            raise HTTPException(404, "The converted audio is no longer available")
        return FileResponse(path, media_type="audio/mpeg", filename=(_title(json.loads(row["meta"])) or "audio") + ".mp3", headers={"Cache-Control": "private, max-age=600"})

    @app.post("/v1/score/render", tags=["Scores"], summary="Render a score document without saving it")
    async def render_score(request: Request, who: Caller = Depends(caller)):
        """Body: {"document": {...}, "arrangement": "piano", "transpose": 0, "musicxml": true, "chart": false,
        "show": {"notes": true, "chords": true, "lyrics": true, "tempo": true, "key": true}}"""
        ok, retry = limiter.allow(("render:ip:" + who.ip) if who.is_public else ("render:key:" + who.owner), 600, 60)
        if not ok:
            raise HTTPException(429, "Too many requests. Please slow down.", headers={"Retry-After": str(retry)})
        body = await read_json(request)
        if not isinstance(body, dict):
            raise HTTPException(400, "The request body must be a JSON object")
        arrangement = body.get("arrangement", "piano")
        transpose = body.get("transpose", 0)
        flags = body.get("show") or {}
        capo = body.get("capo", 0)
        if arrangement not in VIEWS or isinstance(transpose, bool) or not isinstance(transpose, int) or not -11 <= transpose <= 11 or not isinstance(flags, dict):
            raise HTTPException(400, "Invalid arrangement, transpose or show options")
        if isinstance(capo, bool) or not isinstance(capo, int) or not 0 <= capo <= 11:
            raise HTTPException(400, "capo must be a whole number from 0 to 11")
        show = ShowOptions(capo=capo, **{name: flags.get(name, True) is not False for name in ("notes", "chords", "lyrics", "tempo", "key")})
        try:
            doc = normalize_document(body.get("document"))
            out = {"document": doc, "view": document_view(doc, transpose)}
            if body.get("musicxml", True):
                out["musicxml"] = export.musicxml_from_document(doc, arrangement, transpose, show)
            if body.get("chart"):
                out["chart"] = document_chart(doc, transpose, show=show)
            if body.get("tab"):
                out["tab"] = document_tab(doc, transpose, arrangement, show)
        except DocumentError as exc:
            raise HTTPException(400, str(exc)) from exc
        return out

    @app.delete("/v1/transcriptions/{tid}", status_code=204, tags=["Transcriptions"], summary="Delete a transcription")
    async def delete_transcription(tid: str, who: Caller = Depends(caller)):
        row = owned(tid, who)
        _unlink(audio_path(row["id"]))
        store.delete_transcription(row["id"])
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

    # Ad networks need their own script, frame and image origins; none are allowed unless ads are switched on.
    ads = settings.ad_origins if settings.ads else ""
    frames = ads or "'none'"
    page_headers = {
        "Cache-Control": "no-cache",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        # Scripts only from this server; artwork and preview audio only from the catalogue's CDN.
        "Content-Security-Policy": (
            f"default-src 'self'; script-src 'self' {ads}; style-src 'self' 'unsafe-inline'; font-src 'self' data:; "
            f"img-src 'self' data: {settings.media_origins} {ads}; media-src 'self' blob: {settings.media_origins}; "
            f"connect-src 'self' {ads}; frame-src {frames}; object-src 'none'; base-uri 'none'; form-action 'none'"
        ),
    }
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as fh:
        page = fh.read()
    if settings.ads:
        page = page.replace('<html lang="en">', '<html lang="en" class="ads-on">').replace("<!-- ads-script -->", '<script src="/static/ads.js" defer></script>')

    @app.get("/", include_in_schema=False)
    async def index():
        return HTMLResponse(page, headers=page_headers)

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
