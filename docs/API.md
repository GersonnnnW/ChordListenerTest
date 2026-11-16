# ChordListener API

Base URL: wherever the server is deployed, e.g. `https://api.example.com`. All endpoints are under `/v1`. An interactive version of this reference (OpenAPI) is served at `/docs`, and the machine-readable schema at `/openapi.json`.

## Authentication

Send the API key in either header:

```
Authorization: Bearer cl_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
X-API-Key: cl_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Keys are created by the operator with `chordlistener keys create --name "Customer"`. Only a hash is stored, so a lost key is replaced, not recovered.

If the server runs with `CL_PUBLIC_ACCESS=true` (the default, for the public website), requests without a key are accepted under a per-IP hourly limit. With `CL_PUBLIC_ACCESS=false` every request needs a key.

Transcriptions created with a key can only be read with that key. Transcriptions created without one can be read by anyone who has the ID.

## Create a transcription

`POST /v1/transcriptions` — `multipart/form-data`

Send **exactly one** of `file` or `track_id`.

| Field | Type | Default | |
|---|---|---|---|
| `file` | file | | MP3, WAV, M4A/AAC, FLAC, OGG, AIFF or WebM |
| `track_id` | string | | A track from `GET /v1/songs/search` |
| `vocabulary` | string | `standard` | `basic` = major/minor only. `standard` adds 7, maj7, m7, sus2, sus4, dim, aug. `full` adds 6, m6, m7b5, dim7 and reports extended chords more readily |
| `beats_per_bar` | integer | detect | Force `3` or `4` |
| `tempo` | string | `auto` | `half` or `double` counts the beat at half / twice the detected speed |
| `title`, `artist` | string | | Carried into the sheet music and file names |

Query parameter `wait` (default `true`): hold the request open until the result is ready. Most songs finish in a few seconds. With `wait=false`, or if the wait runs out, the response is `202` with `"status": "processing"`; poll `GET /v1/transcriptions/{id}`.

```bash
curl -H "Authorization: Bearer $KEY" \
     -F "file=@song.mp3" -F "title=My Song" \
     https://api.example.com/v1/transcriptions
```

```python
import requests

with open("song.mp3", "rb") as audio:
    r = requests.post(
        "https://api.example.com/v1/transcriptions",
        headers={"Authorization": f"Bearer {KEY}"},
        files={"file": audio},
    )
result = r.json()["result"]
for chord in result["chords"]:
    print(f'{chord["start"]:7.2f}s  bar {chord["bar"]}.{chord["beat"]}  {chord["chord"]}')
```

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);
const response = await fetch("https://api.example.com/v1/transcriptions", {
  method: "POST",
  headers: { Authorization: `Bearer ${KEY}` },
  body: form,
});
const { id, result } = await response.json();
```

### Response

`200` when complete, `202` while processing, `422` if the audio could not be analysed.

```json
{
  "id": "tr_Zk3q9oXc1vJ8mB2nR5tY7w",
  "status": "completed",
  "created_at": "2026-11-16T14:02:11Z",
  "cached": false,
  "source": {"type": "upload", "filename": "song.mp3", "title": "My Song", "artist": "", "audio_seconds": 213.4,
             "options": {"vocabulary": "standard", "beats_per_bar": null, "tempo": "auto"}},
  "result": {
    "engine_version": "1",
    "duration": 213.4,
    "tempo_bpm": 73.0,
    "time_signature": "4/4",
    "key": {"name": "C major", "tonic": "C", "mode": "major", "confidence": 0.91},
    "transpose": 0,
    "beats": [0.51, 1.33, 2.15, 2.97],
    "first_downbeat": 0,
    "chords": [
      {"start": 0.51, "end": 3.79, "chord": "C", "root": "C", "quality": "maj", "bass": null,
       "notes": ["C", "E", "G"], "confidence": 0.62, "bar": 1, "beat": 1}
    ],
    "notes": [
      {"start": 0.51, "end": 1.02, "pitch": 60, "name": "C4", "velocity": 0.71}
    ],
    "processing_ms": 2740
  },
  "links": {
    "self": "/v1/transcriptions/tr_Zk3q9oXc1vJ8mB2nR5tY7w",
    "musicxml": "/v1/transcriptions/tr_Zk3q9oXc1vJ8mB2nR5tY7w/musicxml",
    "midi": "/v1/transcriptions/tr_Zk3q9oXc1vJ8mB2nR5tY7w/midi",
    "chart": "/v1/transcriptions/tr_Zk3q9oXc1vJ8mB2nR5tY7w/chart"
  }
}
```

Field notes:

- `beats` are the beat times in seconds. Beat `i` belongs to bar `i // beats_per_bar + 1`. Bar 1 starts at the bar line at or just before the first sound, so `beats[0]` can be slightly negative when a song opens with a pickup.
- `chords[].bar` and `.beat` are 1-based. `chord` is `"N.C."` (and `root` is `null`) where there is no chord.
- `chords[].bass` is set only for clear inversions, e.g. `C/E`.
- `chords[].confidence` and `key.confidence` run from 0 to 1 and are relative indicators, not probabilities.
- `notes[].pitch` is a MIDI note number (60 = middle C); `velocity` is the model's 0–1 strength for the note.
- `cached` is `true` when this exact audio and settings had been analysed before; the request then returns in milliseconds.

## Fetch a transcription

`GET /v1/transcriptions/{id}`

| Query | Default | |
|---|---|---|
| `transpose` | `0` | Semitones, −11 to 11. Chord names, key and notes are respelled for the new key |
| `notes` | `true` | `false` leaves out the note list, which is most of the payload |

## Sheet music, MIDI, chord chart

| Endpoint | Returns |
|---|---|
| `GET /v1/transcriptions/{id}/musicxml` | MusicXML 4.0. `arrangement=piano` (two staves, all notes, chord symbols) or `arrangement=lead` (melody line with chord symbols) |
| `GET /v1/transcriptions/{id}/midi` | Standard MIDI file with the detected notes at their original timing and chord names as markers |
| `GET /v1/transcriptions/{id}/chart` | Plain-text chord chart, one cell per bar |

All three accept `transpose`. `musicxml` and `chart` accept `download=true` to add a `Content-Disposition: attachment` header. They are generated on request from the stored analysis and take a few milliseconds.

MusicXML opens in MuseScore, Sibelius, Finale, Dorico, Logic and most other notation software, and renders in the browser with OpenSheetMusicDisplay.

## Delete a transcription

`DELETE /v1/transcriptions/{id}` → `204`

## Find a song by name

`GET /v1/songs/search?q=let+it+be&limit=8`

```json
{
  "provider": "itunes",
  "results": [
    {"track_id": "1441164738", "provider": "itunes", "title": "Let It Be", "artist": "The Beatles",
     "album": "Let It Be", "artwork_url": "https://...", "preview_url": "https://...",
     "link": "https://music.apple.com/...", "duration": 243.0, "clip_seconds": 30.0}
  ]
}
```

Pass a result's `track_id` to `POST /v1/transcriptions`. `clip_seconds` says how much audio the provider supplies: the built-in provider gives a 30-second preview, so the transcription covers that excerpt. For whole songs, upload the audio.

### Bring your own catalogue

Search by name is a thin layer over a `CatalogProvider` (`chordlistener/server/catalog.py`) with three methods:

```python
class MyCatalog(CatalogProvider):
    name = "acme"
    async def search(self, query: str, limit: int = 8) -> list[Track]: ...
    async def lookup(self, track_id: str) -> Track | None: ...
    async def download(self, track: Track, dest_path: str, max_bytes: int) -> str: ...  # returns the SHA-256 of the audio
```

Pass an instance to `create_app(settings, catalog=MyCatalog())`. Results are cached per `(provider, track_id)`, so each track is fetched and analysed once.

## Usage

`GET /v1/usage` — for the calling key, this calendar month (UTC):

```json
{"key": {"id": "key_3f9a1c2b7d10", "name": "Acme Music"}, "period_start": "2026-11-01T00:00:00Z",
 "monthly_seconds_limit": 300000, "requests": 412, "audio_seconds": 88213.5, "cached_requests": 97}
```

## Health

`GET /v1/health` → `{"status": "ok", "version": "0.1.0", "engine_version": "1", "catalog": "itunes"}`

## Errors

Every error has the same shape:

```json
{"error": {"code": "rate_limited", "message": "Too many requests. Please slow down."}}
```

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Missing or conflicting fields, unknown option value |
| 401 | `unauthorized` | Missing, invalid or revoked key |
| 404 | `not_found` | Unknown ID, or a transcription that belongs to another key |
| 409 | `not_ready` | Asked for sheet music of a transcription still processing |
| 413 | `file_too_large` | Upload over the size limit |
| 422 | `invalid_audio` / `unprocessable` | Not a readable audio file, too short, or an invalid query value |
| 429 | `rate_limited` | Rate limit or monthly quota reached; see the `Retry-After` header |
| 502 | `catalog_unavailable` | The song catalogue could not be reached |

A failed analysis is also reported inside the transcription object: `"status": "failed"` with an `error` member.

## Limits

| | Default |
|---|---|
| Upload size | 60 MB |
| Audio length analysed | 15 minutes (longer files are cut, not rejected) |
| Keyed requests | 60 per minute per key (set per key) |
| Keyless requests | 30 analyses per hour per IP |
| Result retention | 30 days |

All are configurable by the operator; see the README.

## Versioning

The URL prefix (`/v1`) changes only for breaking changes to the API shape. `engine_version` in every result changes when the analysis itself is improved; results are cached per engine version, so after an engine upgrade the same audio is re-analysed on the next request rather than served stale.
