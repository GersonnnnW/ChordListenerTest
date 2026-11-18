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
| `file` | file | | Audio (MP3, WAV, M4A/AAC, FLAC, OGG, AIFF) or video (MP4, MOV, MKV, WebM, AVI, WMV, FLV, MPEG); a video's soundtrack is used |
| `track_id` | string | | A track from `GET /v1/songs/search` |
| `vocabulary` | string | `standard` | `basic` = major/minor only. `standard` adds 7, maj7, m7, sus2, sus4, dim, aug. `full` adds 6, m6, m7b5, dim7 and reports extended chords more readily |
| `beats_per_bar` | integer | detect | Force `3` or `4` |
| `tempo` | string | `auto` | `half` or `double` counts the beat at half / twice the detected speed |
| `accuracy` | string | `standard` | High-accuracy mode: `piano` or `guitar` adds a large note detector made for that instrument. Takes about ten times longer (a 3-minute song: ~20 s instead of ~2 s). **`piano`**: on piano recordings far more of the notes are written correctly; on other music the standard notes are kept, so it is never worse. **`guitar`**: the guitar detector's notes are combined with the standard ones (each confirms or adds to the other); a small but consistent gain on guitar recordings. `high` is an older name for `piano`. See [Analyse again](#analyse-again-eg-in-high-accuracy-mode) to switch an existing transcription |
| `title`, `artist` | string | | Carried into the sheet music and file names, and used to look up lyrics. If omitted they are guessed from the file name ("Artist - Title.mp3") |

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

Requests made **without** an API key also get an `"edit_token"` in this response, once. Send it as the `X-Edit-Token` header to save corrections to that transcription later. Requests made with a key do not need it: the key is the credential.

```json
{
  "id": "tr_Zk3q9oXc1vJ8mB2nR5tY7w",
  "status": "completed",
  "created_at": "2026-11-16T14:02:11Z",
  "cached": false,
  "source": {"type": "upload", "filename": "song.mp3", "title": "My Song", "artist": "", "audio_seconds": 213.4,
             "options": {"vocabulary": "standard", "beats_per_bar": null, "tempo": "auto", "accuracy": "standard"}},
  "result": {
    "engine_version": "4",
    "duration": 213.4,
    "tempo_bpm": 73.0,
    "time_signature": "4/4",
    "key": {"name": "C major", "tonic": "C", "mode": "major", "confidence": 0.91},
    "transpose": 0,
    "beats": [0.51, 1.33, 2.15, 2.97],
    "first_downbeat": 0,
    "tuning": {"cents": -31, "reference_hz": 432.2, "corrected": true},
    "chords": [
      {"start": 0.51, "end": 3.79, "chord": "C", "root": "C", "quality": "maj", "bass": null,
       "notes": ["C", "E", "G"], "confidence": 0.62, "bar": 1, "beat": 1}
    ],
    "notes": [
      {"start": 0.51, "end": 1.02, "pitch": 60, "name": "C4", "velocity": 0.71}
    ],
    "processing_ms": 2740,
    "accuracy": "standard",
    "note_detector": "standard"
  },
  "links": {
    "self": "/v1/transcriptions/tr_Zk3q9oXc1vJ8mB2nR5tY7w",
    "score": "/v1/transcriptions/tr_Zk3q9oXc1vJ8mB2nR5tY7w/score",
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
- `tuning.cents` is how far the recording sits from concert pitch (negative = flat); `reference_hz` is the same thing as the frequency of its A. When the offset is 15 cents or more the audio is brought to pitch before analysis (`corrected`), so chords and notes are reported at the nearest standard pitch.
- For a video upload, `source.video` is `true` and `links.audio` points to an MP3 of its soundtrack until `audio_expires_at`.
- `cached` is `true` when this exact audio and settings had been analysed before; the request then returns in milliseconds.
- `accuracy` is the mode used: `standard`, `piano` or `guitar`. `note_detector` says which notes were written: the instrument's name when high-accuracy mode recognised that instrument and used its detector, otherwise `standard`.

## Analyse again, e.g. in high-accuracy mode

`POST /v1/transcriptions/{id}/reanalyze` — `multipart/form-data`

Analyses the same song again with the same settings (`vocabulary`, `beats_per_bar`, `tempo`) in the accuracy mode asked for. The result is a **new** transcription with its own ID (and, without an API key, its own `edit_token`); the original, and any corrections saved to it, are left as they were. The response is the same as for creating a transcription, and it counts as one request against rate limits and quotas.

| Field | Type | Default | |
|---|---|---|---|
| `accuracy` | string | `piano` | `standard`, `piano` or `guitar` |
| `file` | file | | The same file again. Needed only for an **uploaded** file whose result in that mode is not cached yet, because uploads are never kept. A different file is rejected |

A song found by name (`track_id`) needs nothing extra: the server fetches it again from the catalogue.

```bash
curl -H "Authorization: Bearer $KEY" -F "accuracy=guitar" \
     https://api.example.com/v1/transcriptions/tr_Zk3q9oXc1vJ8mB2nR5tY7w/reanalyze
```

## Fetch a transcription

`GET /v1/transcriptions/{id}`

| Query | Default | |
|---|---|---|
| `transpose` | `0` | Semitones, −11 to 11. Chord names, key and notes are respelled for the new key |
| `notes` | `true` | `false` leaves out the note list, which is most of the payload |

## Sheet music, guitar tab, MIDI, chord chart

| Endpoint | Returns |
|---|---|
| `GET /v1/transcriptions/{id}/musicxml` | MusicXML 4.0. `arrangement=piano` (two staves, all notes), `lead` (melody line), `guitar` (guitar notation over tablature of everything played) or `guitar_melody` (the same for the melody only) |
| `GET /v1/transcriptions/{id}/tab` | Plain-text guitar tab with chord symbols above. `source=notes` (everything played; best for a guitar recording) or `melody`; also `transpose`, `chords`, `capo`, `original`, `download` |
| `GET /v1/transcriptions/{id}/chart` | Plain-text chord chart, one cell per bar, lyrics under each row |
| `GET /v1/transcriptions/{id}/midi` | Standard MIDI file with the detected notes at their original timing and chord names as markers |

`musicxml` and `chart` reflect any saved corrections (see the next section) and accept these switches, all `true` by default:

| Query | Effect when `false` |
|---|---|
| `notes` | *(musicxml only)* a chord sheet: one rhythm slash per beat instead of the notes |
| `chords` | no chord symbols |
| `lyrics` | no lyrics |
| `tempo` | no tempo mark and no "Tempo: … BPM" in the heading |
| `key` | no "Key: …" in the heading (the key signature itself always stays) |

Also: `transpose` (−11 to 11), `capo=N` to print "Capo N" in the heading (see *Capo chords*), `original=true` to ignore saved corrections, and `download=true` to add a `Content-Disposition: attachment` header. All of these are generated on request and take a few milliseconds. `midi` always contains the notes as detected.

MusicXML opens in MuseScore, Sibelius, Finale, Dorico, Logic and most other notation software, and renders in the browser with OpenSheetMusicDisplay. The website turns it into a PDF in the browser; the API does not produce PDFs.

**Guitar tab.** Each note's string and fret is chosen as a guitarist would play it: the whole piece is fingered at once, following where on the neck real players play and how they move between notes (learned from GuitarSet, where 73% of notes land on the very string the guitarist used, on players never learned from). Bars of strummed chords (most attacks three or more notes at once) show the chord shape to hold, stacked, at the start of the bar and at each chord change, with rhythm slashes on the notation staff. Shapes are found for each chord (all ways to hold it with four fingers and a barre) and chosen to match the notes heard in the bar; on GuitarSet players never learned from, 54% of strings match the exact fret the guitarist played (most others are another way of holding the same chord). Standard tuning, or drop D when the music goes below the low E. The tab follows the notes, so corrections saved to the score (`PUT .../score`) carry over.

**Hammer-ons, pull-offs, slides, bends.** Notes in a single line that were not picked again, where the pitch moves continuously from one note to the next on one string, are marked as a hammer-on (`h`), pull-off (`p`) or slide (`/` up, `\` down); joined notes are kept on the same string. They appear in the text tab (`5h7`, `7p5`, `5/7`) and as hammer-on, pull-off and slide marks in MusicXML. The marks are sparse but mostly right (on GuitarSet, 78% of marked pairs really are joined on one string; only about one in five joined pairs is found). Bends are not detected automatically, because from an ordinary recording that was right only one time in five, but a bend entered in the document (`"b"`, on the note bent up to) is written as `7b9` and a MusicXML bend. They are stored in the score document as `"techniques": [[start, pitch, kind], ...]` (kind `h`, `p`, `/`, `\` or `b`, on the second note), and each note in a transcription's `notes` carries `"technique"` when it has one. `POST /v1/score/render` also takes `"arrangement": "guitar"` or `"guitar_melody"`, and `"tab": true` to return the text tab as `tab`.

## The editable score

Every finished transcription has a *score document*: plain JSON holding what is printed — chords, notes laid out per staff, lyrics, key, tempo, title. It starts as the automatic transcription; people correct it and save it back.

```json
{
  "version": 1,
  "title": "Let It Be", "artist": "The Beatles",
  "key": {"tonic": 0, "minor": false},
  "bpm": 73.0,
  "beats_per_bar": 4,
  "beats": [0.51, 1.33, 2.15],
  "chords": [[0, "C"], [2, "G"], [4, "Am"], [6, "Fmaj7"]],
  "lyrics": {"0": "When I find myself in", "1": "times of trouble"},
  "arrangements": {
    "piano": [[[0, 4, [60, 64, 67]], [4, 2, [72]]], [[0, 8, [36]]]],
    "lead":  [[[0, 4, [67]], [4, 2, [72]]]]
  }
}
```

- `key.tonic` is a pitch class (C = 0 … B = 11).
- `beats` are the beat times in seconds, with one extra entry for the final bar line. They tie the score to the recording; keep them as given.
- `chords` are `[beat index, symbol]`. The symbol is free text. Recognised chord symbols (`F#m7`, `Bb/D`, `Csus4`, `G7sus4`, `Eadd9`, `A5` …) are transposed and exported as proper chord symbols; anything else is printed as written. `N.C.` means no chord.
- `lyrics` maps a bar index (from 0, as a string) to the words sung in that bar.
- `arrangements` has `piano` (two staves: right hand, left hand) and `lead` (one staff). Each staff is a list of `[start, length, [pitches]]`, measured in sixteenth notes from the start of the piece; pitches are MIDI numbers at the original pitch. Notes on one staff may not overlap: a note that runs into the next is shortened, and several notes sounding together are one entry with several pitches. Gaps are rests.

| Endpoint | |
|---|---|
| `GET /v1/transcriptions/{id}/score` | `{"document": …, "edited": bool, "can_save": bool, "view": …}`. `original=true` returns the automatic version; `transpose` affects `view` |
| `PUT /v1/transcriptions/{id}/score` | Save a corrected document (the request body). Needs the API key that created the transcription, or its `X-Edit-Token`; otherwise `403`. A malformed document is refused with `400` and a message saying what is wrong |
| `DELETE /v1/transcriptions/{id}/score` | Discard the corrections; the automatic version is back |
| `POST /v1/score/render` | Render a document without storing anything |

`view` is the document as a client should display it at the requested transposition: the key name, tempo, bar count, and each chord with its start and end time, displayed symbol and note names.

`POST /v1/score/render` takes

```json
{"document": {}, "arrangement": "piano", "transpose": 0, "capo": 0, "musicxml": true, "chart": false,
 "show": {"notes": true, "chords": true, "lyrics": true, "tempo": true, "key": true}}
```

and returns the validated `document`, its `view`, and the `musicxml` and/or `chart` text. This is what the website's editor calls after every change, so unsaved edits can be previewed and downloaded.

### Capo chords

`view.guitar` (from the score endpoints) is `null` when the chords are already comfortable on guitar, and otherwise suggests how a guitarist would play them:

```json
{"transpose": -1, "capo": 1, "tune_down": 0, "chords": ["Am", "G", "C"]}
```

Request anything with `transpose=-1&capo=1` to get those chords, with "Capo 1" printed in the heading of the sheet music and chord chart. `tune_down: 1` (with `transpose: 1`) means the shapes are easy on a guitar tuned a half step down.

## Lyrics

| Endpoint | |
|---|---|
| `GET /v1/transcriptions/{id}/lyrics` | What the lyrics provider found when the transcription was made |
| `POST /v1/transcriptions/{id}/lyrics` | Look again under a corrected `title` and `artist` (form fields). Stored only when called by the creator |

```json
{"found": true, "source": "lrclib", "link": "https://lrclib.net", "title": "Let It Be", "artist": "The Beatles",
 "synced": true, "aligned": true,
 "lines": [{"time": 12.48, "text": "…"}],
 "bars": {"3": "…", "4": "…"}}
```

`aligned` is `true` when the lyrics are timed and the analysed audio is the same full-length recording they were timed for. Then `bars` gives the words per bar (bar index from 0, as in the score document), and an unedited score already contains them. For a catalogue excerpt `aligned` is `false`: the lines are returned but cannot be placed. `{"found": false}` means nothing matched.

To use your own licensed lyrics source, implement `LyricsProvider.find(title, artist, duration)` (`chordlistener/server/lyricsdb.py`) and pass it to `create_app(settings, lyrics=MyLyrics())`.

## Converted audio

`GET /v1/transcriptions/{id}/audio` returns the MP3 made from an uploaded video. It needs the API key that created the transcription, or its edit token as the `X-Edit-Token` header or a `token` query parameter. `404` once it has expired.

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

`GET /v1/health` → `{"status": "ok", "version": "0.1.0", "engine_version": "4", "catalog": "itunes", "lyrics": "lrclib", "max_upload_mb": 1000, "high_accuracy": ["piano", "guitar"]}`

`high_accuracy` lists the instruments this server offers high-accuracy mode for (empty when it is switched off).

## Errors

Every error has the same shape:

```json
{"error": {"code": "rate_limited", "message": "Too many requests. Please slow down."}}
```

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Missing or conflicting fields, unknown option value, high-accuracy mode not offered, or re-analysing an upload without sending the file again |
| 401 | `unauthorized` | Missing, invalid or revoked key |
| 403 | `forbidden` | Saving or discarding corrections without the key or edit token that created the transcription |
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
| Upload size | 1000 MB |
| Converted MP3 of a video upload | kept 120 minutes |
| Audio length analysed | 15 minutes (longer files are cut, not rejected) |
| Keyed requests | 60 per minute per key (set per key) |
| Keyless requests | 30 analyses per hour per IP |
| Result retention | 30 days |

All are configurable by the operator; see the README.

## Versioning

The URL prefix (`/v1`) changes only for breaking changes to the API shape. `engine_version` in every result changes when the analysis itself is improved; results are cached per engine version, so after an engine upgrade the same audio is re-analysed on the next request rather than served stale.
