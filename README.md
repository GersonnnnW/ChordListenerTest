# ChordListener

Give it a song — an audio file, or just the song's name — and get back:

- the **chords**, with timings, bar and beat positions
- the **individual notes** (polyphonic transcription)
- the **key, tempo and time signature**
- **sheet music** as MusicXML (piano score or lead sheet), a **MIDI** file and a plain-text **chord chart**

It ships as three things built on one engine:

| | |
|---|---|
| **Website** | Search or upload, then follow the chords in time with the music, transpose, view and print the sheet music. |
| **HTTP API** | Versioned REST API with API keys, rate limits, quotas and usage metering, for other businesses to integrate. |
| **Python library / CLI** | `chordlistener analyze song.mp3` — the same engine with no server. |

## How it works, and why it is cheap to run

No large language model is involved, so **analysing a song costs zero AI tokens**. The pipeline is signal processing plus one very small neural network (about 17,000 parameters, 230 KB) that runs on an ordinary CPU:

1. **Decode** the audio to mono 22 kHz (ffmpeg).
2. **Pitch model** — one pass of Spotify's open-source Basic Pitch network gives, for every 11 ms frame, which of the 88 piano keys are sounding and where notes start. That single pass feeds everything below.
3. **Notes** are decoded from those activations.
4. **Beats** — tempo from the autocorrelation of an onset-strength curve, then a dynamic-programming beat tracker; bar lines are placed where the harmony changes and the bass lands.
5. **Chords** — pitch activations are folded into a 12-note profile per beat, matched against chord templates, and decoded over the whole song with a Viterbi pass that knows chords tend to change on bar lines.
6. **Key** — from the pitch distribution and the chord progression together (which is what separates G major from E minor).
7. **Notation** — notes are snapped to a sixteenth-note grid, split into bars with ties and rests, spelled according to the key, and written as MusicXML.

Measured on a 4-core cloud VM, no GPU:

| | |
|---|---|
| Analysis speed | about **95× real time** — a 4-minute MP3 in about 2.5 seconds, decoding included |
| 30-second preview clip, end to end | about **0.5 s** |
| Same audio requested again | about **40 ms** (served from the result cache) |
| Sheet music / MIDI / transposed versions of a finished analysis | a few ms each, generated on demand from the stored result |

Results are cached by a hash of the audio, so a song is only ever analysed once per engine version, no matter how many people ask for it.

## Quick start

Requires Python 3.10+ and `ffmpeg` on the PATH.

```bash
pip install -e ".[server,dev]"
```

```bash
chordlistener serve
```

Open http://127.0.0.1:8000 for the website and http://127.0.0.1:8000/docs for interactive API documentation.

Analyse a file without the server:

```bash
chordlistener analyze song.mp3 --out results/
```

That prints a chord chart and writes `song.json`, `song.piano.musicxml`, `song.lead.musicxml`, `song.mid` and `song.chords.txt`.

### Docker

```bash
docker build -t chordlistener .
```

```bash
docker run -p 8000:8000 -v chordlistener-data:/data chordlistener
```

## Using the API

```bash
curl -F "file=@song.mp3" http://127.0.0.1:8000/v1/transcriptions
```

```json
{
  "id": "tr_Zk3...",
  "status": "completed",
  "result": {
    "duration": 213.4,
    "tempo_bpm": 73.0,
    "time_signature": "4/4",
    "key": {"name": "C major", "tonic": "C", "mode": "major", "confidence": 0.91},
    "chords": [
      {"start": 0.51, "end": 3.79, "chord": "C", "root": "C", "quality": "maj", "bass": null,
       "notes": ["C", "E", "G"], "confidence": 0.62, "bar": 1, "beat": 1}
    ],
    "notes": [{"start": 0.51, "end": 1.02, "pitch": 60, "name": "C4", "velocity": 0.71}],
    "beats": [0.51, 1.33, 2.15]
  },
  "links": {"musicxml": "/v1/transcriptions/tr_Zk3.../musicxml", "midi": "...", "chart": "..."}
}
```

By song name:

```bash
curl "http://127.0.0.1:8000/v1/songs/search?q=let+it+be"
```

```bash
curl -F "track_id=1441164738" http://127.0.0.1:8000/v1/transcriptions
```

The full reference, including errors, limits and asynchronous mode, is in [docs/API.md](docs/API.md).

## Offering it to other businesses

Create a key per customer:

```bash
chordlistener keys create --name "Acme Music" --per-minute 120 --monthly-minutes 5000
```

The customer sends it as `Authorization: Bearer cl_...`. Each key has its own rate limit and optional monthly audio quota, its results are private to that key, and every request is recorded in a usage table (seconds of audio, cached or not) that you can bill from. `GET /v1/usage` lets the customer see their own consumption.

Ways a customer can integrate:

- **Server to server** — call the REST API from their backend. This is the normal route.
- **Embedded widget** — put the tool in an iframe: `<iframe src="https://your-host/?embed=1">`. Embed mode hides the site header and footer.
- **Self-hosted** — hand over the Docker image; set `CL_PUBLIC_ACCESS=false` so only keyed requests are served.
- **Their own song catalogue** — implement the two-method `CatalogProvider` interface in `chordlistener/server/catalog.py` so "find a song by name" searches their licensed library instead of the default one.

## Configuration

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `CL_DATA_DIR` | `./data` | Where the SQLite database lives |
| `CL_PUBLIC_ACCESS` | `true` | Allow requests without an API key (the public website). Set `false` for a keys-only API |
| `CL_PUBLIC_PER_HOUR` | `30` | Analyses per hour per IP address for keyless callers |
| `CL_MAX_UPLOAD_MB` | `60` | Largest accepted upload |
| `CL_MAX_SECONDS` | `900` | Longest audio analysed; anything beyond is cut off |
| `CL_WORKERS` | `2` | Analyses running at the same time |
| `CL_MODEL_THREADS` | all cores | CPU threads the pitch model may use per analysis |
| `CL_CATALOG` | `itunes` | Song lookup provider; `none` disables search by name |
| `CL_CORS_ORIGINS` | `*` | Browser origins allowed to call the API |
| `CL_TRUST_PROXY` | `false` | Read the client IP from `X-Forwarded-For` (set when behind a reverse proxy) |
| `CL_RETENTION_DAYS` | `30` | Days results are kept; `0` keeps them forever |
| `CL_WAIT_SECONDS` | `120` | How long a request is held open before it turns into a pollable job |

Uploaded audio is written to a temporary file, analysed and deleted. Only the transcription is stored.

## Accuracy: what to expect

Automatic transcription is not perfect, and it is better to know where it is strong and weak.

On generated test songs with known answers (`python -m tools.evaluate`, 10 songs across styles, keys and meters): chord roots 99%, exact chord 93%, key 10/10, meter 10/10, note F-score 0.74.

On a dozen real commercial recordings checked by hand against their known chords:

- **Chord roots and key are usually right** on songs with a clear harmonic backing — piano, guitar, a band. The typical output for a pop song is the correct progression.
- **Chord colour is conservative.** Sevenths and suspended chords are only reported when clearly present, so some real 7ths come back as plain triads. `vocabulary=full` loosens this; `vocabulary=basic` forces major/minor only.
- **Tempo can come out at half or double** the speed a musician would count (70 vs 140 bpm). The chords and timings are unaffected; the bar lengths and note values in the sheet music are. The website has ½× / 2× buttons and the API a `tempo` option to flip it.
- **Dense or unusual material is harder**: heavily distorted guitars, songs built on drones or fast arpeggios, 6/8 and other meters beyond 3/4 and 4/4, rap, and anything without stable pitched instruments.
- **Sheet music is a faithful sketch, not an engraver's edition.** The piano score writes down every detected note split across two staves; the lead sheet picks the top line as melody. Both will need tidying by a musician before publication.

## Before you sell it: licences and rights

This is not legal advice — treat it as a checklist to take to a lawyer.

- **The code's dependencies allow commercial use.** The pitch model is Apache-2.0, the notation renderer BSD-3, the Python libraries MIT/BSD. Attribution notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and must ship with the product. ffmpeg is called as a separate program; check the licence of the build you distribute.
- **Search by name uses Apple's iTunes Search API and its 30-second previews.** That API is intended for promoting Apple's store, and its terms limit how previews may be used. It is fine for a demo; a commercial product should use a catalogue you have rights to, which is what the `CatalogProvider` interface is for. This is also why search by name analyses a 30-second clip: there is no legitimate free source of full-length commercial recordings. Full songs come from audio the user or the business customer supplies.
- **Chords and sheet music of copyrighted songs are themselves subject to copyright.** A tool that transcribes a user's own upload for them is in a different position from a site that publishes transcriptions of famous songs; the second normally needs agreements with music publishers.
- **This repository has no licence file**, which means all rights are reserved by its owner. Add the commercial licence terms you intend to sell under before giving anyone access.

## Development

```bash
python -m pytest
```

```bash
python -m tools.evaluate
```

The tests need only ffmpeg; they synthesise their own audio. `tools.evaluate` uses FluidSynth and a General MIDI soundfont for realistic instrument sounds when they are installed, and falls back to the built-in synth otherwise.

```
chordlistener/
  audio.py      decode with ffmpeg; reject anything that is not a real audio container
  pitch.py      pitch model inference and note decoding
  beats.py      onset strength, tempo, beat tracking, bar finding
  chords.py     beat-synchronous chroma, chord templates, Viterbi decoding, key
  theory.py     note spelling, key signatures, chord vocabulary
  pipeline.py   ties the stages together; the Analysis result type
  score.py      quantisation into bars, voices, ties and rests
  musicxml.py   MusicXML writer
  midi.py       MIDI writer
  export.py     MusicXML / MIDI / chord chart from an Analysis
  cli.py        command line
  models/       the pitch model (ONNX) and its licence
  server/       FastAPI app, SQLite store, catalogue providers, website
tools/          test-song generator and accuracy evaluation
tests/
```

### Scaling notes

One container handles a steady stream of requests: an analysis occupies a worker for a few seconds, and repeats are cache hits. The rate limiter and the in-flight de-duplication are in memory, and storage is a single SQLite file, so the current design is one process per data directory. To run several replicas behind a load balancer, move `Store` to PostgreSQL and the rate limiter to Redis; both are isolated behind small classes (`server/store.py`, `RateLimiter` in `server/app.py`) for that reason.
