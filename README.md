# ChordListener

Give it a song — an audio or video file, or just the song's name — and get back:

- the **chords**, with timings, bar and beat positions
- the **individual notes** (polyphonic transcription)
- the **key, tempo and time signature**
- **sheet music** as MusicXML (piano score or lead sheet), a **MIDI** file and a plain-text **chord chart**

It ships as three things built on one engine:

| | |
|---|---|
| **Website** | Search or upload, follow the chords in time with the music, correct mistakes, add lyrics, and download the sheet music as a PDF. |
| **HTTP API** | Versioned REST API with API keys, rate limits, quotas and usage metering, for other businesses to integrate. |
| **Python library / CLI** | `chordlistener analyze song.mp3` — the same engine with no server. |

## How it works, and why it is cheap to run

No large language model is involved, so **analysing a song costs zero AI tokens**. The pipeline is signal processing plus one very small neural network (about 17,000 parameters, 230 KB) that runs on an ordinary CPU:

1. **Decode** the audio to mono 22 kHz (ffmpeg). Video files work too: the soundtrack is extracted.
1. **Tuning check** — measure how far the recording sits from concert pitch (A = 440 Hz). Old tape transfers and bands that tune to each other can be a third of a semitone off, which blurs every note across two piano keys; if so the audio is brought to pitch before analysis.
2. **Pitch model** — one pass of Spotify's open-source Basic Pitch network gives, for every 11 ms frame, which of the 88 piano keys are sounding and where notes start. That single pass feeds everything below.
3. **Notes** — candidate notes are read from those activations generously, then a small trained **note filter** judges each one in context (what sounds an octave or a twelfth below it, whether the same key was already ringing, how it starts, what the audio spectrum shows at its partials). It removes overtones and drum hits mistaken for notes, and recovers quiet notes a fixed threshold would miss.
4. **Beats** — tempo from the autocorrelation of an onset-strength curve, then a dynamic-programming beat tracker; bar lines are placed where the harmony changes and the bass lands.
5. **Chords** — pitch activations are folded into a 12-note profile per beat, matched against chord templates, and decoded over the whole song with a Viterbi pass that knows chords tend to change on bar lines.
6. **Key** — from the pitch distribution and the chord progression together (which is what separates G major from E minor).
7. **Arrangement** — every trusted note is written, each given to a hand that can reach it (no hand stretches beyond a ninth or so; where a chord is too wide, only notes that double another an octave away are left out, so the harmony is kept). Sixteenths are only used when the music really has them. For the lead sheet a second trained model picks out the **tune**: it scores each note for being the melody, and a line-finder chooses one smooth, likely line through the song.
8. **Notation** — the result is split into bars with ties and rests, spelled according to the key, and written as MusicXML.

**Optional high-accuracy mode, for piano or guitar** (a choice on the website, `accuracy=piano` or `accuracy=guitar` in the API, `--high-accuracy piano|guitar` on the command line). A second, much larger note detector made for that instrument (about 20 million parameters) listens to the recording as well. Beats, chords and key always come from the standard pipeline. A finished transcription can be switched afterwards with one click (or `POST /v1/transcriptions/{id}/reanalyze`).

- **Piano** — ByteDance's high-resolution piano transcription model. Where the music is piano (recognised because the two detectors then agree on at least 85% of the notes) its notes are written instead. On guitar, voice and band recordings, where it misses or invents notes, the standard notes are kept, so the mode is never worse. A big gain on piano.
- **Guitar** — Xavier Riley's guitar version of the same design (trained on classical guitar). On its own it is no better than the standard notes, but the two make different mistakes, so they are combined: a note the standard filter was unsure of is kept only if the guitar detector heard it too, and a note only the guitar detector heard is added when it is sure. A small but consistent gain (1–3 points on every guitar test set, never worse). A guitar-trained model of piano quality would need training on guitar recordings, which needs a GPU.

Measured on a 4-core cloud VM, no GPU:

| | |
|---|---|
| Analysis speed | about **60× real time** — a 4-minute MP3 in about 4 seconds, decoding included |
| High-accuracy mode | about **8× real time** — a 3-minute song in about 23 seconds |
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

## The website

- **Chords** — one cell per bar, highlighted in time with the audio. Click a bar to jump there. Transpose up or down.
- **Sheet music** — piano score or lead sheet, drawn in the browser.
- **Show / hide** — switches for Notes, Chords, Lyrics, Key and BPM. With Notes off the sheet becomes a chord sheet (rhythm slashes with chord symbols and lyrics).
- **Edit** — fix what the transcription got wrong:
  - *Chords:* click any beat and type the chord (`F#m7`, `Bb/D`, `Csus4`, `N.C.` …). Empty removes it.
  - *Notes:* on the sheet, click a note to raise or lower it, change its length, move it earlier or later, add notes to make a chord, or delete it; click a rest to add a note there. Arrow keys, `[` `]`, Delete and Ctrl+Z also work.
  - *Lyrics:* type under each bar, or paste a whole verse and have it spread across bars.
  - *Title, artist, key, BPM.*
  - **Save** stores the corrected version; **Reset to original** brings back the automatic one. Undo and redo cover the last 60 changes.
- **Download** — a PDF with a choice of what goes in it: the notes, chords, lyrics, key and tempo, each on or off, as a piano score or a lead sheet. MusicXML, MIDI, a text chord chart and JSON are also offered.

- **Guitar tip** — when the sounding chords are awkward on guitar (Bbm, Ab, Db…), the page suggests the capo position that turns them into easy shapes (capo 1: Am, G, C) and shows those chords with one click. This is usually why chords "look wrong" next to a chord sheet: sheets are written for the capo, the recording sounds a few semitones higher.
- **High accuracy** — under the search and upload panels, choose *Standard*, *High accuracy: piano* or *High accuracy: guitar* before choosing a song, or change it with the drop-down next to the result afterwards. It takes about ten times longer. Piano: 95% of written notes right instead of 89%, and 93% of played notes written instead of 73%. Guitar: a smaller gain (see Accuracy). On music that is not the chosen instrument the page says "standard notes kept" and uses the standard notes, which are better there.
- **Tuning** — if the recording is not at concert pitch, the page says so (for example "A = 432 Hz").
- **Video uploads** — MP4, MOV, MKV, WebM, AVI, WMV, FLV and MPEG files are accepted directly, up to 1 GB by default, with an upload progress bar. The soundtrack is converted to MP3 on the server, used for playback when the browser cannot play the video itself, and offered as a download.

Who can save: the browser that created a transcription holds a private edit token for it, so a link can be shared without others overwriting the work. Anyone opening the link can still edit their own copy on the page and download it.

**Lyrics** are looked up by title and artist in LRCLIB, a free community lyrics database that also has line timings:

- For a **full-length upload** whose title, artist and length match a database entry, the lines are placed under the bars where they are sung, and appear on the chord chart, the sheet music and the PDF. The title and artist are read from the file name ("Artist - Title.mp3").
- For a **song found by name**, only a 30-second excerpt is analysed, and there is no way to know which lines fall in it. The full lyrics are shown as text under the chords instead.
- If nothing is found, or it is the wrong cut, fix the title and artist in Edit and press **Find lyrics**, or paste the words in.

When only a file name is known, a match is accepted only if the title is identical and the recording length is the same to within a few seconds, because showing the wrong song's words is worse than showing none. Lyrics are not transcribed from the singing: doing that reliably on a full mix needs a large speech model, which would cost far more time and compute than everything else here combined.

PDFs are produced in the visitor's browser, so they cost the server nothing.

### Advertising spaces

The page layout has three reserved ad positions — under the header, below a finished transcription, and above the footer — that take up no room until switched on:

```bash
CL_ADS=true chordlistener serve
```

With ads on, each position reserves banner-sized space (728×90, or 320×50 on phones) and shows a dashed placeholder until an ad fills it. Put your ad network's code in `chordlistener/server/web/ads.js` (the file explains how) and list the network's domains in `CL_AD_ORIGINS` so the page's security policy allows them. Ads never appear in embed mode or in downloaded sheet music.

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
| `CL_MAX_UPLOAD_MB` | `1000` | Largest accepted upload (videos are big). A reverse proxy in front has its own limit, e.g. nginx `client_max_body_size` |
| `CL_KEEP_AUDIO_MINUTES` | `120` | How long the MP3 made from an uploaded video is kept for playback; `0` keeps none |
| `CL_LYRICS` | `lrclib` | Lyrics provider; `none` disables lyric lookup |
| `CL_MAX_SECONDS` | `900` | Longest audio analysed; anything beyond is cut off |
| `CL_WORKERS` | `2` | Analyses running at the same time |
| `CL_MODEL_THREADS` | all cores | CPU threads the pitch model may use per analysis |
| `CL_CATALOG` | `itunes` | Song lookup provider; `none` disables search by name |
| `CL_CORS_ORIGINS` | `*` | Browser origins allowed to call the API |
| `CL_HIGH_ACCURACY` | `true` | Offer the optional high-accuracy mode (about ten times the CPU per song that uses it) |
| `CL_ADS` | `false` | Show the reserved advertising spaces on the website |
| `CL_AD_ORIGINS` | none | Ad-network domains the page may load scripts, frames and images from (space separated) |
| `CL_MEDIA_ORIGINS` | Apple's CDN | Origins the website may load artwork and preview audio from; change it when you plug in your own catalogue |
| `CL_TRUST_PROXY` | `false` | Read the client IP from `X-Forwarded-For` (set when behind a reverse proxy) |
| `CL_RETENTION_DAYS` | `30` | Days results are kept; `0` keeps them forever |
| `CL_WAIT_SECONDS` | `120` | How long a request is held open before it turns into a pollable job |

Uploads are written to a temporary file, analysed and deleted. Only the transcription is stored, plus, for video uploads, an MP3 of the soundtrack that is deleted after `CL_KEEP_AUDIO_MINUTES` and can only be fetched by the uploader.

## Accuracy: what to expect

Automatic transcription is not perfect, and it is better to know where it is strong and weak.

On generated test songs with known answers (`python -m tools.evaluate`, 10 songs across styles, keys and meters): chord roots 99%, exact chord 92%, key 10/10, meter 10/10, note F-score 0.87.

**Sheet music** (`python -m tools.evaluate_sheet --real DATA`), scored against what was actually played. "Right" is the share of written notes that were played; "written" is the share of played notes that appear on the page:

| Piano score, both hands | Right | Written |
|---|---|---|
| Real piano recordings (never used in training) | 89% (was 78%) | 73% (was 52%) |
| Real piano, **high-accuracy mode: piano** | **95%** | **93%** |
| Real guitar recordings, players held out of training | 83% (was 72%) | 77% (was 56%) |
| Generated songs | 89% (was 62%) | 78% (was 55%) |
| Generated songs, instruments held out of training | 77% (was 69%) | 75% (was 61%) |

The lead-sheet melody on generated songs: 77–95% of written notes are the tune, depending on the test set (was 53–80%), and 64–86% of the tune is written. Real-recording figures come from public research datasets with human-checked notes (piano: SMD; guitar: GuitarSet).

On a dozen real commercial recordings checked by hand against their known chords:

- **Chord roots and key are usually right** on songs with a clear harmonic backing — piano, guitar, a band. The typical output for a pop song is the correct progression.
- **Chord colour is conservative.** Sevenths and suspended chords are only reported when clearly present, so some real 7ths come back as plain triads. `vocabulary=full` loosens this; `vocabulary=basic` forces major/minor only.
- **Tempo can come out at half or double** the speed a musician would count (70 vs 140 bpm). The chords and timings are unaffected; the bar lengths and note values in the sheet music are. The website has ½× / 2× buttons and the API a `tempo` option to flip it.
- **Dense or unusual material is harder**: heavily distorted guitars, songs built on drones or fast arpeggios, 6/8 and other meters beyond 3/4 and 4/4, rap, and anything without stable pitched instruments.
- **Sheet music is close, not perfect.** Roughly one written note in ten is wrong and about a quarter of what was played is still missing, mostly fast inner notes and notes buried under louder ones. Expect a musician to correct a few notes per page (the website's Edit mode is for that). **Singing is the weakest part**: on solo-voice recordings only about 60% of detected notes are right, because voices slide and wobble between pitches; in full mixes this shows up as stray neighbouring notes in the right hand. A low (typically male) vocal is written an octave up on the lead sheet, as lead sheets do. The MIDI download contains every detected note.

### Retraining the models

Both models are small (under 100 KB together) and run in plain numpy. To rebuild them, install the training extras (`pip install -e ".[train]"`) plus FluidSynth and a General MIDI soundfont, download the datasets listed in `tools/notefilter/real.py`, and run, from the repository folder:

```bash
python -m tools.notefilter.build_synth DATA/C_train_0.npz 80 0
```

```bash
python -m tools.notefilter.real DATA/C_real_0.npz DATASETS
```

```bash
python -m tools.notefilter.train DATA --prefix=C --write
```

The melody model is trained with `python -m tools.notefilter.melody`. Real piano recordings (SMD) carry a non-commercial licence, so they are used only to test, never to train.

## Before you sell it: licences and rights

This is not legal advice — treat it as a checklist to take to a lawyer.

- **The code's dependencies allow commercial use.** The pitch model is Apache-2.0, the high-accuracy piano model CC BY 4.0 and guitar model MIT (attribution required, which THIRD_PARTY_NOTICES.md provides; worth a line in the lawyer's review: like most transcription models they were trained on recordings whose own terms are research-only — MAESTRO for piano, which is non-commercial, and the GAPS performances from YouTube for guitar), the notation renderer BSD-3, the Python libraries MIT/BSD. Attribution notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and must ship with the product. ffmpeg is called as a separate program; check the licence of the build you distribute.
- **Lyrics are copyrighted.** The built-in lookup uses LRCLIB, a volunteer-run database with no licence from publishers. That is fine for trying the feature; a commercial product that displays or prints lyrics needs a licensed provider (Musixmatch, LyricFind and similar), which plugs in through the one-method `LyricsProvider` interface in `chordlistener/server/lyricsdb.py`, or should run with `CL_LYRICS=none`.
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
  notefilter.py the trained note filter: candidate notes, their features, which ones to keep
  melody.py     the trained melody model and the line-finder for the lead sheet
  beats.py      onset strength, tempo, beat tracking, bar finding
  chords.py     beat-synchronous chroma, chord templates, Viterbi decoding, key
  theory.py     note spelling, key signatures, chord vocabulary
  pipeline.py   ties the stages together; the Analysis result type
  tuning.py     how far a recording is from concert pitch, and correcting it
  arrange.py    laying notes out for two hands within reach, grid choice, the lead-sheet line
  guitar.py     capo and tuning suggestions for easier chord shapes
  lyrics.py     LRC parsing, title/artist guessing, placing timed lines into bars
  document.py   the editable score: built from an analysis, validated, corrected by people, laid out
  score.py      notation building blocks: bars, voices, ties, rests, lyric underlay
  musicxml.py   MusicXML writer
  midi.py       MIDI writer
  export.py     MusicXML / MIDI / chord chart from an Analysis
  cli.py        command line
  models/       the pitch model (ONNX) and its licence
  server/       FastAPI app, SQLite store, catalogue and lyrics providers, website
tools/          test-song generator, accuracy evaluation, and (tools/notefilter) the training of both models
tests/
```

### Scaling notes

One container handles a steady stream of requests: an analysis occupies a worker for a few seconds, and repeats are cache hits. The rate limiter and the in-flight de-duplication are in memory, and storage is a single SQLite file, so the current design is one process per data directory. To run several replicas behind a load balancer, move `Store` to PostgreSQL and the rate limiter to Redis; both are isolated behind small classes (`server/store.py`, `RateLimiter` in `server/app.py`) for that reason.
