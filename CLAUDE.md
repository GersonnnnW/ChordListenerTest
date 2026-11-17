# ChordListener — briefing for new sessions

Audio/video or a song name in → chords, notes, key, tempo, editable sheet music (MusicXML/PDF/MIDI), lyrics. Website + HTTP API + CLI, meant to be sold/licensed to businesses. No LLM at runtime (zero tokens per song).

## Owner
- Not a programmer; explain in plain words, short. Runs it on a Mac (Python 3.13; 3.14+ lacks onnxruntime wheels): `python3.13 -m pip install ".[server]"` then `python3.13 -m chordlistener.cli serve` → http://127.0.0.1:8000.
- Priorities: sheet music a musician can play correctly; fast and cheap to run; ready for ads later (CL_ADS); keep token use low (keep replies brief, avoid dumping big outputs).
- Work on branch `claude/tender-sagan-dlasng`; commit + push when done; no PRs unless asked.

## Layout (chordlistener/)
audio.py decode (ffmpeg, video too) · tuning.py concert-pitch check · pitch.py Basic Pitch ONNX + decode_events · notefilter.py trained note filter (models/notefilter.json) · melody.py trained melody model (models/melody.json) · beats.py · chords.py · arrange.py two-hand layout, grid, lead line · document.py editable score JSON · score.py/musicxml.py notation · guitar.py capo tips · lyrics.py + server/lyricsdb.py (LRCLIB) · server/ FastAPI app, SQLite store, website in server/web.

## Commands
- Tests: `python -m pytest -q` (all must pass; CI on Python 3.10 + 3.13 — avoid 3.12-only syntax).
- Chords/beats/key: `python -m tools.evaluate`
- Written sheet music vs truth: `python -m tools.evaluate_sheet --real DATASETS --songs 20`
- Retrain note filter: build data with `tools.notefilter.build_synth` and `tools.notefilter.real`, then `python -m tools.notefilter.train DATA --prefix=X --write`. Melody: `python -m tools.notefilter.melody`. Bump `ENGINE_VERSION` in `chordlistener/__init__.py` whenever output changes.
- Needs FluidSynth + FluidR3_GM soundfont for generated songs (`apt-get install fluidsynth fluid-soundfont-gm`).

## Training data (re-download; scratch space is not durable)
GuitarSet https://zenodo.org/records/3371780 (annotation.zip, audio_mono-mic.zip) · vocadito https://zenodo.org/records/5578807 · BabySlakh https://zenodo.org/records/4603870 — all CC BY 4.0, may train. SMD piano (audiolabs-erlangen.de/resources/MIR/SMD/midi) is CC BY-NC-SA: **test only, never train**. Layout expected by tools/notefilter/real.py: DATASETS/{guitarset/{ann,audio},vocadito,slakh/babyslakh_16k,smd/files}.

## Accuracy now (written piano score, right / played-notes-written)
Real piano 89% / 73% · real guitar 83% / 77% · generated 89% / 78%. Melody line 77–95% right. Chord roots 99%.
Ceiling: the Basic Pitch detector itself finds only 85–95% of played notes, so recall cannot exceed that without a stronger pitch model. Weakest: singing (~60%), dense full-band mixes.

## Tried, did not help (don't repeat)
- Wider local-context features (87 instead of 42) + 480 generated songs: better on generated songs, worse on real piano/voice. The note filter is at a plateau (real piano F≈0.81, guitar 0.84); more filter tuning gives ~1 point at most.
- Resolving semitone clashes in chords: no measurable gain.

## Ideas not yet done
Stronger (piano-specific) transcription model as an optional high-accuracy mode; vocal-specific handling; more full-band training data (full Slakh2100 is ~100 GB).
