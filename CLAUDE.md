# ChordListener — briefing for new sessions

Audio/video or a song name in → chords, notes, key, tempo, editable sheet music (MusicXML/PDF/MIDI), lyrics. Website + HTTP API + CLI, meant to be sold/licensed to businesses. No LLM at runtime (zero tokens per song).

## Owner
- Not a programmer; explain in plain words, short. Runs it on a Mac (Python 3.13; 3.14+ lacks onnxruntime wheels): `python3.13 -m pip install ".[server]"` then `python3.13 -m chordlistener.cli serve` → http://127.0.0.1:8000.
- Priorities: sheet music a musician can play correctly; fast and cheap to run; ready for ads later (CL_ADS); keep token use low (keep replies brief, avoid dumping big outputs).
- Work on the branch the session names (earlier: `claude/tender-sagan-dlasng`, `claude/epic-pasteur-t4fqcn`); commit + push when done; no PRs unless asked.

## Layout (chordlistener/)
audio.py decode (ffmpeg, video too) · tuning.py concert-pitch check · pitch.py Basic Pitch ONNX + decode_events · highres.py high-accuracy mode (specialist models per instrument: models/piano.onnx ByteDance, models/guitar.onnx Riley GAPS; 41 MB each, made by tools/transcriber_export.py) · notefilter.py trained note filter (models/notefilter.json) · melody.py trained melody model (models/melody.json) · beats.py · chords.py · arrange.py two-hand layout, grid, lead line · document.py editable score JSON · score.py/musicxml.py notation · guitar.py capo tips · lyrics.py + server/lyricsdb.py (LRCLIB) · server/ FastAPI app, SQLite store, website in server/web.

## Commands
- Tests: `python -m pytest -q` (all must pass; CI on Python 3.10 + 3.13 — avoid 3.12-only syntax).
- Chords/beats/key: `python -m tools.evaluate`
- Written sheet music vs truth: `python -m tools.evaluate_sheet --real DATASETS --songs 20`
- Retrain note filter: build data with `tools.notefilter.build_synth` and `tools.notefilter.real`, then `python -m tools.notefilter.train DATA --prefix=X --write`. Melody: `python -m tools.notefilter.melody`. Bump `ENGINE_VERSION` in `chordlistener/__init__.py` whenever output changes.
- Needs FluidSynth + FluidR3_GM soundfont for generated songs (`apt-get install fluidsynth fluid-soundfont-gm`).

## Training data (re-download; scratch space is not durable)
GuitarSet https://zenodo.org/records/3371780 (annotation.zip, audio_mono-mic.zip) · vocadito https://zenodo.org/records/5578807 · BabySlakh https://zenodo.org/records/4603870 — all CC BY 4.0, may train. SMD piano (audiolabs-erlangen.de/resources/MIR/SMD/midi) is CC BY-NC-SA: **test only, never train**. Layout expected by tools/notefilter/real.py: DATASETS/{guitarset/{ann,audio},vocadito,slakh/babyslakh_16k,smd/files}.

## Accuracy now (written piano score, right / played-notes-written)
Standard: real piano 89% / 73% · real guitar 83% / 77% · generated 89% / 78%. Melody line 77–95% right. Chord roots 99%.
High-accuracy mode (`accuracy=high`): real piano 95% / 93% (F 0.82 → 0.94). The piano model is used only when it agrees with the standard notes on ≥85% (piano.AGREEMENT); otherwise standard notes are kept (guitar/band unchanged). ~10× slower (3-min song ≈ 23 s on 4 cores). Evaluate: `python -m tools.evaluate_sheet --real DATASETS --accuracy high`.
Weakest: singing (~60%), dense full-band mixes.

## Tried, did not help (don't repeat)
- Wider local-context features (87 instead of 42) + 480 generated songs: better on generated songs, worse on real piano/voice. The note filter is at a plateau (real piano F≈0.81, guitar 0.84); more filter tuning gives ~1 point at most.
- Resolving semitone clashes in chords: no measurable gain.

- Piano model on its own for everything: much worse on guitar (F 0.80 → 0.52) and band songs (melody line collapses). Union of both detectors' notes: worse than either alone on guitar.

## Ideas not yet done
Per-section (not per-song) choice of detector, for songs that are piano only in parts; a guitar- or vocal-specific model the same way; more full-band training data (full Slakh2100 is ~100 GB).
