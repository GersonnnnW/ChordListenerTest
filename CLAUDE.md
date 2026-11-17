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
High-accuracy mode (`accuracy=piano|guitar`, `high` = piano), ~10× slower (3-min song ≈ 23 s on 4 cores). Evaluate: `python -m tools.evaluate_sheet --real DATASETS --accuracy piano`.
- piano: real piano 95% / 93% (F 0.82 → 0.94). Used only when it agrees with the standard notes on ≥85% (highres.AGREEMENT); otherwise standard notes kept (guitar/band unchanged).
- guitar (Riley's GAPS model, trained on classical guitar, MIT): alone it is NOT better than standard (GuitarSet F 0.775 vs 0.799). So it is combined (highres.combine: unsure standard notes need its confirmation, its sure notes are added): GuitarSet 0.799 → 0.807, unseen BabySlakh guitar stems 0.784 → 0.803 (only 6 stems), all 20 stems 0.728 → 0.741. Small gain only.
- `guitar-gaps-paper-version-12200_iterations.pth` (same HF repo) scores 0.91–0.95 on all GuitarSet players: almost certainly trained on GuitarSet, so it can't be tested fairly here; on unseen BabySlakh guitar it was no better than guitar-gaps. Not used.
Weakest: singing (~60%), dense full-band mixes.

## Tried, did not help (don't repeat)
- Wider local-context features (87 instead of 42) + 480 generated songs: better on generated songs, worse on real piano/voice. The note filter is at a plateau (real piano F≈0.81, guitar 0.84); more filter tuning gives ~1 point at most.
- Resolving semitone clashes in chords: no measurable gain.

- Guitar model replacing the standard notes (any agreement threshold): worse on real steel-string guitar.
- Piano model on its own for everything: much worse on guitar (F 0.80 → 0.52) and band songs (melody line collapses). Union of both detectors' notes: worse than either alone on guitar.

## Guitar fine-tune on the owner's PC (RTX 3070) — ready, not yet run
Cloud sessions have no GPU. On the PC, from the repo folder (Windows: `py -3.13` for `python`):
1. `python -m pip install torch --index-url https://download.pytorch.org/whl/cu126` then `python -m pip install -e ".[server,finetune,dev]"`; check `python -c "import torch; print(torch.cuda.is_available())"` is True.
2. `python -m tools.guitar_finetune` (downloads GuitarSet + starting model into ./finetune, ~0.5 h). Train players 00–02, stop on 03, report on 04–05 (the app's guitar test players; never train on them). If GPU memory runs out: `--batch 4`.
3. If finetune/report.txt shows a clear test gain: copy finetune/guitar_finetuned.onnx over chordlistener/models/guitar.onnx, then re-measure the written score (`python -m tools.evaluate_sheet --real DATASETS --real-only --accuracy guitar`, DATASETS/guitarset can be ./finetune/guitarset) against 0.807. A model that is now better alone may beat the combination: try `highres._COMBINE.pop("guitar")` with AGREEMENT 0.5–0.85 too. Update NOTICE-guitar-transcription.txt (fine-tuned on GuitarSet, CC BY 4.0), README, this file; bump ENGINE_VERSION; commit + push.

## Ideas not yet done
Per-section (not per-song) choice of detector, for songs that are piano only in parts; a real guitar jump needs fine-tuning the guitar model on GuitarSet players 00–03 (needs a GPU), or retraining the note filter with the guitar model's activations as extra features (CPU is enough); a vocal-specific model; more full-band training data (full Slakh2100 is ~100 GB).
