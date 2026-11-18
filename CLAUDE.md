# ChordListener — briefing for new sessions

Audio/video or a song name in → chords, notes, key, tempo, editable sheet music (MusicXML/PDF/MIDI), lyrics. Website + HTTP API + CLI, meant to be sold/licensed to businesses. No LLM at runtime (zero tokens per song).

## Owner
- Not a programmer; explain in plain words, short. Runs it on a Mac (Python 3.13; 3.14+ lacks onnxruntime wheels): `python3.13 -m pip install ".[server]"` then `python3.13 -m chordlistener.cli serve` → http://127.0.0.1:8000.
- Priorities: sheet music a musician can play correctly; fast and cheap to run; ready for ads later (CL_ADS); keep token use low (keep replies brief, avoid dumping big outputs).
- Work on `claude/epic-pasteur-t4fqcn` (the current branch; `claude/tender-sagan-dlasng` is old and lacks high-accuracy mode) unless the session names another; commit + push when done; no PRs unless asked.

## Layout (chordlistener/)
audio.py decode (ffmpeg, video too) · tuning.py concert-pitch check · pitch.py Basic Pitch ONNX + decode_events · highres.py high-accuracy mode (specialist models per instrument: models/piano.onnx ByteDance, models/guitar.onnx Riley GAPS fine-tuned on GuitarSet 00–02; 41 MB each, made by tools/transcriber_export.py) · notefilter.py trained note filter (models/notefilter.json) · melody.py trained melody model (models/melody.json) · beats.py · chords.py · arrange.py two-hand layout, grid, lead line · document.py editable score JSON · score.py/musicxml.py notation · guitar.py capo tips · lyrics.py + server/lyricsdb.py (LRCLIB) · server/ FastAPI app, SQLite store, website in server/web.

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
- guitar (Riley's GAPS model, MIT, fine-tuned 2026-11-18 on GuitarSet players 00–02 on the PC's RTX 3070): combined with the standard notes (highres.combine: unsure standard notes need its confirmation, its sure notes are added). Written score on GuitarSet test players 04–05: F 0.807 → **0.825** (right 83.5 → 84.7%, written 78.1 → 80.4%). Model onset F1 on 04–05: 0.854 → 0.886. Fine-tuned model alone (no combine, AGREEMENT 0.5/0.7/0.85 all same): 0.812, so combination kept. BabySlakh guitar stems (sample-library guitars, re-measured 2026-11-18; standard / old model / fine-tuned): 6 unseen stems 0.784 / 0.803 / 0.809; all 20 stems 0.728 / 0.741 / 0.729 (14 of them were in the note filter's training songs, so standard is favoured there). Kept the fine-tuned model: real recordings matter most.
- `guitar-gaps-paper-version-12200_iterations.pth` (same HF repo) scores 0.91–0.95 on all GuitarSet players: almost certainly trained on GuitarSet, so it can't be tested fairly here; on unseen BabySlakh guitar it was no better than guitar-gaps. Not used.
Weakest: singing (~60%), dense full-band mixes.

## Tried, did not help (don't repeat)
- Wider local-context features (87 instead of 42) + 480 generated songs: better on generated songs, worse on real piano/voice. The note filter is at a plateau (real piano F≈0.81, guitar 0.84); more filter tuning gives ~1 point at most.
- Resolving semitone clashes in chords: no measurable gain.
- Note filter with guitar weighted more (2026-11-17, on the PC; not the guitar-model fine-tune below): retrained on the shipped 42 features with guitar counted 2×/4×/6× (`--features=shipped --weight.guitar=N`, 160 generated train songs). Best (4×) vs shipped filter F: guitar 0.845 vs 0.841, but voice 0.580 vs 0.610, unseen instruments 0.732 vs 0.752, piano 0.830 vs 0.833. Not shipped.

- Guitar model replacing the standard notes (any agreement threshold): worse on real steel-string guitar.
- Piano model on its own for everything: much worse on guitar (F 0.80 → 0.52) and band songs (melody line collapses). Union of both detectors' notes: worse than either alone on guitar.

## Guitar fine-tune (done 2026-11-18, shipped)
`python -m tools.guitar_finetune --batch 4` on the RTX 3070 (Windows: `py -3.13`; torch from the cu126 index, then `pip install -e ".[server,finetune,dev]"`). Train 00–02, early-stop on 03, report on 04–05. 4000 steps, ~6.6 s/step, ~7.5 h; validation onset F1 0.862 → 0.894, still rising slowly at the end (more steps might help a little). Lessons: half precision (fp16 autocast) gives NaN with this model, so it is now off by default (`--amp` to try); batch 6 fills the 8 GB card and runs 2× slower per step than batch 4. Original model backed up locally as finetune/guitar_original.onnx (not in git; git history has it).

## Ideas not yet done
Per-section (not per-song) choice of detector, for songs that are piano only in parts; tuning highres._COMBINE KEEP/ADD for the fine-tuned guitar model; retraining the note filter with the guitar model's activations as extra features (CPU is enough); a vocal-specific model; more full-band training data (full Slakh2100 is ~100 GB).
