# Third-party notices

ChordListener includes or depends on the following third-party work. These notices must accompany any distribution of the product.

## Bundled in this repository

### Basic Pitch model — Apache License 2.0

`chordlistener/models/nmp.onnx` is the ICASSP 2022 "Basic Pitch" note transcription model, unmodified, from <https://github.com/spotify/basic-pitch>.
Copyright 2022 Spotify AB. Licence text: `chordlistener/models/LICENSE-basic-pitch.txt`. Upstream NOTICE: `chordlistener/models/NOTICE-basic-pitch.txt`.

The note-decoding routine in `chordlistener/pitch.py` is a re-implementation of the algorithm in `basic_pitch/note_creation.py` from the same project (Apache License 2.0).

> Rachel M. Bittner, Juan José Bosch, David Rubinstein, Gabriel Meseguer-Brocal, Sebastian Ewert. "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation." ICASSP 2022.

### High-resolution piano transcription model — CC BY 4.0 (weights), MIT (decoding)

`chordlistener/models/piano.onnx`, used by the optional high-accuracy mode, is converted from the checkpoint `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth` published at <https://zenodo.org/records/4034264> under CC BY 4.0: only the note model is kept, converted to ONNX with 16-bit weights (`tools/transcriber_export.py`). The output decoding in `chordlistener/highres.py` re-implements <https://github.com/qiuqiangkong/piano_transcription_inference> (MIT, Copyright (c) 2020 Qiuqiang Kong). Details and licence text: `chordlistener/models/NOTICE-piano-transcription.txt`.

> Qiuqiang Kong, Bochen Li, Xuchen Song, Yuan Wan, Yuxuan Wang. "High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times." IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021.

### High-resolution guitar transcription model — MIT

`chordlistener/models/guitar.onnx`, used by the optional high-accuracy mode for guitar, is converted from `guitar-gaps.pth` at <https://huggingface.co/xavriley/midi-transcription-models> (MIT, Xavier Riley): fine-tuned on GuitarSet players 00–02 (CC BY 4.0, see below; `tools/guitar_finetune.py`), then converted to ONNX with 16-bit weights (`tools/transcriber_export.py`). Details and licence text: `chordlistener/models/NOTICE-guitar-transcription.txt`.

> Xavier Riley, Zixun Guo, Drew Edwards, Simon Dixon. "GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model." ISMIR 2024.

### OpenSheetMusicDisplay — BSD 3-Clause

`chordlistener/server/web/vendor/opensheetmusicdisplay.min.js` (version 2.3.1), from <https://github.com/opensheetmusicdisplay/opensheetmusicdisplay>. It embeds VexFlow (MIT).
Copyright 2019 PhonicScore. Licence text: `chordlistener/server/web/vendor/LICENSE-opensheetmusicdisplay.txt`.

### jsPDF — MIT

`chordlistener/server/web/vendor/jspdf.umd.min.js` (version 4.2.1), from <https://github.com/parallax/jsPDF>.
Copyright (c) 2010-2025 James Hall, (c) 2015-2025 yWorks GmbH. Licence text: `chordlistener/server/web/vendor/LICENSE-jspdf.txt`.

### svg2pdf.js — MIT

`chordlistener/server/web/vendor/svg2pdf.umd.min.js` (version 2.9.0), from <https://github.com/yWorks/svg2pdf.js>.
Copyright (c) 2015-2026 yWorks GmbH, (c) 2013-2015 Vitaly Puzrin. Licence text: `chordlistener/server/web/vendor/LICENSE-svg2pdf.txt`.

### Training data for the note filter and melody models — CC BY 4.0

`chordlistener/models/notefilter.json`, `chordlistener/models/melody.json`, `chordlistener/models/tabs.json` (guitar fingering habits, from GuitarSet) and `chordlistener/models/chords.json` (chord recogniser, from the timed chord labels of GuitarSet, the Schubert Winterreise Dataset and AAM, below) are original models, trained by this project on songs it generated itself and on these datasets:

- **GuitarSet** — Q. Xi, R. Bittner, J. Pauwels, X. Ye, J. P. Bello, "GuitarSet: A Dataset for Guitar Transcription", ISMIR 2018. <https://zenodo.org/records/3371780>, CC BY 4.0.
- **vocadito** — R. Bittner, K. Pasalo, J. J. Bosch, G. Meseguer-Brocal, D. Rubinstein, "vocadito: A dataset of solo vocals with f0, note, and lyric annotations", 2021. <https://zenodo.org/records/5578807>, CC BY 4.0.
- **BabySlakh** — E. Manilow, G. Wichern, P. Seetharaman, J. Le Roux, "Cutting Music Source Separation Some Slakh", WASPAA 2019. <https://zenodo.org/records/4603870>, CC BY 4.0.

- **Schubert Winterreise Dataset** — C. Weiß, F. Zalkow, V. Arifi-Müller, M. Müller, H. V. Koops, A. Volk, H. G. Grohganz, "Schubert Winterreise Dataset: A Multimodal Scenario for Music Analysis", ACM Journal on Computing and Cultural Heritage, 2021. <https://zenodo.org/records/10839767>, CC BY 3.0 (performance SC06 by R. Scarlata and J. Denk, Isabella Stewart Gardner Museum).
- **AAM: Artificial Audio Multitracks** — F. Ostermann, I. Vatolkin, M. Ebeling, "AAM: a dataset of Artificial Audio Multitracks for diverse music information retrieval tasks", EURASIP Journal on Audio, Speech, and Music Processing, 2023. <https://zenodo.org/records/5794629>, CC BY 4.0.

The **Saarland Music Data (SMD)** piano recordings (CC BY-NC-SA 3.0) were used only to measure accuracy. They are not in the models or in this repository.

## Installed as dependencies

| Package | Licence |
|---|---|
| NumPy | BSD 3-Clause |
| ONNX Runtime | MIT |
| FastAPI | MIT |
| Starlette | BSD 3-Clause |
| Uvicorn | BSD 3-Clause |
| python-multipart | Apache 2.0 |
| HTTPX | BSD 3-Clause |
| scikit-learn (training only, not shipped) | BSD 3-Clause |

## Used as an external program

**FFmpeg** decodes audio. It is run as a separate process and is not linked into ChordListener. FFmpeg is licensed under the LGPL 2.1 or later, or the GPL, depending on how the particular build was configured; check the build you ship.

## Algorithms

The beat tracker follows D. Ellis, "Beat Tracking by Dynamic Programming" (2007). Key estimation uses the Krumhansl-Kessler key profiles. Both are implemented from the published descriptions.
