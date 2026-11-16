# Third-party notices

ChordListener includes or depends on the following third-party work. These notices must accompany any distribution of the product.

## Bundled in this repository

### Basic Pitch model — Apache License 2.0

`chordlistener/models/nmp.onnx` is the ICASSP 2022 "Basic Pitch" note transcription model, unmodified, from <https://github.com/spotify/basic-pitch>.
Copyright 2022 Spotify AB. Licence text: `chordlistener/models/LICENSE-basic-pitch.txt`. Upstream NOTICE: `chordlistener/models/NOTICE-basic-pitch.txt`.

The note-decoding routine in `chordlistener/pitch.py` is a re-implementation of the algorithm in `basic_pitch/note_creation.py` from the same project (Apache License 2.0).

> Rachel M. Bittner, Juan José Bosch, David Rubinstein, Gabriel Meseguer-Brocal, Sebastian Ewert. "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation." ICASSP 2022.

### OpenSheetMusicDisplay — BSD 3-Clause

`chordlistener/server/web/vendor/opensheetmusicdisplay.min.js` (version 2.3.1), from <https://github.com/opensheetmusicdisplay/opensheetmusicdisplay>. It embeds VexFlow (MIT).
Copyright 2019 PhonicScore. Licence text: `chordlistener/server/web/vendor/LICENSE-opensheetmusicdisplay.txt`.

### jsPDF — MIT

`chordlistener/server/web/vendor/jspdf.umd.min.js` (version 4.2.1), from <https://github.com/parallax/jsPDF>.
Copyright (c) 2010-2025 James Hall, (c) 2015-2025 yWorks GmbH. Licence text: `chordlistener/server/web/vendor/LICENSE-jspdf.txt`.

### svg2pdf.js — MIT

`chordlistener/server/web/vendor/svg2pdf.umd.min.js` (version 2.9.0), from <https://github.com/yWorks/svg2pdf.js>.
Copyright (c) 2015-2026 yWorks GmbH, (c) 2013-2015 Vitaly Puzrin. Licence text: `chordlistener/server/web/vendor/LICENSE-svg2pdf.txt`.

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

## Used as an external program

**FFmpeg** decodes audio. It is run as a separate process and is not linked into ChordListener. FFmpeg is licensed under the LGPL 2.1 or later, or the GPL, depending on how the particular build was configured; check the build you ship.

## Algorithms

The beat tracker follows D. Ellis, "Beat Tracking by Dynamic Programming" (2007). Key estimation uses the Krumhansl-Kessler key profiles. Both are implemented from the published descriptions.
