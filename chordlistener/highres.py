"""High-accuracy mode: large note detectors specialised for one instrument.

Each is a high-resolution transcription network in the design of Kong et al.
(ByteDance, 2021; about 20 million parameters), run through ONNX Runtime:

  piano   ByteDance's model, trained on concert piano recordings
  guitar  Xavier Riley's guitar adaptation of it (Queen Mary University of London)

They are about a thousand times larger than Basic Pitch and make an analysis
about ten times slower, and each knows only its instrument; but on that
instrument they find far more of the notes, with exact attack and release times.

The model files (models/piano.onnx, models/guitar.onnx) are conversions of the
published checkpoints made with tools/transcriber_export.py. The decoding of
their outputs below follows piano_transcription_inference (Copyright 2020
Qiuqiang Kong, MIT licence), re-implemented with numpy so it runs in
milliseconds. Licences and credits: models/NOTICE-*.txt.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .audio import SAMPLE_RATE
from .pitch import Note

RATE = 16000
HOP = 160
FPS = RATE / HOP  # 100 frames a second
N_FFT = 2048
MIDI_OFFSET = 21
_SEGMENT = 1000  # frames per window: the 10 seconds the model was trained on
_BATCH = 4

ONSET_THRESHOLD = 0.3
OFFSET_THRESHOLD = 0.3
FRAME_THRESHOLD = 0.1
_MAX_FRAMES = 600  # a held note is cut after six seconds

INSTRUMENTS = ("piano", "guitar")
_MODELS = {
    name: os.environ.get(f"CL_{name.upper()}_MODEL") or os.path.join(os.path.dirname(__file__), "models", f"{name}.onnx")
    for name in INSTRUMENTS
}
_sessions: dict = {}
_lock = threading.Lock()


def available(instrument: str = "piano") -> bool:
    return instrument in _MODELS and os.path.exists(_MODELS[instrument])


def installed() -> list[str]:
    """The instruments whose model is present."""
    return [name for name in INSTRUMENTS if available(name)]


def _get_session(instrument: str):
    if instrument not in _sessions:
        with _lock:
            if instrument not in _sessions:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                threads = int(os.environ.get("CL_MODEL_THREADS", "0"))
                if threads > 0:
                    opts.intra_op_num_threads = threads
                opts.log_severity_level = 3
                _sessions[instrument] = ort.InferenceSession(_MODELS[instrument], opts, providers=["CPUExecutionProvider"])
    return _sessions[instrument]


def resample(samples: np.ndarray, src: int = SAMPLE_RATE, dst: int = RATE) -> np.ndarray:
    """Band-limited resampling by trimming the spectrum (an ideal low-pass)."""
    n = len(samples)
    if n == 0 or src == dst:
        return samples.astype(np.float32, copy=False)
    m = int(round(n * dst / src))
    spectrum = np.fft.rfft(samples.astype(np.float64))
    keep = min(len(spectrum), m // 2 + 1)
    out = np.fft.irfft(spectrum[:keep], m) * (m / n)
    return out.astype(np.float32)


_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic Hann


def _frames(audio: np.ndarray) -> np.ndarray:
    """Overlapping analysis frames (a view, no copy): centred on every hop, edges reflected."""
    pad = N_FFT // 2
    if len(audio) <= pad:
        audio = np.concatenate([audio, np.zeros(pad + 1 - len(audio), dtype=np.float32)])
    padded = np.pad(audio, pad, mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    return np.lib.stride_tricks.as_strided(padded, (n_frames, N_FFT), (padded.strides[0] * HOP, padded.strides[0]), writeable=False)


def _power(frames: np.ndarray, start: int, count: int) -> np.ndarray:
    """(count, 1025) power spectrum of frames start.., silent past the end."""
    out = np.zeros((count, N_FFT // 2 + 1), dtype=np.float32)
    have = max(0, min(count, len(frames) - start))
    if have:
        spec = np.fft.rfft(frames[start : start + have] * _WINDOW, axis=1)
        out[:have] = spec.real**2 + spec.imag**2
    return out


def run_model(samples: np.ndarray, instrument: str = "piano") -> dict[str, np.ndarray]:
    """The model's four (frames, 88) outputs at 100 frames a second, for 22.05 kHz audio."""
    frames = _frames(resample(samples))
    total = len(frames)
    step = _SEGMENT // 2
    n_windows = max(1, -(-(total - step) // step)) if total > _SEGMENT else 1
    session = _get_session(instrument)
    names = ["frame", "onset", "offset", "velocity"]
    out = {k: np.zeros(((n_windows - 1) * step + _SEGMENT, 88), np.float32) for k in names}
    # Spectra are made window by window, so memory stays flat however long the song.
    for first in range(0, n_windows, _BATCH):
        idx = list(range(first, min(first + _BATCH, n_windows)))
        batch = np.stack([_power(frames, i * step, _SEGMENT) for i in idx])
        results = session.run(names, {"power": batch})
        for k, res in zip(names, results):
            for j, i in enumerate(idx):
                # Keep each window's middle half; the edges lack context on one side.
                lo = 0 if i == 0 else step // 2
                hi = _SEGMENT if i == n_windows - 1 else _SEGMENT - step // 2
                out[k][i * step + lo : i * step + hi] = res[j, lo:hi]
    return {k: v[:total] for k, v in out.items()}


def _peaks(reg: np.ndarray, threshold: float, neighbour: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames where a regression curve tops out, and the sub-frame shift of that peak."""
    n = len(reg)
    hit = np.zeros(reg.shape, dtype=bool)
    shift = np.zeros(reg.shape, dtype=np.float32)
    if n <= 2 * neighbour:
        return hit, shift
    mid = slice(neighbour, n - neighbour)
    ok = reg[mid] > threshold
    d = np.diff(reg, axis=0)  # d[i] = reg[i+1] - reg[i]
    for i in range(neighbour):
        ok &= d[neighbour - i - 1 : n - neighbour - i - 1] >= 0  # rising into the peak
        ok &= d[neighbour + i : n - neighbour + i] <= 0  # falling after it
    hit[mid] = ok
    t, f = np.nonzero(hit)
    if len(t):
        x0, x1, x2 = reg[t - 1, f], reg[t, f], reg[t + 1, f]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(x0 > x2, (x2 - x0) / (x1 - x2), (x2 - x0) / (x1 - x0)) / 2
        shift[t, f] = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return hit, shift


def _next_true(mask: np.ndarray) -> np.ndarray:
    """For every frame, the first frame at or after it where `mask` holds (len(mask) if none)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n)[:, None], n)
    return np.minimum.accumulate(idx[::-1], axis=0)[::-1]


def decode(out: dict[str, np.ndarray]) -> list[Note]:
    """Notes from the model's outputs: each onset peak runs until its release."""
    frame, velocity, onset_reg = out["frame"], out["velocity"], out["onset"]
    n = len(frame)
    if n < 10:
        return []
    onset, onset_shift = _peaks(onset_reg, ONSET_THRESHOLD, 2)
    offset, offset_shift = _peaks(out["offset"], OFFSET_THRESHOLD, 4)
    gone = _next_true(frame <= FRAME_THRESHOLD)
    next_offset = _next_true(offset)
    next_onset = _next_true(onset)
    notes: list[Note] = []
    for bgn, f in zip(*np.nonzero(onset)):
        if bgn == 0 or bgn + 1 >= n:
            continue
        limit = min(bgn + _MAX_FRAMES, n - 1)
        again = next_onset[bgn + 1, f]
        fade = gone[bgn + 1, f]
        if again <= fade and again <= limit:
            fin, fin_shift = again - 1, 0.0  # struck again before it faded
        elif fade <= limit:
            off = next_offset[bgn + 1, f]
            fin = off if off <= fade and off - bgn > fade - off else fade
            fin_shift = float(offset_shift[fin, f])
        else:
            fin, fin_shift = limit, float(offset_shift[limit, f])
        start = (bgn + float(onset_shift[bgn, f])) / FPS
        end = (fin + fin_shift) / FPS
        if end - start < 0.03:
            continue
        sure = float(min(0.99, onset_reg[bgn, f]))
        notes.append(Note(start, end, int(f) + MIDI_OFFSET, float(min(1.0, velocity[bgn, f])), sure))
    notes.sort(key=lambda x: (x.start, x.pitch))
    return notes


# Share of the standard detector's confident notes a specialist must also find
# before its notes are used at all.
#   piano   On real piano recordings the two agree on 81-97% of notes; on guitar,
#           voice and band songs mostly far less, and there the piano model misses
#           or invents notes. 0.85 keeps those exactly as good as standard mode
#           while real piano gains almost all of the improvement.
#   guitar  Below 0.5 the guitar model plainly is not hearing a guitar.
AGREEMENT = {"piano": 0.85, "guitar": 0.5}

# The piano model's notes replace the standard ones outright. The guitar model
# is no better than the standard notes on its own (they make different
# mistakes), so the two are combined instead: a standard note the filter was
# unsure of (confidence below KEEP) stays only if the guitar model heard it too,
# and a note only the guitar model heard is added when it is sure (from ADD).
_COMBINE = {"guitar": (0.7, 0.7)}  # instrument: (KEEP, ADD)
_SAME_NOTE = 0.08  # seconds between two detections of one note


def agreement(standard: list[Note], found: list[Note]) -> float:
    """How many of the standard detector's confident notes a specialist found too (same key, within 0.1 s)."""
    sure = [n for n in standard if n.confidence >= 0.5]
    if not sure:
        return 0.0
    by_key: dict[int, list[float]] = {}
    for n in found:
        by_key.setdefault(n.pitch, []).append(n.start)
    starts = {k: np.array(sorted(v)) for k, v in by_key.items()}
    hit = 0
    for n in sure:
        times = starts.get(n.pitch)
        if times is None:
            continue
        i = np.searchsorted(times, n.start)
        near = [abs(times[j] - n.start) for j in (i - 1, i) if 0 <= j < len(times)]
        hit += bool(near) and min(near) < 0.1
    return hit / len(sure)


def _starts_by_key(notes: list[Note]) -> dict[int, np.ndarray]:
    by_key: dict[int, list[float]] = {}
    for n in notes:
        by_key.setdefault(n.pitch, []).append(n.start)
    return {k: np.array(sorted(v)) for k, v in by_key.items()}


def _heard(starts: dict[int, np.ndarray], note: Note) -> bool:
    times = starts.get(note.pitch)
    if times is None:
        return False
    i = np.searchsorted(times, note.start)
    return any(abs(times[j] - note.start) < _SAME_NOTE for j in (i - 1, i) if 0 <= j < len(times))


def combine(standard: list[Note], found: list[Note], keep: float, add: float) -> list[Note]:
    """The standard notes checked against a specialist's, plus the ones only it is sure of."""
    theirs, ours = _starts_by_key(found), _starts_by_key(standard)
    notes = [n for n in standard if n.confidence >= keep or _heard(theirs, n)]
    notes += [n for n in found if n.confidence >= add and not _heard(ours, n)]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def merge(instrument: str, standard: list[Note], found: list[Note]) -> list[Note] | None:
    """The notes to write in high-accuracy mode, or None when the music is not this instrument."""
    if agreement(standard, found) < AGREEMENT[instrument]:
        return None
    if instrument in _COMBINE:
        return combine(standard, found, *_COMBINE[instrument])
    return found


def transcribe(samples: np.ndarray, instrument: str = "piano") -> list[Note]:
    """Notes played in mono 22.05 kHz audio, by the instrument's specialist model."""
    return decode(run_model(samples, instrument))
