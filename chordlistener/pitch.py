"""Polyphonic note transcription.

Runs Spotify's Basic Pitch network (Apache-2.0, ~17k parameters) through ONNX
Runtime. The same forward pass also yields the pitch-activation matrix the
chord recogniser uses, so the audio is only "listened to" once.

The note decoding below follows the algorithm of basic_pitch.note_creation
(Copyright 2022 Spotify AB, Apache-2.0), re-implemented for speed.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE

HOP = 256
FPS = SAMPLE_RATE / HOP
MIDI_OFFSET = 21
N_PITCHES = 88
_WINDOW = 43844  # samples the network expects
_WINDOW_FRAMES = 172
_TRIM = 16  # frames dropped at each window edge
_STEP_FRAMES = _WINDOW_FRAMES - 2 * _TRIM  # 140 -> windows tile the frame grid exactly
_STEP = _STEP_FRAMES * HOP
_BATCH = 16

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "nmp.onnx")
_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                threads = int(os.environ.get("CL_MODEL_THREADS", "0"))
                if threads > 0:
                    opts.intra_op_num_threads = threads
                opts.log_severity_level = 3
                _session = ort.InferenceSession(_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    return _session


@dataclass
class Activations:
    """Network output on a uniform grid of FPS frames per second."""

    frames: np.ndarray  # (T, 88) probability a pitch is sounding
    onsets: np.ndarray  # (T, 88) probability a note starts
    contour: np.ndarray | None = None  # (T, 264) fine pitch, three steps per semitone from A0


def run_model(samples: np.ndarray) -> Activations:
    session = _get_session()
    n_out = int(len(samples) / HOP)
    padded = np.concatenate([np.zeros(_TRIM * HOP, dtype=np.float32), samples.astype(np.float32, copy=False)])
    n_windows = max(1, int(np.ceil(len(samples) / _STEP)))
    need = (n_windows - 1) * _STEP + _WINDOW
    if len(padded) < need:
        padded = np.concatenate([padded, np.zeros(need - len(padded), dtype=np.float32)])
    frames, onsets, contours = [], [], []
    name = session.get_inputs()[0].name
    for start in range(0, n_windows, _BATCH):
        idx = range(start, min(start + _BATCH, n_windows))
        batch = np.stack([padded[i * _STEP : i * _STEP + _WINDOW] for i in idx])[:, :, None]
        contour, note, onset = session.run(["StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2"], {name: batch})
        frames.append(note[:, _TRIM:-_TRIM].reshape(-1, N_PITCHES))
        onsets.append(onset[:, _TRIM:-_TRIM].reshape(-1, N_PITCHES))
        contours.append(contour[:, _TRIM:-_TRIM].reshape(-1, contour.shape[-1]))
    frames_out, onsets_out = np.concatenate(frames)[:n_out], np.concatenate(onsets)[:n_out]
    contour_out = np.concatenate(contours)[:n_out]
    # The network normalises its input, so on silence it amplifies the noise
    # floor into phantom notes. Mute frames that are ~50 dB below the music.
    audible = _audible(samples, n_out)
    frames_out[~audible] = 0
    onsets_out[~audible] = 0
    contour_out[~audible] = 0
    return Activations(frames_out, onsets_out, contour_out)


def _audible(samples: np.ndarray, n_frames: int) -> np.ndarray:
    usable = samples[: n_frames * HOP].astype(np.float64)
    power = (usable * usable).reshape(n_frames, HOP).mean(axis=1)
    power = np.convolve(power, np.ones(5) / 5, mode="same")
    loud = np.percentile(power, 95) if n_frames else 0.0
    mask = power > max(loud * 1e-5, 1e-10)
    # Keep a short margin so soft attacks and releases survive.
    return np.convolve(mask.astype(np.float64), np.ones(9), mode="same") > 0


@dataclass
class Note:
    start: float  # seconds
    end: float
    pitch: int  # MIDI number
    velocity: float  # 0..1
    confidence: float = 1.0  # how sure the note filter is that this was really played
    technique: str = ""  # guitar: how it was reached from the previous note: h, p, /, \\ (slide up/down) or b (bent up to)
    bend: float = 0.0  # guitar: semitones the note itself is bent up

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "pitch": self.pitch, "velocity": round(self.velocity, 3), "confidence": round(self.confidence, 2)}


def _inferred_onsets(onsets: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Add onsets where a pitch's activation jumps without the onset head firing."""
    d1 = np.diff(frames, n=1, axis=0, prepend=np.zeros((1, frames.shape[1]), frames.dtype))
    shifted = np.concatenate([np.zeros((2, frames.shape[1]), frames.dtype), frames[:-2]])
    diff = np.minimum(d1, frames - shifted)
    np.maximum(diff, 0, out=diff)
    diff[:2] = 0
    peak = diff.max()
    if peak > 0:
        diff *= onsets.max() / peak
    return np.maximum(onsets, diff)


def decode_notes(
    act: Activations,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_seconds: float = 0.11,
    energy_tolerance: int = 11,
    min_amplitude: float = 0.35,
) -> list[Note]:
    """Notes straight from the network's activations, by fixed thresholds."""
    events = decode_events(act, onset_threshold, frame_threshold, min_note_seconds, energy_tolerance)
    # Faint detections are mostly drum hits and overtones mistaken for notes.
    return [Note(t0 / FPS, t1 / FPS, f + MIDI_OFFSET, min(1.0, amp)) for t0, t1, f, amp, _ in events if amp >= min_amplitude]


def decode_events(
    act: Activations,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_seconds: float = 0.11,
    energy_tolerance: int = 11,
) -> list[tuple[int, int, int, float, bool]]:
    """Candidate notes as (first frame, end frame, key 0..87, mean activation, has a clear attack)."""
    frames = act.frames
    n_frames = frames.shape[0]
    if n_frames < 4:
        return []
    min_len = max(1, int(round(min_note_seconds * FPS)))
    onsets = _inferred_onsets(act.onsets, frames)

    is_peak = np.zeros(onsets.shape, dtype=bool)
    is_peak[1:-1] = (onsets[1:-1] > onsets[:-2]) & (onsets[1:-1] > onsets[2:]) & (onsets[1:-1] >= onset_threshold)
    peak_t, peak_f = np.nonzero(is_peak)
    order = np.argsort(-peak_t, kind="stable")  # latest first, so earlier notes stop at later ones

    remaining = frames.copy()
    events: list[tuple[int, int, int, float, bool]] = []
    last = n_frames - 1
    for t0, f in zip(peak_t[order].tolist(), peak_f[order].tolist()):
        if t0 >= last:
            continue
        col = remaining[:, f]
        i, k = t0 + 1, 0
        while i < last and k < energy_tolerance:
            k = k + 1 if col[i] < frame_threshold else 0
            i += 1
        i -= k
        if i - t0 <= min_len:
            continue
        remaining[t0:i, max(f - 1, 0) : f + 2] = 0
        events.append((t0, i, f, float(frames[t0:i, f].mean()), True))

    # Notes with no clear attack (voice, strings, pads): grow outwards from the
    # strongest leftover activation until it fades. Visiting leftovers from
    # strongest to weakest and skipping ones already claimed is the same as
    # repeatedly searching for the maximum, without rescanning the matrix.
    left_t, left_f = np.nonzero(remaining > frame_threshold)
    strongest_first = np.argsort(-remaining[left_t, left_f], kind="stable")
    for mid, f in zip(left_t[strongest_first].tolist(), left_f[strongest_first].tolist()):
        col = remaining[:, f]
        if col[mid] <= frame_threshold:
            continue
        col[mid] = 0
        i, k = mid + 1, 0
        while i < last and k < energy_tolerance:
            k = k + 1 if col[i] < frame_threshold else 0
            i += 1
        end = i - 1 - k
        hi = i
        i, k = mid - 1, 0
        while i > 0 and k < energy_tolerance:
            k = k + 1 if col[i] < frame_threshold else 0
            i -= 1
        begin = i + 1 + k
        remaining[max(i, 0) : hi, max(f - 1, 0) : min(f + 2, N_PITCHES)] = 0
        if end - begin > min_len:
            events.append((begin, end, f, float(frames[begin:end, f].mean()), False))

    events.sort()
    return events
