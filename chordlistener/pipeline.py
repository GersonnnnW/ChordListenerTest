"""The full analysis: audio in, notes + chords + beat grid out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import ENGINE_VERSION
from .audio import SAMPLE_RATE, load_audio
from .beats import BeatGrid, estimate_tempo, fallback_grid, find_bars, onset_envelope, track_beats
from .chords import ChordRecognizer, estimate_key, segment_chroma
from . import notefilter
from .pitch import FPS, MIDI_OFFSET, Note, decode_notes, run_model
from .theory import QUALITIES, Chord, Key, note_name, spell_pc
from .tuning import estimate_tuning, retune


@dataclass
class AnalysisOptions:
    vocabulary: str = "standard"  # basic | standard | full
    beats_per_bar: int | None = None  # force 3 or 4 instead of detecting
    tempo_scale: float = 1.0  # 0.5 or 2 to count the beat at half / double speed
    max_seconds: float | None = None

    def cache_key(self) -> str:
        return f"{self.vocabulary}:{self.beats_per_bar or 'auto'}:{self.tempo_scale:g}"


@dataclass
class ChordEvent:
    start: float
    end: float
    chord: Chord | None
    confidence: float
    beat: float  # position on the beat grid where the chord starts


@dataclass
class Analysis:
    duration: float
    key: Key
    key_confidence: float
    grid: BeatGrid
    chords: list[ChordEvent]
    notes: list[Note]
    processing_ms: int = 0
    timings: dict = field(default_factory=dict)
    tuning_cents: float = 0.0  # how far the recording sits from concert pitch
    tuning_corrected: bool = False  # whether that was undone before analysing

    @property
    def bpm(self) -> float:
        return self.grid.bpm

    def to_dict(self, transpose: int = 0, include_notes: bool = True) -> dict:
        key = self.key.transposed(transpose)
        grid = self.grid
        chords = []
        for event in self.chords:
            chord = event.chord.transposed(transpose) if event.chord else None
            position = int(round(event.beat)) - grid.first_downbeat
            chords.append(
                {
                    "start": round(event.start, 3),
                    "end": round(event.end, 3),
                    "chord": chord.symbol(key) if chord else "N.C.",
                    "root": spell_pc(chord.root, key) if chord else None,
                    "quality": chord.quality if chord else None,
                    "bass": spell_pc(chord.bass, key) if chord and chord.bass is not None else None,
                    "notes": [spell_pc(pc, key) for pc in chord.pitch_classes()] if chord else [],
                    "confidence": event.confidence,
                    "bar": position // grid.per_bar + 1,
                    "beat": position % grid.per_bar + 1,
                }
            )
        out = {
            "engine_version": ENGINE_VERSION,
            "duration": round(self.duration, 3),
            "tempo_bpm": round(self.bpm, 1),
            "time_signature": f"{grid.per_bar}/4",
            "key": {
                "name": key.name,
                "tonic": spell_pc(key.tonic, key),
                "mode": "minor" if key.minor else "major",
                "confidence": self.key_confidence,
            },
            "transpose": transpose,
            "beats": [round(float(t), 3) for t in grid.times],
            "first_downbeat": grid.first_downbeat,
            "tuning": {
                "cents": round(self.tuning_cents),
                "reference_hz": round(440.0 * 2 ** (self.tuning_cents / 1200), 1),
                "corrected": self.tuning_corrected,
            },
            "chords": chords,
            "processing_ms": self.processing_ms,
        }
        if include_notes:
            out["notes"] = [
                {
                    "start": round(n.start, 3),
                    "end": round(n.end, 3),
                    "pitch": n.pitch + transpose,
                    "name": note_name(n.pitch + transpose, key),
                    "velocity": round(n.velocity, 3),
                    "confidence": round(n.confidence, 2),
                }
                for n in self.notes
                if 0 <= n.pitch + transpose <= 127
            ]
        return out

    # Compact, lossless form used for storage.
    def to_record(self) -> dict:
        return {
            "v": ENGINE_VERSION,
            "duration": self.duration,
            "key": [self.key.tonic, self.key.minor, self.key_confidence],
            "beats": [round(float(t), 4) for t in self.grid.times],
            "per_bar": self.grid.per_bar,
            "first_downbeat": self.grid.first_downbeat,
            "chords": [
                [round(c.start, 4), round(c.end, 4), c.chord.root if c.chord else -1, c.chord.quality if c.chord else "", -1 if not c.chord or c.chord.bass is None else c.chord.bass, c.confidence, c.beat]
                for c in self.chords
            ],
            "notes": [[round(n.start, 4), round(n.end, 4), n.pitch, round(n.velocity, 3), round(n.confidence, 2)] for n in self.notes],
            "ms": self.processing_ms,
            "tuning": [round(self.tuning_cents, 1), self.tuning_corrected],
        }

    @classmethod
    def from_record(cls, rec: dict) -> "Analysis":
        chords = []
        for start, end, root, quality, bass, confidence, beat in rec["chords"]:
            chord = Chord(root, quality, None if bass < 0 else bass) if root >= 0 and quality in QUALITIES else None
            chords.append(ChordEvent(start, end, chord, confidence, beat))
        return cls(
            duration=rec["duration"],
            key=Key(rec["key"][0], bool(rec["key"][1])),
            key_confidence=rec["key"][2],
            grid=BeatGrid(np.array(rec["beats"], dtype=np.float64), rec["per_bar"], rec["first_downbeat"]),
            chords=chords,
            notes=[Note(*n) for n in rec["notes"]],
            processing_ms=rec.get("ms", 0),
            tuning_cents=rec.get("tuning", [0.0, False])[0],
            tuning_corrected=bool(rec.get("tuning", [0.0, False])[1]),
        )


_LEAD_FRAMES = 3


def _extend(beats: np.ndarray, duration: float) -> np.ndarray:
    """Continue the beat grid at a steady pace to cover the whole recording."""
    period = float(np.median(np.diff(beats)))
    before = np.arange(beats[0] - period, -period * 0.5, -period)[::-1]
    after = np.arange(beats[-1] + period, duration + period, period)
    return np.concatenate([before, beats, after])


def _align_to_onsets(beats: np.ndarray, notes: list[Note]) -> np.ndarray:
    """Nudge the beat grid so that notes played on the beat land exactly on it.

    The tracker follows a smoothed loudness curve and sits a few tens of
    milliseconds off the note attacks, enough to tip a note onto the previous
    sixteenth when the score is quantised.
    """
    if len(notes) < 8:
        return beats
    onsets = np.array([n.start for n in notes])
    nearest = np.clip(np.searchsorted(beats, onsets), 1, len(beats) - 1)
    nearest = np.where(np.abs(onsets - beats[nearest - 1]) < np.abs(onsets - beats[nearest]), nearest - 1, nearest)
    offset = onsets - beats[nearest]
    period = float(np.median(np.diff(beats)))
    on_beat = offset[np.abs(offset) < 0.2 * period]
    if len(on_beat) < 8:
        return beats
    return beats + float(np.median(on_beat))


def analyze_samples(samples: np.ndarray, options: AnalysisOptions | None = None) -> Analysis:
    """Analyse mono float32 audio sampled at 22.05 kHz."""
    options = options or AnalysisOptions()
    started = time.perf_counter()
    cents, confidence = estimate_tuning(samples)
    if confidence < 0.35 or abs(cents) < _RETUNE_FROM:
        result = _analyze(samples, options)
        result.tuning_cents = cents if confidence >= 0.35 else 0.0
    else:
        # Bring the recording to concert pitch first, then map every time back.
        shifted, stretch = retune(samples, cents)
        result = _analyze(shifted, options)
        result.tuning_cents, result.tuning_corrected = cents, True
        result.duration = len(samples) / SAMPLE_RATE
        result.grid.times = result.grid.times * stretch
        for note in result.notes:
            note.start, note.end = note.start * stretch, note.end * stretch
        for chord in result.chords:
            chord.start, chord.end = chord.start * stretch, chord.end * stretch
    result.processing_ms = int((time.perf_counter() - started) * 1000)
    return result


_RETUNE_FROM = 15.0  # cents; below this the pitch model copes on its own


def _analyze(samples: np.ndarray, options: AnalysisOptions) -> Analysis:
    started = time.perf_counter()
    timings = {}
    duration = len(samples) / SAMPLE_RATE
    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    if peak > 0:
        samples = samples * (0.9 / peak)

    t0 = time.perf_counter()
    act = run_model(samples)
    timings["model_ms"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    # A trained filter judges each candidate note in context; without it, fall back to fixed thresholds.
    notes = notefilter.filtered_notes(act, samples) if notefilter.available() else decode_notes(act)
    timings["notes_ms"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    env = onset_envelope(samples, act.onsets)
    bpm = float(np.clip(estimate_tempo(env) * options.tempo_scale, 30.0, 300.0))
    beats = track_beats(env, bpm)
    if len(beats) >= 4:
        times = _extend(_align_to_onsets(beats, notes), duration)
    else:
        times = fallback_grid(duration).times
    timings["beats_ms"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    n_frames = len(act.frames)
    bounds = np.clip(np.round(times * FPS), 0, n_frames)
    # Pitch activations start rising a few frames before the audible attack,
    # so each beat's window is read slightly early; otherwise a new chord
    # leaks into the end of the previous beat.
    treble, bass = segment_chroma(act.frames, np.clip(bounds - _LEAD_FRAMES, 0, n_frames))

    # Bar lines: where the harmony changes and the bass lands.
    unit = treble / np.maximum(np.linalg.norm(treble, axis=1, keepdims=True), 1e-9)
    novelty = np.concatenate([[0.0], 1.0 - (unit[1:] * unit[:-1]).sum(axis=1)])
    low = act.onsets[:, : 52 - MIDI_OFFSET].sum(axis=1)
    low_cum = np.concatenate([[0.0], np.cumsum(low)])
    centre = bounds[:-1].astype(int)
    lo_i, hi_i = np.clip(centre - 6, 0, n_frames), np.clip(centre + 8, 0, n_frames)
    accents = low_cum[hi_i] - low_cum[lo_i]
    per_bar, phase = find_bars(novelty, accents, options.beats_per_bar)
    bar_position = (np.arange(len(treble)) - phase) % per_bar

    spans = ChordRecognizer(options.vocabulary).decode(treble, bass, bar_position, per_bar)
    key, key_confidence = estimate_key(treble, spans)

    # Start bar 1 at the bar line at or before the first sound, and stop the
    # grid at the end of the last sounding bar.
    voiced = [s for s in spans if s.chord is not None]
    first_time = min([n.start for n in notes[:1]] + [times[s.start_beat] for s in voiced[:1]], default=0.0)
    last_time = max([n.end for n in notes] + [times[s.end_beat] for s in voiced[-1:]], default=duration)
    first_beat = int(np.searchsorted(times, first_time + 0.05, side="right")) - 1
    origin = first_beat - (first_beat - phase) % per_bar
    shift = max(0, -origin)  # a pickup before the first tracked bar line
    if shift:
        period = float(times[1] - times[0])
        times = np.concatenate([times[0] + period * np.arange(-shift, 0), times])
        origin = 0
    last_beat = int(np.searchsorted(times, last_time - 0.05, side="left"))
    bars = max(1, -(-(last_beat - origin) // per_bar))
    stop = origin + bars * per_bar  # index of the final bar line
    while stop >= len(times):
        times = np.append(times, times[-1] + (times[-1] - times[-2]))

    chords = []
    for i, s in enumerate(spans):
        lo, hi = max(s.start_beat + shift, origin), min(s.end_beat + shift, stop)
        if hi <= lo or (s.chord is None and i in (0, len(spans) - 1)):
            continue  # outside the music, or the silence before and after it
        chords.append(ChordEvent(float(times[lo]), float(times[hi]), s.chord, s.confidence, float(lo - origin)))
    timings["chords_ms"] = int((time.perf_counter() - t0) * 1000)

    grid = BeatGrid(times[origin : stop + 1], per_bar, 0)
    total = int((time.perf_counter() - started) * 1000)
    return Analysis(duration, key, key_confidence, grid, chords, notes, total, timings)


def analyze(path: str, options: AnalysisOptions | None = None) -> Analysis:
    """Analyse an audio file (any format ffmpeg can read)."""
    options = options or AnalysisOptions()
    started = time.perf_counter()
    samples = load_audio(path, options.max_seconds)
    decode_ms = int((time.perf_counter() - started) * 1000)
    result = analyze_samples(samples, options)
    result.timings["decode_ms"] = decode_ms
    result.processing_ms += decode_ms
    return result
