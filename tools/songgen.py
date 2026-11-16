"""Generate songs with known chords and notes, for tests and accuracy checks.

A song is built from a chord progression (piano or guitar accompaniment, bass,
melody, drums) and rendered either with a small built-in numpy synth (no
dependencies; used by the unit tests) or with FluidSynth and a General MIDI
soundfont (realistic timbres; used by tools/evaluate.py).
"""

from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np

from chordlistener.midi import MidiTrack, write_midi
from chordlistener.theory import QUALITIES

SR = 22050
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SUFFIX = {q.suffix: q.id for q in QUALITIES.values()}
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def parse_chord(symbol: str) -> tuple[int, str]:
    m = re.fullmatch(r"([A-G])([#b]?)(.*)", symbol)
    if not m or m.group(3) not in _SUFFIX:
        raise ValueError(f"Unknown chord symbol: {symbol}")
    root = (_PC[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    return root, _SUFFIX[m.group(3)]


@dataclass
class Song:
    bpm: float
    beats_per_bar: int
    key_tonic: int
    key_minor: bool
    chords: list[tuple[float, float, int, str]]  # start_s, end_s, root, quality
    tracks: list[MidiTrack] = field(default_factory=list)
    beats: list[float] = field(default_factory=list)
    duration: float = 0.0

    @property
    def pitched_notes(self) -> list[tuple[float, float, int]]:
        return sorted((s, e, p) for t in self.tracks if t.channel != 9 for s, e, p, _ in t.notes)

    def midi_bytes(self) -> bytes:
        return write_midi(self.tracks, bpm=self.bpm, time_signature=(self.beats_per_bar, 4))


STYLES = {
    #            comp program, bass program, lead program, comp pattern
    "piano": (0, 32, 73, "block"),
    "guitar": (25, 33, 65, "strum"),
    "epiano": (4, 33, 56, "arp"),
    "organ": (16, 32, 40, "sustain"),
}


def make_song(
    progression: list[tuple[str, int]],
    bpm: float = 110.0,
    beats_per_bar: int = 4,
    key: str = "C",
    style: str = "piano",
    repeats: int = 2,
    seed: int = 0,
    drums: bool = True,
    melody: bool = True,
    lead_in: float = 0.5,
) -> Song:
    """Build a song. `progression` is a list of (chord symbol, length in beats)."""
    rng = random.Random(seed)
    key_minor = key.endswith("m")
    key_tonic = parse_chord(key[:-1] if key_minor else key)[0]
    scale = [(key_tonic + d) % 12 for d in (_MINOR_SCALE if key_minor else _MAJOR_SCALE)]
    comp_prog, bass_prog, lead_prog, pattern = STYLES[style]
    beat = 60.0 / bpm

    comp: list = []
    bass: list = []
    lead: list = []
    kit: list = []
    chords: list = []
    t = lead_in
    n_beat = 0
    lead_pitch = 72
    for _ in range(repeats):
        for symbol, length in progression:
            root, quality = parse_chord(symbol)
            tones = [(root + i) % 12 for i in QUALITIES[quality].intervals]
            start, end = t, t + length * beat
            chords.append((start, end, root, quality))
            # Close-position voicing around middle C.
            voicing = sorted(48 + ((pc - 48) % 12) + (12 if (pc - 48) % 12 < 4 else 0) for pc in tones)
            bass_note = 36 + (root - 36) % 12
            if bass_note > 45:
                bass_note -= 12
            for b in range(length):
                bt = t + b * beat
                on_bar = (n_beat + b) % beats_per_bar == 0
                if pattern == "block":
                    if b % 2 == 0 or rng.random() < 0.35:
                        for p in voicing:
                            comp.append((bt, bt + beat * 0.9, p, rng.randint(62, 80)))
                    if rng.random() < 0.3:
                        for p in voicing[1:]:
                            comp.append((bt + beat / 2, bt + beat * 0.95, p, rng.randint(50, 64)))
                elif pattern == "strum":
                    for half in (0, 1):
                        if half == 0 or rng.random() < 0.7:
                            notes = voicing if half == 0 else voicing[::-1]
                            for j, p in enumerate(notes):
                                st = bt + half * beat / 2 + j * 0.012
                                comp.append((st, st + beat * 0.45, p + (12 if j == 0 and half else 0), rng.randint(58, 78)))
                elif pattern == "arp":
                    order = voicing + voicing[-2:0:-1]
                    for j in range(2):
                        p = order[(b * 2 + j) % len(order)]
                        st = bt + j * beat / 2
                        comp.append((st, st + beat * 0.8, p, rng.randint(60, 76)))
                    if b == 0:
                        for p in voicing:
                            comp.append((bt, bt + min(length, 2) * beat * 0.95, p, 54))
                elif pattern == "sustain" and b == 0:
                    for p in voicing:
                        comp.append((bt, end - 0.03, p, 66))

                if b == 0 or on_bar:
                    bass.append((bt, bt + beat * 0.9, bass_note, 84))
                elif rng.random() < 0.5:
                    fifth = bass_note + 7 if quality not in ("dim", "aug", "m7b5", "dim7") else bass_note
                    bass.append((bt, bt + beat * 0.8, rng.choice([bass_note, fifth, bass_note + 12]), 72))

                if melody:
                    slots = [(0.0, 1.0)] if rng.random() < 0.45 else [(0.0, 0.5), (0.5, 0.5)]
                    for offset, dur in slots:
                        if rng.random() < 0.15:
                            continue
                        strong = offset == 0.0
                        pool = [p for p in range(lead_pitch - 5, lead_pitch + 6) if 64 <= p <= 86]
                        chord_tones = [p for p in pool if p % 12 in tones]
                        scale_tones = [p for p in pool if p % 12 in scale]
                        choices = chord_tones if (strong or rng.random() < 0.5) and chord_tones else (scale_tones or pool)
                        lead_pitch = rng.choice(choices)
                        st = bt + offset * beat
                        lead.append((st, st + dur * beat * 0.92, lead_pitch, rng.randint(78, 96)))

                if drums:
                    pos = (n_beat + b) % beats_per_bar
                    if pos == 0:
                        kit.append((bt, bt + 0.1, 36, 100))
                    elif (beats_per_bar == 4 and pos == 2) or (beats_per_bar == 3 and pos > 0):
                        kit.append((bt, bt + 0.1, 38, 90))
                    elif beats_per_bar == 4:
                        kit.append((bt, bt + 0.1, 38 if pos == 3 and rng.random() < 0.2 else 36, 70) if rng.random() < 0.3 else (bt, bt + 0.05, 42, 60))
                    kit.append((bt, bt + 0.05, 42, 64))
                    kit.append((bt + beat / 2, bt + beat / 2 + 0.05, 42, 48))
            t = end
            n_beat += length

    tracks = [MidiTrack(comp, comp_prog, 0, "Comp"), MidiTrack(bass, bass_prog, 1, "Bass")]
    if melody:
        tracks.append(MidiTrack(lead, lead_prog, 2, "Lead"))
    if drums:
        tracks.append(MidiTrack(kit, 0, 9, "Drums"))
    beats = [lead_in + i * beat for i in range(n_beat)]
    return Song(bpm, beats_per_bar, key_tonic, key_minor, chords, tracks, beats, t + 1.0)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_numpy(song: Song, seed: int = 0) -> np.ndarray:
    """Tiny additive synth: decaying harmonic tones plus noise drums."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(song.duration * SR) + SR, dtype=np.float32)
    for track in song.tracks:
        for start, end, pitch, velocity in track.notes:
            i0 = int(start * SR)
            gain = velocity / 127.0
            if track.channel == 9:
                n = int(0.12 * SR)
                tt = np.arange(n) / SR
                if pitch == 36:
                    sig = np.sin(2 * np.pi * (50 * tt + 40 * (1 - np.exp(-tt * 40)) / 40)) * np.exp(-tt * 28)
                else:
                    sig = rng.standard_normal(n) * np.exp(-tt * (70 if pitch == 42 else 30))
                    if pitch == 42:
                        sig = np.diff(sig, prepend=0.0)
                    sig *= 0.5
            else:
                n = int((end - start + 0.15) * SR)
                tt = np.arange(n) / SR
                f0 = 440.0 * 2 ** ((pitch - 69) / 12)
                sustained = track.program >= 16
                sig = np.zeros(n)
                for h in range(1, 9):
                    if f0 * h > SR * 0.45:
                        break
                    decay = 0.4 if sustained else 1.6 + 0.5 * h
                    sig += np.sin(2 * np.pi * f0 * h * tt) / h**1.3 * np.exp(-tt * decay)
                env = np.minimum(1.0, tt / 0.008) * np.minimum(1.0, np.maximum(0.0, (end - start + 0.15 - tt) / 0.15))
                sig *= env * 0.3
            out[i0 : i0 + len(sig)] += (sig * gain).astype(np.float32)[: len(out) - i0]
    peak = np.abs(out).max()
    return out / peak * 0.8 if peak > 0 else out


def find_soundfont() -> str | None:
    for path in (os.environ.get("CL_SOUNDFONT"), "/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/sounds/sf2/default-GM.sf2"):
        if path and os.path.exists(path):
            return path
    return None


def fluidsynth_available() -> bool:
    return shutil.which("fluidsynth") is not None and find_soundfont() is not None


def render_fluidsynth(song: Song, wav_path: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as fh:
        fh.write(song.midi_bytes())
        midi_path = fh.name
    try:
        subprocess.run(
            ["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", "44100", "-F", wav_path, find_soundfont(), midi_path],
            check=True,
            capture_output=True,
            timeout=300,
        )
    finally:
        os.unlink(midi_path)


def write_wav(path: str, samples: np.ndarray, sr: int = SR) -> None:
    import wave

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())
