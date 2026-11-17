"""Teach the guitar model steel-string guitar: fine-tune it on GuitarSet, on a GPU.

    python -m tools.guitar_finetune

Downloads what it needs into ./finetune (GuitarSet, about 700 MB, CC BY 4.0;
the starting model, 100 MB, MIT), trains, and writes:

    finetune/guitar_finetuned.onnx   the new model, ready for chordlistener/models/guitar.onnx
    finetune/report.txt              how it did, before and after

Players are kept apart so the numbers are honest:
    00, 01, 02  training
    03          choosing the best moment to stop
    04, 05      the final test (also the app's own guitar test set: never trained on)

Needs PyTorch with CUDA plus: pip install torchlibrosa piano_transcription_inference librosa onnx
(on an RTX 3070 about half an hour). Options: --steps --batch --lr --device --data.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
import sys
import time
import urllib.request
import wave
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chordlistener import highres  # noqa: E402
from tools.notefilter.data import distinct, match  # noqa: E402

RATE = 16000
SEGMENT = 10 * RATE  # the model works on 10-second windows
FRAMES = SEGMENT // 160 + 1  # 1001 frames, 100 a second
TRAIN, VALID, TEST = ("00", "01", "02"), ("03",), ("04", "05")
DOWNLOADS = {
    "annotation.zip": "https://zenodo.org/api/records/3371780/files/annotation.zip/content",
    "audio_mono-mic.zip": "https://zenodo.org/api/records/3371780/files/audio_mono-mic.zip/content",
    "guitar-gaps.pth": "https://huggingface.co/xavriley/midi-transcription-models/resolve/main/guitar-gaps.pth",
}


# ---------------------------------------------------------------- data


def fetch(folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for name, url in DOWNLOADS.items():
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
    for name, sub in (("annotation.zip", "ann"), ("audio_mono-mic.zip", "audio")):
        target = os.path.join(folder, "guitarset", sub)
        if not os.path.isdir(target) or not os.listdir(target):
            with zipfile.ZipFile(os.path.join(folder, name)) as z:
                z.extractall(target)


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    dtype = {2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / np.iinfo(dtype).max
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return highres.resample(audio, rate, RATE)


def excerpts(folder: str, players: tuple[str, ...]) -> list[dict]:
    """GuitarSet excerpts of these players: 16 kHz audio and (start, end, pitch) notes."""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "guitarset", "ann", "*.jams"))):
        name = os.path.basename(path)[:-5]
        wav = os.path.join(folder, "guitarset", "audio", name + "_mic.wav")
        if name[:2] not in players or not os.path.exists(wav):
            continue
        notes = []
        for annotation in json.load(open(path))["annotations"]:
            if annotation["namespace"] == "note_midi":
                notes += [(d["time"], d["time"] + d["duration"], int(round(d["value"]))) for d in annotation["data"]]
        out.append({"name": name, "audio": read_wav(wav), "notes": sorted(notes)})
    return out


def targets(notes, start: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training targets for a 10-second window from `start`, as in Kong et al.:
    onset and offset rise to 1 within 50 ms of the true time (so the model learns
    exact timing), and the frame target is 1 while the note sounds."""
    onset = np.zeros((FRAMES, 88), np.float32)
    offset = np.zeros((FRAMES, 88), np.float32)
    frame = np.zeros((FRAMES, 88), np.float32)
    times = np.arange(FRAMES) / 100.0
    end = start + SEGMENT / RATE
    for t0, t1, pitch in notes:
        key = pitch - 21
        if not 0 <= key < 88 or t1 < start - 0.1 or t0 > end + 0.1:
            continue
        a, b = t0 - start, t1 - start
        frame[max(0, int(round(a * 100))) : max(0, min(FRAMES, int(round(b * 100)) + 1)), key] = 1
        for when, roll in ((a, onset), (b, offset)):
            near = np.abs(times - when) < 0.05
            roll[near, key] = np.maximum(roll[near, key], 1 - np.abs(times[near] - when) / 0.05)
    return onset, offset, frame


def batch(data: list[dict], size: int, rng: random.Random):
    weights = [len(d["audio"]) for d in data]
    audio, on, off, fr = [], [], [], []
    for d in rng.choices(data, weights=weights, k=size):
        x = d["audio"]
        first = rng.randint(0, max(0, len(x) - SEGMENT))
        clip = x[first : first + SEGMENT]
        clip = np.pad(clip, (0, SEGMENT - len(clip))) * rng.uniform(0.3, 1.0)  # random loudness
        o, f, r = targets(d["notes"], first / RATE)
        audio.append(clip)
        on.append(o)
        off.append(f)
        fr.append(r)
    return [np.stack(v) for v in (audio, on, off, fr)]


# ---------------------------------------------------------------- model


def outputs(model, audio: np.ndarray, device: str) -> dict[str, np.ndarray]:
    """Run the model over a whole recording the way the app does (overlapping 10-second windows)."""
    import torch

    step = SEGMENT // 2
    n = max(1, -(-(len(audio) - step) // step)) if len(audio) > SEGMENT else 1
    padded = np.pad(audio, (0, (n - 1) * step + SEGMENT - len(audio)))
    windows = np.stack([padded[i * step : i * step + SEGMENT] for i in range(n)])
    parts = []
    with torch.no_grad():
        for i in range(0, n, 4):  # a few windows at a time, to stay within GPU memory
            parts.append(model(torch.from_numpy(windows[i : i + 4]).to(device)))
    res = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    keys = {"frame": "frame_output", "onset": "reg_onset_output", "offset": "reg_offset_output", "velocity": "velocity_output"}
    frames = len(audio) // 160 + 1
    out = {k: np.zeros(((n - 1) * 500 + 1000, 88), np.float32) for k in keys}
    for k, name in keys.items():
        values = res[name].float().cpu().numpy()[:, :1000]
        for i in range(n):
            lo = 0 if i == 0 else 250
            hi = 1000 if i == n - 1 else 750
            out[k][i * 500 + lo : i * 500 + hi] = values[i, lo:hi]
    return {k: v[:frames] for k, v in out.items()}


def onset_f1(model, data: list[dict], device: str) -> float:
    """Share of notes found with the right pitch within 50 ms (F-measure), as the app decodes them."""
    model.eval()
    hit = found = played = 0
    for d in data:
        notes = highres.decode(outputs(model, d["audio"], device))
        est = [(n.start, n.pitch) for n in notes]
        truth = distinct([(t0, p) for t0, _, p in d["notes"]])
        hit += int(match(est, truth, 0.05).sum())
        found += len(est)
        played += len(truth)
    return 2 * hit / max(found + played, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="finetune")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=4, help="stop after this many checks without improvement")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0, help="use only this many excerpts per group (quick trial)")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    from tools.transcriber_export import export, load_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and not args.limit:
        print("warning: no CUDA GPU found; this will be very slow", file=sys.stderr)
    fetch(args.data)
    print("reading GuitarSet ...", flush=True)
    sets = [excerpts(args.data, p) for p in (TRAIN, VALID, TEST)]
    if args.limit:
        sets = [s[: args.limit] for s in sets]
    train, valid, test = sets
    print(f"{len(train)} training, {len(valid)} validation, {len(test)} test excerpts", flush=True)

    model = load_model(os.path.join(args.data, "guitar-gaps.pth")).to(device)
    for p in model.velocity_model.parameters():
        p.requires_grad = False  # GuitarSet has no loudness labels: leave loudness as it is

    def freeze_norms():
        # Small batches would skew the batch-norm statistics learned from far more data.
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.eval()

    use_amp = device.startswith("cuda")
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(0)

    start_score = best = onset_f1(model, valid, device)
    best_state, best_step, waited = copy.deepcopy(model.state_dict()), 0, 0
    print(f"step 0: validation onset F1 {best:.3f}", flush=True)
    began = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        freeze_norms()
        audio, on, off, fr = (torch.from_numpy(v).to(device) for v in batch(train, args.batch, rng))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(audio)

        def bce(name, target):
            return F.binary_cross_entropy(out[name].float().clamp(1e-7, 1 - 1e-7), target)

        loss = bce("reg_onset_output", on) + bce("reg_offset_output", off) + bce("frame_output", fr)
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.4f} ({(time.time() - began) / step:.2f} s/step)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            score = onset_f1(model, valid, device)
            print(f"step {step}: validation onset F1 {score:.3f} (best {best:.3f})", flush=True)
            if score > best:
                best, best_state, best_step, waited = score, copy.deepcopy(model.state_dict()), step, 0
            else:
                waited += 1
                if waited >= args.patience:
                    print("no longer improving; stopping", flush=True)
                    break

    original = load_model(os.path.join(args.data, "guitar-gaps.pth")).to(device)
    before = onset_f1(original, test, device)
    model.load_state_dict(best_state)
    after = onset_f1(model, test, device)
    onnx_path = os.path.join(args.data, "guitar_finetuned.onnx")
    export(model, onnx_path)
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "step": best_step}, os.path.join(args.data, "guitar_finetuned.pth"))
    report = (
        f"Guitar model fine-tuned on GuitarSet players {', '.join(TRAIN)} (best at step {best_step}).\n"
        f"Validation (player {VALID[0]}): onset F1 {start_score:.3f} -> {best:.3f}\n"
        f"Test (players {', '.join(TEST)}, never trained on): onset F1 {before:.3f} -> {after:.3f}\n"
        f"Model: {onnx_path}\n"
    )
    open(os.path.join(args.data, "report.txt"), "w").write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
