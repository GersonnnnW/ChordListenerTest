"""Convert a high-resolution transcription checkpoint (Kong et al.'s design) to ONNX.

One-off, developer-only (needs PyTorch, torchlibrosa and piano_transcription_inference).
The app itself runs the result with ONNX Runtime and never needs PyTorch.

    curl -L -o piano.pth "https://zenodo.org/api/records/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth/content"
    python -m tools.transcriber_export piano.pth chordlistener/models/piano.onnx

    curl -L -o guitar.pth "https://huggingface.co/xavriley/midi-transcription-models/resolve/main/guitar-gaps.pth"
    python -m tools.transcriber_export guitar.pth chordlistener/models/guitar.onnx

Checkpoints are read with PyTorch's weights-only loader, so a downloaded file
cannot run code. Only the note part is kept (a pedal model is dropped). The
spectrogram is computed outside the network (chordlistener/piano.py), so the
graph takes a power spectrogram (batch, frames, 1025) and maps it to mel bands
itself. Weights are stored as float16 and widened to float32 when the model
loads, which halves the file without changing the results measurably.
"""

from __future__ import annotations

import sys


def main(checkpoint: str, out: str) -> None:
    import numpy as np
    import onnx
    import torch
    from onnx import numpy_helper
    from piano_transcription_inference.models import Regress_onset_offset_frame_velocity_CRNN

    import codecs

    from numpy._core.multiarray import _reconstruct

    # Some checkpoints carry numpy arrays (training-sampler state); allow just those types.
    numpy_types = [type(np.dtype(t)) for t in ("float32", "float64", "int64", "int32")]
    torch.serialization.add_safe_globals([(_reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype, codecs.encode, *numpy_types])
    full = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=100, classes_num=88)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    state = state.get("note_model", state)
    missing, unexpected = full.load_state_dict(state, strict=False)
    assert not [k for k in missing if "spectrogram" not in k and "logmel" not in k], missing
    full.eval()
    mel = full.logmel_extractor.melW.detach().clone()  # (1025, 229)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = full
            self.register_buffer("mel", mel)

        def forward(self, power):  # (B, T, 1025)
            x = torch.matmul(power, self.mel)
            x = 10.0 * torch.log10(torch.clamp(x, min=1e-10))
            x = x[:, None]  # (B, 1, T, 229)
            m = self.m
            x = m.bn0(x.transpose(1, 3)).transpose(1, 3)
            frame = m.frame_model(x)
            onset = m.reg_onset_model(x)
            offset = m.reg_offset_model(x)
            velocity = m.velocity_model(x)
            y, _ = m.reg_onset_gru(torch.cat((onset, (onset**0.5) * velocity), dim=2))
            onset = torch.sigmoid(m.reg_onset_fc(y))
            y, _ = m.frame_gru(torch.cat((frame, onset, offset), dim=2))
            frame = torch.sigmoid(m.frame_fc(y))
            return frame, onset, offset, velocity

    net = Net().eval()
    dummy = torch.rand(1, 1001, 1025)
    tmp = out + ".fp32"
    torch.onnx.export(
        net, (dummy,), tmp, input_names=["power"], output_names=["frame", "onset", "offset", "velocity"],
        dynamic_axes={"power": {0: "batch", 1: "frames"}, **{k: {0: "batch", 1: "frames"} for k in ("frame", "onset", "offset", "velocity")}},
        opset_version=17, dynamo=False,
    )
    model = onnx.load(tmp)

    # Store large float32 weights as float16 and cast them back on load.
    graph = model.graph
    casts = []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.float32 or arr.size < 1024:
            continue
        name = init.name
        half = numpy_helper.from_array(arr.astype(np.float16), name + "_fp16")
        init.CopyFrom(half)
        casts.append(onnx.helper.make_node("Cast", [name + "_fp16"], [name], to=onnx.TensorProto.FLOAT))
    for node in reversed(casts):
        graph.node.insert(0, node)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    import os

    os.unlink(tmp)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
