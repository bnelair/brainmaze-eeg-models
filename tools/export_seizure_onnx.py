"""Export the brainmaze-torch 0.2.0 seizure models to ONNX (development tool, NOT shipped).

Writes brainmaze_eeg_models/seizure/_models/{modelA_paper,modelB_full}.onnx from the PyTorch
weights released in brainmaze-torch 0.2.0, checks them against PyTorch, and prints their
SHA-256 (to be pinned in brainmaze_eeg_models/seizure/_models/__init__.py).

    pip install -e ".[export]"          # torch (CPU is enough), onnx, brainmaze-torch==0.2.0
    python tools/export_seizure_onnx.py [--check-only]

The exported graph is an export-safe wrapper with exactly the math of
``SeizureDetectModel.forward`` in eval mode (dropout is the identity):

- ``squeeze(x)`` -> ``squeeze(2)`` (the original squeezes every size-1 axis and then
  re-adds the batch axis with a Python ``if bs == 1``, which a graph cannot express);
- the explicit zero initial LSTM state -> the LSTM default (zeros);
- outputs ``(logits, probs)``, both ``(n_times, batch, 4)``, as the original.

TorchScript-based exporter (``dynamo=False``), opset 18, dynamic batch and time axes. This
is the configuration validated by the ONNX prototype (parity with PyTorch ~1e-6 end to end).
"""
import argparse
import hashlib
import os
import sys

import numpy as np

OPSET = 18
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "brainmaze_eeg_models", "seizure", "_models")
FILES = {"modelA": "modelA_paper.onnx", "modelB": "modelB_full.onnx"}


def build_wrapper(base):
    import torch.nn as nn
    import torch.nn.functional as F

    class FixedExport(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.conv1, self.conv2, self.lstm, self.fc1 = m.conv1, m.conv2, m.lstm, m.fc1

        def forward(self, x):
            x = x.unsqueeze(1)                    # (B, 1, 100, T)
            x = F.relu(self.conv1(x))             # (B, 20, 96, T)
            x = F.relu(self.conv2(x))             # (B, 200, 1, T)
            x = x.squeeze(2).permute(2, 0, 1)     # (T, B, 200)
            x, _ = self.lstm(x)                   # zero initial state, as the original
            x = self.fc1(x)
            return x, F.softmax(x, dim=2)

    return FixedExport(base).eval()


def export(name, path):
    import torch
    from brainmaze_torch.seizure_detection import load_trained_model as load_torch

    mod = build_wrapper(load_torch(name))
    x = torch.randn(2, 100, 599)
    torch.onnx.export(mod, (x,), path, dynamo=False, opset_version=OPSET,
                      input_names=["x"], output_names=["logits", "probs"],
                      dynamic_axes={"x": {0: "batch", 2: "time"},
                                    "logits": {0: "time", 1: "batch"},
                                    "probs": {0: "time", 1: "batch"}})


def check(name, path):
    """Max |ONNX - PyTorch| on random spectrograms of several shapes (original forward)."""
    import onnx
    import onnxruntime as ort
    import torch
    from brainmaze_torch.seizure_detection import load_trained_model as load_torch

    onnx.checker.check_model(onnx.load(path), full_check=True)
    base = load_torch(name)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for bs, t in [(1, 599), (2, 599), (7, 599), (3, 41), (1, 2), (5, 2)]:
        xx = rng.standard_normal((bs, 100, t)).astype(np.float32)
        with torch.inference_mode():
            _, p_t = base(torch.from_numpy(xx))
        _, p_o = sess.run(None, {"x": xx})
        assert p_o.shape == tuple(p_t.shape), (p_o.shape, tuple(p_t.shape))
        worst = max(worst, float(np.abs(p_o - p_t.numpy()).max()))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-only", action="store_true", help="only check the committed files")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ok = True
    for name, fname in FILES.items():
        path = os.path.join(args.out, fname)
        if not args.check_only:
            export(name, path)
        err = check(name, path)
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"{name}: {path}  {os.path.getsize(path) / 1e6:.2f} MB  sha256 {sha}  max|ORT-torch| probs {err:.2e}")
        ok &= err < 1e-5
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
