"""Parity of the ONNX seizure models with the PyTorch originals (brainmaze-torch 0.2.0).

Runs only where PyTorch and brainmaze-torch 0.2.0 are installed (``pip install -e ".[export]"``);
CI without them skips it (the golden tests cover the ONNX path there; the shipped models are
SHA-256 pinned). Run it by hand whenever torch, onnx or onnxruntime move:
``pip install -e ".[test,export]" && pytest tests/test_seizure_parity.py``.

Tolerance 1e-4 on probabilities, the same as the brainmaze-torch golden tests (a real change
of the model moves them by up to ~0.8). Observed differences are ~1e-6 typically and up to
~2.1e-5 (modelB, 400 Hz synthetic input, p = 0.49: the float32 BiLSTM over 599 steps
accumulates rounding differently in ONNX Runtime and PyTorch; largest where the softmax is
steepest). NaN masks must be identical.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from brainmaze_eeg_models.seizure import TRAINED_MODELS, load_trained_model, predict_channel_seizure_probability
from brainmaze_eeg_models.seizure import infer_seizure_probability, preprocess_input

MODEL_DIR = Path(__file__).resolve().parents[1] / "brainmaze_eeg_models" / "seizure" / "_models"
DATA = Path(__file__).resolve().parent / "data"
TOL = 1e-4


def test_committed_models_match_pinned_hashes():
    for name, (fname, sha) in TRAINED_MODELS.items():
        assert hashlib.sha256((MODEL_DIR / fname).read_bytes()).hexdigest() == sha, name


torch = pytest.importorskip("torch")
bt = pytest.importorskip("brainmaze_torch.seizure_detection")


@pytest.fixture(scope="module", params=["modelA", "modelB"])
def models(request):
    return request.param, load_trained_model(request.param, device="cpu"), bt.load_trained_model(request.param)


def _cases():
    rng = np.random.default_rng(0)
    d = np.load(DATA / "seizure_segment_1.npz")
    x, fs = d["data"].squeeze().astype(np.float64), int(d["fs"].squeeze())
    yield "fixture", x, fs
    y = x.copy()
    y[500 * fs:700 * fs] = np.nan
    yield "fixture_gap", y, fs
    for fs in (200, 256, 500, 512, 1000):
        z = rng.standard_normal(int(15 * 60 * fs))
        t = np.arange(z.size) / fs
        burst = (t > 400) & (t < 460)
        z[burst] += 5 * np.sin(2 * np.pi * 7 * t[burst])          # rhythmic burst
        z[int(100 * fs):int(130 * fs)] = np.nan                       # gap
        z[int(600 * fs):int(603 * fs)] = 1.5                           # flat segment
        yield f"synthetic_{fs}", z, fs
    # the independent review's worst case (#5 R1): modelB differed by 2.1e-5 here (p ~ 0.49)
    rng2 = np.random.default_rng(42)
    for fs in (200, 256, 400):
        n = int(fs * 900)
        t = np.arange(n) / fs
        z = np.cumsum(rng2.standard_normal(n)) * 0.05 + rng2.standard_normal(n) * 20
        b = (t > 360) & (t < 420)
        z[b] += 150 * np.sin(2 * np.pi * 4 * t[b]) * np.sin(np.pi * (t[b] - 360) / 60)
    yield "review_worst_400", z, 400


@pytest.mark.parametrize("case", [c[0] for c in _cases()])
def test_end_to_end_parity(models, case):
    name, onnx_model, torch_model = models
    _, x, fs = next(c for c in _cases() if c[0] == case)
    t1, p1 = predict_channel_seizure_probability(x, fs, onnx_model)
    t2, p2 = bt.predict_channel_seizure_probability(x, fs, torch_model)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(np.isnan(p1), np.isnan(p2))
    assert np.nanmax(np.abs(p1 - p2)) <= TOL


def test_raw_model_parity(models):
    name, onnx_model, torch_model = models
    rng = np.random.default_rng(1)
    for bs, t in [(1, 599), (16, 599), (3, 41), (2, 2)]:
        sxx = rng.standard_normal((bs, 100, t)).astype(np.float32)
        y1 = infer_seizure_probability(sxx, onnx_model)
        y2 = bt.infer_seizure_probability(sxx, torch_model)
        assert y1.shape == y2.shape
        assert np.abs(y1 - y2).max() <= TOL


def test_preprocessing_is_identical():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 300 * 500))
    x[1, 1000:5000] = np.nan
    np.testing.assert_array_equal(preprocess_input(x, 500), bt.preprocess_input(x, 500))
