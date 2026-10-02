import os
import warnings

import numpy as np
import pytest

from brainmaze_eeg_models import runtime
from brainmaze_eeg_models.runtime import OnnxModel

TINY = os.path.join(os.path.dirname(__file__), "data", "tiny_affine.onnx")
TINY_SHA = runtime._sha256(TINY)
_REAL_SESSION = runtime._ort().InferenceSession


def _inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    return {"a": rng.standard_normal((n, 3)), "b": rng.standard_normal((n, 4))}


def test_cpu_run_matches_reference():
    m = OnnxModel(TINY, device="cpu", sha256=TINY_SHA)
    assert m.device == "cpu" and m.providers == ["CPUExecutionProvider"]
    assert m.input_names == ["a", "b"] and m.output_names == ["y", "s"]
    x = _inputs(10)
    out = m.run(x)
    np.testing.assert_allclose(out["y"], 2 * x["a"].astype(np.float32) + 1, rtol=1e-6)
    np.testing.assert_allclose(out["s"], x["b"].astype(np.float32).sum(-1), rtol=1e-5, atol=1e-6)
    assert out["y"].dtype == np.float32


@pytest.mark.parametrize("bs", [1, 3, 7, 10, 64])
def test_batching_is_transparent(bs):
    m = OnnxModel(TINY, device="cpu", batch_size=4)
    x = _inputs(10, seed=1)
    ref = m.run(x)                 # default batch_size=4 -> 3 chunks
    out = m.run(x, batch_size=bs)
    for k in ref:
        assert out[k].shape[0] == 10
        np.testing.assert_allclose(out[k], ref[k], rtol=1e-6, atol=1e-6)


def test_auto_on_cpu_only_machine_is_cpu(monkeypatch):
    monkeypatch.setattr(runtime._ort(), "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert OnnxModel(TINY, device="auto").device == "cpu"


def test_cuda_requested_without_provider_raises_clearly(monkeypatch):
    monkeypatch.setattr(runtime._ort(), "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError) as err:
        OnnxModel(TINY, device="cuda")
    msg = str(err.value)
    assert "device='cuda'" in msg and "onnxruntime-gpu" in msg and "device='cpu'" in msg


class _FallbackSession:
    """Mimics onnxruntime-gpu whose CUDA libraries fail to load: the session silently gets CPU."""

    def __init__(self, path, sess_options=None, providers=None):
        self._s = _REAL_SESSION(path, sess_options=sess_options, providers=["CPUExecutionProvider"])

    def __getattr__(self, k):
        return getattr(self._s, k)


def test_cuda_silent_fallback_is_detected(monkeypatch):
    ort = runtime._ort()
    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", _FallbackSession)
    with pytest.raises(RuntimeError, match="could not initialise the CUDA provider"):
        OnnxModel(TINY, device="cuda")
    with pytest.warns(RuntimeWarning, match="runs on the CPU"):
        m = OnnxModel(TINY, device="auto")
    assert m.device == "cpu"
    np.testing.assert_allclose(m.run(_inputs(2))["y"].shape, (2, 3))


def test_cuda_session_error_auto_falls_back(monkeypatch):
    ort = runtime._ort()
    real = _REAL_SESSION

    def boom(path, sess_options=None, providers=None):
        if "CUDAExecutionProvider" in providers:
            raise RuntimeError("libcudnn.so.9: cannot open shared object file")
        return real(path, sess_options=sess_options, providers=providers)

    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", boom)
    with pytest.raises(RuntimeError, match="libcudnn"):
        OnnxModel(TINY, device="cuda")
    with pytest.warns(RuntimeWarning):
        assert OnnxModel(TINY, device="auto").device == "cpu"


def test_threads_option():
    m = OnnxModel(TINY, device="cpu", threads=1)
    assert m.threads == 1
    np.testing.assert_allclose(m.run(_inputs(5))["y"], OnnxModel(TINY, device="cpu").run(_inputs(5))["y"])


@pytest.mark.parametrize("kw", [dict(device="gpu"), dict(threads=0), dict(threads=1.5), dict(threads=True),
                                dict(batch_size=0), dict(batch_size=2.0)])
def test_bad_arguments(kw):
    with pytest.raises(ValueError):
        OnnxModel(TINY, **kw)


def test_checksum_mismatch_raises(tmp_path):
    p = tmp_path / "m.onnx"
    p.write_bytes(open(TINY, "rb").read())
    OnnxModel(str(p), sha256=TINY_SHA)
    with pytest.raises(RuntimeError, match="corrupted"):
        OnnxModel(str(p), sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        OnnxModel(str(tmp_path / "missing.onnx"))


def test_input_validation():
    m = OnnxModel(TINY, device="cpu")
    x = _inputs(4)
    bad = dict(x, a=x["a"].copy())
    bad["a"][1, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        m.run(bad)
    with pytest.raises(ValueError, match="do not match"):
        m.run({"a": x["a"]})
    with pytest.raises(ValueError, match="same batch size"):
        m.run({"a": x["a"], "b": x["b"][:3]})
    with pytest.raises(ValueError, match="shape"):
        m.run({"a": x["a"][:, :2], "b": x["b"]})
    with pytest.raises(ValueError, match="dimensions"):
        m.run({"a": x["a"][0], "b": x["b"]})
    with pytest.raises(ValueError, match="empty"):
        m.run({"a": x["a"][:0], "b": x["b"][:0]})


def test_device_report_mentions_providers():
    rep = runtime.device_report()
    assert "CPUExecutionProvider" in rep and "onnxruntime" in rep
