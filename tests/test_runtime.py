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

    seen = []

    def boom(path, sess_options=None, providers=None):
        seen.append(providers)
        if any(isinstance(p, tuple) and p[0] == "CUDAExecutionProvider" for p in providers):
            raise RuntimeError("libcudnn.so.9: cannot open shared object file")
        return real(path, sess_options=sess_options, providers=providers)

    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", boom)
    with pytest.raises(RuntimeError, match="libcudnn"):
        OnnxModel(TINY, device="cuda")
    with pytest.warns(RuntimeWarning):
        assert OnnxModel(TINY, device="auto").device == "cpu"
    # TF32 is disabled on the CUDA provider unless asked for
    assert seen[0][0] == ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})
    with pytest.raises(RuntimeError):
        OnnxModel(TINY, device="cuda", cuda_tf32=True, cuda_device_id=1)
    assert seen[-1][0] == ("CUDAExecutionProvider", {"device_id": 1, "use_tf32": 1})


def test_threads_option():
    m = OnnxModel(TINY, device="cpu", threads=1)
    assert m.threads == 1
    np.testing.assert_allclose(m.run(_inputs(5))["y"], OnnxModel(TINY, device="cpu").run(_inputs(5))["y"])


@pytest.mark.parametrize("kw", [dict(device="gpu"), dict(threads=0), dict(threads=1.5), dict(threads=True),
                                dict(batch_size=0), dict(batch_size=2.0),
                                dict(cuda_device_id=-1), dict(cuda_device_id=True)])
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


# --- round 2 of the review (#2 R1-R6) ----------------------------------------------------

def test_output_batch_axes():
    m = OnnxModel(TINY, device="cpu", batch_size=3, output_batch_axes={"s": 0})
    x = _inputs(7)
    np.testing.assert_allclose(m.run(x)["s"], m.run(x, batch_size=7)["s"], rtol=1e-6)
    with pytest.raises(ValueError, match="unknown output"):
        OnnxModel(TINY, output_batch_axes={"nope": 1})


@pytest.mark.parametrize("ax", [1.5, True, -1, 1, "0", None])
def test_output_batch_axes_validation(ax):
    # 's' has 1 dimension, so only 0 is valid; floats/bools/strings are rejected, not truncated
    with pytest.raises(ValueError, match="output_batch_axes"):
        OnnxModel(TINY, device="cpu", output_batch_axes={"s": ax})


class _Meta:
    def __init__(self, name, shape, type="tensor(float)"):
        self.name, self.shape, self.type = name, shape, type


class _TransposingSession:
    """A real session whose output 'y' comes back as (3, batch): a non-batch-major output whose
    metadata says nothing (unnamed dims), so only the per-chunk check can catch a wrong axis."""

    def __init__(self, path, sess_options=None, providers=None):
        self._s = _REAL_SESSION(path, sess_options=sess_options, providers=["CPUExecutionProvider"])

    def run(self, names, feed):
        return [o.T if n == "y" else o for n, o in zip(names, self._s.run(names, feed))]

    def get_outputs(self):
        return [_Meta("y", [None, None]), _Meta("s", [None])]

    def __getattr__(self, k):
        return getattr(self._s, k)


@pytest.mark.parametrize("n, bs", [(10, 4), (2, 4), (4, 4)])
def test_wrong_batch_axis_raises_instead_of_mixing_items(monkeypatch, n, bs):
    monkeypatch.setattr(runtime._ort(), "InferenceSession", _TransposingSession)
    m = OnnxModel(TINY, device="cpu", batch_size=bs)       # output_batch_axes not set: 0 is wrong
    with pytest.raises(RuntimeError, match="batch axis 0"):
        m.run(_inputs(n))
    # with the right axis the chunks are concatenated correctly
    m = OnnxModel(TINY, device="cpu", batch_size=bs, output_batch_axes={"y": 1})
    out = m.run(_inputs(n))["y"]
    np.testing.assert_allclose(out, (2 * _inputs(n)["a"].astype(np.float32) + 1).T, rtol=1e-6)


class _TimeBatchMetaSession(_TransposingSession):
    """Declares output 'y' as (3, 'batch'), like the seizure models' (time, batch, 4)."""

    def get_outputs(self):
        return [_Meta("y", [3, "batch"]), _Meta("s", ["batch"])]


def test_batch_axis_is_cross_checked_with_the_model_metadata(monkeypatch):
    monkeypatch.setattr(runtime._ort(), "InferenceSession", _TimeBatchMetaSession)
    with pytest.raises(ValueError, match="fixed size 3"):
        OnnxModel(TINY, device="cpu")
    assert OnnxModel(TINY, device="cpu", output_batch_axes={"y": 1}).output_batch_axes == {"y": 1, "s": 0}

    class Named(_TransposingSession):
        def get_outputs(self):
            return [_Meta("y", ["time", "batch"]), _Meta("s", ["batch"])]

    monkeypatch.setattr(runtime._ort(), "InferenceSession", Named)
    with pytest.raises(ValueError, match="batch dimension at axis 1"):
        OnnxModel(TINY, device="cpu")


@pytest.mark.parametrize("val", ["false", 1, 0, None, np.float32(0)])
def test_cuda_tf32_must_be_bool(val):
    with pytest.raises(TypeError, match="cuda_tf32"):
        OnnxModel(TINY, device="cpu", cuda_tf32=val)
    assert OnnxModel(TINY, device="cpu", cuda_tf32=np.bool_(False)).cuda_tf32 is False


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("bad", [np.inf, -np.inf, 1e39])   # 1e39 overflows to inf in float32
def test_inf_and_float32_overflow_are_rejected(bad):
    m = OnnxModel(TINY, device="cpu", batch_size=2)
    x = _inputs(5)
    x["b"][4, 0] = bad                                      # last chunk: still raised before any run
    calls = []
    real_run = m._session.run
    m._session = type("S", (), {"run": lambda self, *a: calls.append(1) or real_run(*a),
                                "get_providers": lambda self: ["CPUExecutionProvider"]})()
    with pytest.raises(ValueError, match="NaN or inf"):
        m.run(x)
    assert calls == []


def test_inputs_are_cast_per_chunk_not_copied_whole():
    m = OnnxModel(TINY, device="cpu", batch_size=2)
    x = _inputs(9)
    big = np.zeros((9, 6))
    big[:, ::2] = x["a"]
    a = big[:, ::2]                                           # non-contiguous float64 view
    out = m.run({"a": a, "b": x["b"]})
    np.testing.assert_allclose(out["y"], 2 * x["a"].astype(np.float32) + 1, rtol=1e-6)
    feed, dtypes, n = m._prepare({"a": a, "b": x["b"]})
    assert n == 9 and np.shares_memory(feed["a"], a)          # no full-size copy before chunking


class _CudaSession:
    """Mimics a working CUDA session (computation on the CPU)."""
    created = []

    def __init__(self, path, sess_options=None, providers=None):
        _CudaSession.created.append(providers)
        self._s = _REAL_SESSION(path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        self._cuda = any(isinstance(p, tuple) and p[0] == "CUDAExecutionProvider" for p in providers)

    def get_providers(self):
        return (["CUDAExecutionProvider"] if self._cuda else []) + ["CPUExecutionProvider"]

    def __getattr__(self, k):
        return getattr(self._s, k)


def test_auto_and_cuda_use_a_working_cuda_provider(monkeypatch):
    ort = runtime._ort()
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", _CudaSession)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for dev in ("auto", "cuda"):
            m = OnnxModel(TINY, device=dev)
            assert m.device == "cuda" and m.providers[0] == "CUDAExecutionProvider"
            assert m.run(_inputs(3))["y"].shape == (3, 3)
        assert OnnxModel(TINY, device="cpu").device == "cpu"


def test_auto_warns_when_cpu_build_clobbered_the_gpu_build(monkeypatch):
    # both distributions installed, the CPU build is the active module: no CUDA provider at all
    monkeypatch.setattr(runtime._ort(), "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_installed_ort_distributions",
                        lambda: ["onnxruntime 1.30.0", "onnxruntime-gpu 1.30.0"])
    with pytest.warns(RuntimeWarning, match="Both onnxruntime and onnxruntime-gpu"):
        assert OnnxModel(TINY, device="auto").device == "cpu"
    assert "CONFLICT" in runtime.device_report()
    with pytest.raises(RuntimeError, match="conflict"):
        OnnxModel(TINY, device="cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        OnnxModel(TINY, device="cpu")                        # explicit CPU: no warning


def test_auto_is_silent_on_a_plain_cpu_install(monkeypatch):
    monkeypatch.setattr(runtime._ort(), "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_installed_ort_distributions", lambda: ["onnxruntime 1.30.0"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert OnnxModel(TINY, device="auto").device == "cpu"
    assert "CONFLICT" not in runtime.device_report()


def test_preload_runs_once_before_the_cuda_session_and_is_quiet(monkeypatch, capfd):
    ort = runtime._ort()
    events = []

    def fake_preload():
        events.append("preload")
        print("Failed to load libcublasLt.so.13: cannot open shared object file")

    class S(_CudaSession):
        def __init__(self, *a, **k):
            events.append("session")
            super().__init__(*a, **k)

    monkeypatch.setattr(runtime, "_PRELOAD_DONE", False)
    monkeypatch.setattr(runtime, "_PRELOAD_OUTPUT", "")
    monkeypatch.setattr(ort, "preload_dlls", fake_preload, raising=False)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", S)
    OnnxModel(TINY, device="auto")
    OnnxModel(TINY, device="cuda")
    assert events == ["preload", "session", "session"]        # once per process, before the session
    assert "Failed to load" not in capfd.readouterr().out      # captured, not printed
    assert "libcublasLt" in runtime.device_report()


# --- check_gpu (verification r2, #2 V1): a real check, not the provider list ----------------

def test_check_gpu_embeds_the_tiny_test_model():
    with open(TINY, "rb") as fh:
        assert runtime._TINY_ONNX == fh.read()


def test_check_gpu_raises_without_cuda(monkeypatch):
    monkeypatch.setattr(runtime._ort(), "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="device='cuda'"):
        runtime.check_gpu()


def test_check_gpu_detects_the_silent_cpu_fallback(monkeypatch):
    # a GPU build without usable CUDA libraries: the provider is LISTED, the session gets the CPU
    ort = runtime._ort()
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", _FallbackSession)
    assert "does not prove" in runtime.device_report()
    with pytest.raises(RuntimeError, match="could not initialise the CUDA provider"):
        runtime.check_gpu()


def test_check_gpu_reports_inference_failures(monkeypatch):
    ort = runtime._ort()

    class Broken(_CudaSession):
        def run(self, *a, **k):
            raise RuntimeError("CUBLAS_STATUS_NOT_INITIALIZED")

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", Broken)
    with pytest.raises(RuntimeError, match="inference failed: CUBLAS"):
        runtime.check_gpu()


def test_check_gpu_success_on_a_working_cuda_session(monkeypatch):
    import brainmaze_eeg_models as bm
    ort = runtime._ort()
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", _CudaSession)
    _CudaSession.created.clear()
    msg = bm.check_gpu(device_id=2)
    assert msg.startswith("CUDA works: GPU 2") and "CUDAExecutionProvider" in msg
    assert _CudaSession.created[-1][0] == ("CUDAExecutionProvider", {"device_id": 2, "use_tf32": 0})


# --- models from bytes; check_gpu needs no temporary file (PR #7 R3) ----------------------

def test_model_from_bytes_matches_the_file():
    with open(TINY, "rb") as fh:
        data = fh.read()
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((2, 4), dtype=np.float32)
    from_file = OnnxModel(TINY, device="cpu").run({"a": a, "b": b})
    m = OnnxModel(data, device="cpu", name="tiny", sha256=runtime._sha256(TINY))
    assert m.path is None and m.name == "tiny"
    out = m.run({"a": a, "b": b})
    for k in from_file:
        np.testing.assert_array_equal(out[k], from_file[k])
    with pytest.raises(ValueError, match="name is required"):
        OnnxModel(data, device="cpu")
    with pytest.raises(RuntimeError, match="given as bytes.*SHA-256"):
        OnnxModel(data, device="cpu", name="tiny", sha256="0" * 64)


def test_check_gpu_does_not_need_a_writable_temp_dir(monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", os.path.join(os.path.dirname(TINY), "no", "such", "dir"))
    ort = runtime._ort()
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", _CudaSession)
    assert runtime.check_gpu().startswith("CUDA works: GPU 0")
    # without CUDA it is still the documented RuntimeError, never an OSError
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="device='cuda'"):
        runtime.check_gpu()
