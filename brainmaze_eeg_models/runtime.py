"""
ONNX Runtime layer shared by every model in the package (inference only).

:class:`OnnxModel` wraps one ``onnxruntime.InferenceSession``:

- **device**: ``'auto'`` (CUDA if it really works, else CPU), ``'cpu'`` or ``'cuda'``.
  ``'cuda'`` never falls back silently: if the CUDA execution provider is missing or cannot
  be initialised, it raises :class:`RuntimeError` with what to install. (ONNX Runtime itself
  silently creates a CPU session when the CUDA libraries fail to load; this wrapper checks
  the providers the session actually got.)
- **threads**: number of CPU threads for one inference call (``intra_op_num_threads``);
  ``None`` = ONNX Runtime's default (one per physical core).
- **batching**: :meth:`OnnxModel.run` splits the inputs along the first (batch) axis into
  chunks of ``batch_size``, casts and checks each chunk just before its inference call, and
  concatenates the outputs along each output's batch axis (checked on every chunk, see
  ``output_batch_axes``). This bounds the memory of each inference call (activations and the
  cast copy of the input); the outputs of the whole batch are of course all kept.
- **integrity**: an optional SHA-256 of the model file is verified before loading, so a
  truncated or swapped model file fails loudly instead of producing wrong numbers.
- **inputs**: every input is cast to the dtype the model declares and must be finite (NaN or
  inf reaching a network silently corrupts its output; the detectors handle gaps before).

GPU installation (NVIDIA)
-------------------------
There is deliberately **no** ``[gpu]`` extra. ``onnxruntime`` (CPU) and ``onnxruntime-gpu``
install the **same** Python module ``onnxruntime``: with both installed, the CPU build
typically wins without any error (``pip check`` passes), and uninstalling one breaks the other.
Swap the package by hand::

    pip install brainmaze-eeg-models
    pip uninstall -y onnxruntime
    pip install "onnxruntime-gpu[cuda,cudnn]"     # CUDA + cuDNN as pip wheels
    python -c "import brainmaze_eeg_models as bm; print(bm.check_gpu())"

:func:`check_gpu` is the real check: it creates a CUDA session and runs a tiny inference on
the GPU, and raises :class:`RuntimeError` with the reason (and what to install) if that does
not work. A ``CUDAExecutionProvider`` in ``onnxruntime.get_available_providers()`` only shows
that the GPU build is installed: a GPU build without usable CUDA libraries lists it too, and
then every session falls back to the CPU. Any model created with ``device='cuda'`` (e.g.
``SpindleDetector('eeg', device='cuda')``) also raises if its CUDA session cannot be created.
:func:`device_report` lists what is installed.

**Upgrades:** ``pip install -U brainmaze-eeg-models`` re-installs the CPU ``onnxruntime``
(it is a dependency), which overwrites the GPU module again. Upgrade with
``pip install -U --no-deps brainmaze-eeg-models`` (or redo the swap afterwards). With
``device='auto'`` a clobbered install is reported by a :class:`RuntimeWarning`.

**CUDA versions:** ``onnxruntime-gpu`` >= 1.27 on PyPI is built for **CUDA 13.0** + cuDNN 9
(needs a driver that supports CUDA 13). For a CUDA 12 system (e.g. an older cluster driver)
the simplest route is the last CUDA 12 builds on PyPI, 1.24-1.26 (Python >= 3.11), whose
``[cuda,cudnn]`` extras install the ``-cu12`` CUDA and cuDNN wheels::

    pip uninstall -y onnxruntime onnxruntime-gpu
    pip install "onnxruntime-gpu[cuda,cudnn]<1.27"

For a newer CUDA 12 build use Microsoft's CUDA 12 feed, whose builds need **CUDA >= 12.8**
(1.27-1.29, Python 3.11-3.14 on 2026-10-02; pick the newest version listed there)::

    pip uninstall -y onnxruntime onnxruntime-gpu
    pip install --no-deps onnxruntime-gpu==1.29.0 \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
        nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12   # or a system CUDA >= 12.8 + cuDNN 9

(``--no-deps``: the feed also mirrors other packages, in old versions (without it pip would
install e.g. numpy 2.1.2 and protobuf 5.28.3 from the feed); the dependencies are already
installed with the CPU package. With ``--index-url`` pip does not look at PyPI at all, so it
cannot pick the CUDA 13 build from there.) **Python 3.10:** the newest ``onnxruntime-gpu`` with Python 3.10 wheels is 1.23.2, a CUDA 12
build, so the first recipe gives CUDA 12 (its ``[cuda,cudnn]`` extras install the ``-cu12``
wheels). See the `ONNX Runtime CUDA requirements
<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_.

The GPU path is tested here with mocked sessions and on a GPU-less host with a real
``onnxruntime-gpu`` (it raises / warns as documented); numerical GPU-vs-CPU agreement has not
been measured yet in this package (TF32 is disabled by default, see ``cuda_tf32``).
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.metadata as _md
import io
import logging
import os
import tempfile
import threading
import warnings
from typing import Mapping, Sequence

import numpy as np

__all__ = ['OnnxModel', 'available_providers', 'device_report', 'check_gpu', 'DEVICES']

DEVICES = ('auto', 'cpu', 'cuda')

_log = logging.getLogger(__name__)

_CPU = 'CPUExecutionProvider'
_CUDA = 'CUDAExecutionProvider'

# ONNX tensor type strings -> numpy dtypes (only what our models use).
_ORT_DTYPES = {
    'tensor(float)': np.float32,
    'tensor(double)': np.float64,
    'tensor(int64)': np.int64,
    'tensor(int32)': np.int32,
}


def _ort():
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - both distributions missing / broken
        raise ImportError(
            "onnxruntime cannot be imported. Install exactly one of `onnxruntime` (CPU) or "
            "`onnxruntime-gpu` (NVIDIA GPU): pip install --force-reinstall onnxruntime"
        ) from exc
    return onnxruntime


def _installed_ort_distributions() -> list[str]:
    names = []
    for dist in ('onnxruntime', 'onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-openvino'):
        try:
            names.append(f"{dist} {_md.version(dist)}")
        except _md.PackageNotFoundError:
            pass
    return names


def _ort_conflict() -> bool:
    """True if both the CPU and the GPU distribution are installed (the CPU build usually wins)."""
    dists = _installed_ort_distributions()
    return (any(d.startswith('onnxruntime ') for d in dists)
            and any(d.startswith('onnxruntime-gpu ') for d in dists))


_SWAP_FIX = ("pip uninstall -y onnxruntime onnxruntime-gpu && pip install \"onnxruntime-gpu[cuda,cudnn]\" "
             "(later upgrades: pip install -U --no-deps brainmaze-eeg-models)")


def available_providers() -> list[str]:
    """Execution providers compiled into the installed onnxruntime (not necessarily usable)."""
    return list(_ort().get_available_providers())


def _cuda_help(reason: str) -> str:
    dists = _installed_ort_distributions()
    lines = [f"device='cuda' requested but CUDA cannot be used: {reason}",
             f"Installed onnxruntime distributions: {', '.join(dists) or 'none'}."]
    has_cpu = any(d.startswith('onnxruntime ') for d in dists)
    has_gpu = any(d.startswith('onnxruntime-gpu ') for d in dists)
    if has_cpu and has_gpu:
        lines.append(
            "Both `onnxruntime` and `onnxruntime-gpu` are installed; they provide the same module "
            f"and conflict. Fix: {_SWAP_FIX}")
    elif not has_gpu:
        lines.append(
            "The CPU build is installed. For NVIDIA GPUs: pip uninstall -y onnxruntime && "
            "pip install \"onnxruntime-gpu[cuda,cudnn]\" (CUDA 12 systems and Python 3.10: see "
            "'GPU installation' in the brainmaze_eeg_models.runtime documentation).")
    else:
        lines.append(
            "onnxruntime-gpu is installed but the CUDA execution provider could not be "
            "initialised: an NVIDIA driver, and CUDA + cuDNN libraries matching this "
            "onnxruntime-gpu build, are required (pip install \"onnxruntime-gpu[cuda,cudnn]\" "
            "installs matching CUDA/cuDNN wheels). See "
            "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements")
    lines.append("Use device='cpu' (or 'auto') to run on the CPU.")
    return "\n".join(lines)


_PRELOAD_LOCK = threading.Lock()
_PRELOAD_DONE = False
_PRELOAD_OUTPUT = ''


def _preload_cuda_libraries() -> None:
    """Load CUDA/cuDNN from the nvidia-* pip wheels if present (onnxruntime >= 1.21).

    Runs once per process. ``onnxruntime.preload_dlls`` prints a "Failed to load ..." line per
    missing library on every call; that output is captured (not printed), logged once at
    DEBUG level on the ``brainmaze_eeg_models.runtime`` logger and shown by
    :func:`device_report`. Whether CUDA really works is decided by the session check, not here.
    """
    global _PRELOAD_DONE, _PRELOAD_OUTPUT
    with _PRELOAD_LOCK:
        if _PRELOAD_DONE:
            return
        _PRELOAD_DONE = True
        preload = getattr(_ort(), 'preload_dlls', None)
        if preload is None:
            return
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                preload()
        except Exception as exc:  # noqa: BLE001 - best effort; the session check decides
            buf.write(f"preload_dlls raised {type(exc).__name__}: {exc}\n")
        _PRELOAD_OUTPUT = buf.getvalue().strip()
        if _PRELOAD_OUTPUT:
            _log.debug("onnxruntime.preload_dlls output:\n%s", _PRELOAD_OUTPUT)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def device_report() -> str:
    """Human-readable summary of the installed onnxruntime distributions and providers."""
    ort = _ort()
    lines = [
        f"onnxruntime module {ort.__version__} ({os.path.dirname(ort.__file__)})",
        f"installed distributions: {', '.join(_installed_ort_distributions()) or 'none'}",
        f"available providers: {', '.join(ort.get_available_providers())}",
        f"device='auto' would try: {'cuda, then cpu' if _CUDA in ort.get_available_providers() else 'cpu'}",
    ]
    if _CUDA in ort.get_available_providers():
        lines.append("note: a listed CUDAExecutionProvider does not prove that CUDA works (a GPU build without "
                     "usable CUDA libraries lists it too); run brainmaze_eeg_models.check_gpu()")
    if _ort_conflict():
        lines.append("CONFLICT: both onnxruntime and onnxruntime-gpu are installed; the module above is "
                     f"whichever was installed last. Fix: {_SWAP_FIX}")
    if _PRELOAD_OUTPUT:
        lines.append("onnxruntime.preload_dlls output (CUDA libraries):\n  "
                     + _PRELOAD_OUTPUT.replace("\n", "\n  "))
    return "\n".join(lines)


# tests/data/tiny_affine.onnx (273 bytes; y = 2 * a + 1, s = sum(b, -1)), used by check_gpu().
_TINY_ONNX = base64.b64decode(
    "CAgSD2JyYWlubWF6ZS10ZXN0czr1AQoRCgFhCgN0d28SAmEyIgNNdWwKEQoCYTIKA29uZRIBeSIDQWRkCigKAWIKBGF4ZXMSAXMiCVJl"
    "ZHVjZVN1bSoPCghrZWVwZGltcxgAoAECEgt0aW55X2FmZmluZSoNEAFCA3R3b0oEAAAAQCoNEAFCA29uZUoEAACAPyoUCAEQB0IEYXhl"
    "c0oI//////////9aGAoBYRITChEIARINCgcSBWJhdGNoCgIIA1oYCgFiEhMKEQgBEg0KBxIFYmF0Y2gKAggEYhgKAXkSEwoRCAESDQoH"
    "EgViYXRjaAoCCANiFAoBcxIPCg0IARIJCgcSBWJhdGNoQgQKABAR")


def check_gpu(device_id: int = 0) -> str:
    """Check that CUDA really works: create a CUDA session and run a tiny inference on the GPU.

    Unlike ``onnxruntime.get_available_providers()`` (which lists ``CUDAExecutionProvider``
    whenever the GPU build is installed, even if the CUDA libraries are missing and every
    session would fall back to the CPU), this runs a small model with ``device='cuda'`` on
    GPU ``device_id`` and compares its output with the exact result.

    Returns
    -------
    str
        A one-line confirmation (GPU index, session providers, onnxruntime version).

    Raises
    ------
    RuntimeError
        If CUDA cannot be used, with the reason and what to install (as ``device='cuda'``).
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'check_gpu.onnx')
        with open(path, 'wb') as fh:
            fh.write(_TINY_ONNX)
        m = OnnxModel(path, device='cuda', cuda_device_id=device_id, name='check_gpu')
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        b = np.ones((2, 4), dtype=np.float32)
        try:
            out = m.run({'a': a, 'b': b})
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(_cuda_help(f"the CUDA session was created but inference failed: {exc}")) from exc
        providers = m.providers
        del m                          # release the session before the file is removed
    if not (np.array_equal(out['y'], 2 * a + 1) and np.array_equal(out['s'], np.full(2, 4, np.float32))):
        raise RuntimeError(f"CUDA inference returned wrong values on GPU {device_id}: {out}")
    return (f"CUDA works: GPU {device_id}, session providers {providers}, "
            f"onnxruntime {_ort().__version__} ({', '.join(_installed_ort_distributions()) or '?'})")


class OnnxModel:
    """One ONNX model on one device.

    Parameters
    ----------
    path : str
        Path to the ``.onnx`` file.
    device : {'auto', 'cpu', 'cuda'}
        ``'auto'``: CUDA when the CUDA execution provider is installed and initialises, else
        CPU. A :class:`RuntimeWarning` is issued when the CPU is used although a GPU setup
        was evidently intended: ``onnxruntime-gpu`` is installed but CUDA cannot be
        initialised, or both ``onnxruntime`` and ``onnxruntime-gpu`` are installed and the CPU
        build is the active module. ``'cpu'``: always CPU.
        ``'cuda'``: CUDA or :class:`RuntimeError` (never a silent CPU fallback).
    threads : int, optional
        CPU threads per inference call (``intra_op_num_threads``). ``None``: ONNX Runtime
        default (one per physical core). Also used for the CPU parts of a CUDA session.
    batch_size : int
        Default number of items per inference call in :meth:`run`.
    cuda_device_id : int
        Index of the GPU used with ``device='cuda'`` / ``'auto'`` (default 0).
    cuda_tf32 : bool
        Allow TF32 matrix maths on the GPU. Default False: ONNX Runtime's CUDA provider enables
        TF32 by default, which changes results at the ~1e-3 level (not the ~1e-6 of float32
        rounding) on Ampere and newer GPUs; off, GPU and CPU outputs are expected to agree to
        float32 rounding (not yet measured on a GPU for these models). Must be a ``bool``.
    output_batch_axes : mapping name -> int, optional
        Batch axis of each output, if not 0 (e.g. ``{'probs': 1}`` for a ``(time, batch, ...)``
        output). :meth:`run` checks on every chunk that this axis has the chunk's length and
        concatenates the chunks along it; a wrong axis raises instead of silently mixing items.
    sha256 : str, optional
        Expected SHA-256 hex digest of the file; a mismatch raises :class:`RuntimeError`.
    name : str, optional
        Name used in messages.
    """

    def __init__(self, path: str, *, device: str = 'auto', threads: int | None = None,
                 batch_size: int = 32, cuda_device_id: int = 0, cuda_tf32: bool = False,
                 output_batch_axes: Mapping[str, int] | None = None,
                 sha256: str | None = None, name: str | None = None):
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        if threads is not None and (not isinstance(threads, (int, np.integer)) or isinstance(threads, bool)
                                    or threads < 1):
            raise ValueError(f"threads must be None or an integer >= 1, got {threads!r}")
        self.batch_size = self._check_batch_size(batch_size)
        self.path = os.fspath(path)
        self.name = name or os.path.basename(self.path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Model file not found: {self.path}")
        if sha256 is not None:
            digest = _sha256(self.path)
            if digest != sha256.lower():
                raise RuntimeError(
                    f"Model file {self.path} is corrupted or not the expected model "
                    f"(SHA-256 {digest}, expected {sha256}). Reinstall the package.")
        self.threads = None if threads is None else int(threads)
        if not isinstance(cuda_tf32, (bool, np.bool_)):
            raise TypeError(f"cuda_tf32 must be a bool, got {cuda_tf32!r}")
        self.cuda_tf32 = bool(cuda_tf32)
        if isinstance(cuda_device_id, bool) or not isinstance(cuda_device_id, (int, np.integer)) or cuda_device_id < 0:
            raise ValueError(f"cuda_device_id must be an integer >= 0, got {cuda_device_id!r}")
        self.cuda_device_id = int(cuda_device_id)

        ort = _ort()
        avail = ort.get_available_providers()
        if device == 'cpu':
            self._session = self._make_session([_CPU])
        elif device == 'cuda':
            if _CUDA not in avail:
                raise RuntimeError(_cuda_help(
                    f"this onnxruntime has no CUDAExecutionProvider (available: {avail})"))
            self._session = self._make_cuda_session(strict=True)
        else:  # auto
            session = self._make_cuda_session(strict=False) if _CUDA in avail else None
            if session is None and _CUDA not in avail and _ort_conflict():
                warnings.warn(
                    f"Both onnxruntime and onnxruntime-gpu are installed and the CPU build is the active "
                    f"module (no CUDAExecutionProvider), so {self.name} runs on the CPU. Fix: {_SWAP_FIX}",
                    RuntimeWarning, stacklevel=2)
            self._session = session if session is not None else self._make_session([_CPU])
        self.device = 'cuda' if self._session.get_providers()[0] == _CUDA else 'cpu'

        self._inputs = {i.name: i for i in self._session.get_inputs()}
        self.input_names = [i.name for i in self._session.get_inputs()]
        self.output_names = [o.name for o in self._session.get_outputs()]
        ranks = {o.name: len(o.shape) for o in self._session.get_outputs()}
        self.output_batch_axes = {k: 0 for k in self.output_names}
        for k, ax in (output_batch_axes or {}).items():
            if k not in self.output_batch_axes:
                raise ValueError(f"{self.name}: unknown output {k!r} in output_batch_axes ({self.output_names})")
            if isinstance(ax, (bool, np.bool_)) or not isinstance(ax, (int, np.integer)) or not 0 <= ax < ranks[k]:
                raise ValueError(f"{self.name}: output_batch_axes[{k!r}] must be an integer in "
                                 f"[0, {ranks[k]}) (the output has {ranks[k]} dimensions), got {ax!r}")
            self.output_batch_axes[k] = int(ax)
        # Cross-check with the model's declared shapes where they say something: the batch axis
        # cannot have a fixed size, and if the inputs' batch dimension is named (dim_param) and
        # that name appears in an output, it must be at the configured axis.
        in_batch = {i.shape[0] for i in self._session.get_inputs() if i.shape and isinstance(i.shape[0], str)}
        for o in self._session.get_outputs():
            ax = self.output_batch_axes[o.name]
            if len(o.shape) == 0:
                raise ValueError(f"{self.name}: output {o.name!r} is a scalar and has no batch axis")
            if isinstance(o.shape[ax], int):
                raise ValueError(f"{self.name}: output {o.name!r} {o.shape} has the fixed size {o.shape[ax]} "
                                 f"on its batch axis {ax}; set output_batch_axes")
            named = [j for j, d in enumerate(o.shape) if isinstance(d, str) and d in in_batch]
            if named and ax not in named:
                raise ValueError(f"{self.name}: output {o.name!r} {o.shape} has the inputs' batch dimension "
                                 f"at axis {named[0]}, not at {ax}; set output_batch_axes")

    # -- construction helpers -------------------------------------------------------------
    @staticmethod
    def _check_batch_size(batch_size) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {batch_size!r}")
        return int(batch_size)

    def _options(self):
        so = _ort().SessionOptions()
        if self.threads is not None:
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
        so.log_severity_level = 3  # errors only; provider problems are reported by us
        return so

    def _make_session(self, providers: Sequence):
        return _ort().InferenceSession(self.path, sess_options=self._options(), providers=list(providers))

    def _make_cuda_session(self, *, strict: bool):
        _preload_cuda_libraries()
        try:
            session = self._make_session([(_CUDA, {'device_id': self.cuda_device_id,
                                                   'use_tf32': int(self.cuda_tf32)}), _CPU])
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise RuntimeError(_cuda_help(f"creating the CUDA session failed: {exc}")) from exc
            warnings.warn(f"CUDA is installed but unusable ({exc}); {self.name} runs on the CPU.",
                          RuntimeWarning, stacklevel=3)
            return None
        got = session.get_providers()
        if not got or got[0] != _CUDA:
            # onnxruntime fell back to CPU silently (e.g. CUDA/cuDNN libraries not found).
            if strict:
                raise RuntimeError(_cuda_help(
                    f"onnxruntime could not initialise the CUDA provider (session providers: {got})"))
            warnings.warn(
                f"onnxruntime-gpu is installed but CUDA could not be initialised (session providers: "
                f"{got}); {self.name} runs on the CPU. See brainmaze_eeg_models.runtime.device_report().",
                RuntimeWarning, stacklevel=3)
            return session if got else None
        return session

    # -- inference ----------------------------------------------------------------------
    @property
    def providers(self) -> list[str]:
        """Execution providers of the session, in priority order."""
        return list(self._session.get_providers())

    def _prepare(self, inputs: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict, int]:
        """Validate names, ranks, fixed dimensions and the batch length; no full-size copies."""
        missing = [n for n in self.input_names if n not in inputs]
        extra = [n for n in inputs if n not in self._inputs]
        if missing or extra:
            raise ValueError(f"{self.name}: inputs {sorted(inputs)} do not match the model inputs "
                             f"{self.input_names} (missing {missing}, unexpected {extra})")
        feed, dtypes, n = {}, {}, None
        for k in self.input_names:
            meta = self._inputs[k]
            dtype = _ORT_DTYPES.get(meta.type)
            if dtype is None:
                raise TypeError(f"{self.name}: unsupported input type {meta.type} for {k!r}")
            v = np.asarray(inputs[k])          # a view for arrays; cast per chunk in run()
            if v.ndim != len(meta.shape):
                raise ValueError(f"{self.name}: input {k!r} must have {len(meta.shape)} dimensions "
                                 f"{meta.shape}, got shape {v.shape}")
            for ax, (want, got) in enumerate(zip(meta.shape, v.shape)):
                if ax > 0 and isinstance(want, int) and want != got:
                    raise ValueError(f"{self.name}: input {k!r} must have shape {meta.shape}, got {v.shape}")
            if n is None:
                n = v.shape[0]
            elif v.shape[0] != n:
                raise ValueError(f"{self.name}: all inputs must have the same batch size "
                                 f"(first axis); got {n} and {v.shape[0]} for {k!r}")
            feed[k], dtypes[k] = v, dtype
        if not n:
            raise ValueError(f"{self.name}: empty batch (first axis has length 0)")
        return feed, dtypes, n

    def _chunk(self, feed, dtypes, i, bs) -> dict[str, np.ndarray]:
        """Items ``i:i + bs`` cast to the model dtypes; NaN/inf (also from the cast) raise."""
        out = {}
        for k, v in feed.items():
            c = np.ascontiguousarray(v[i:i + bs], dtype=dtypes[k])
            if np.issubdtype(c.dtype, np.floating) and not np.isfinite(c).all():
                raise ValueError(f"{self.name}: input {k!r} contains NaN or inf (items {i}..{i + len(c) - 1}, "
                                 "after casting to the model dtype); the model would return silently "
                                 "wrong values. Handle gaps before inference.")
            out[k] = c
        return out

    def run(self, inputs: Mapping[str, np.ndarray], batch_size: int | None = None) -> dict[str, np.ndarray]:
        """Run the model on a batch, in chunks along the first axis.

        Parameters
        ----------
        inputs : mapping name -> array
            One array per model input (:attr:`input_names`), all with the same length of the
            first (batch) axis, at least 1. Cast to the model's dtype chunk by chunk; must be
            finite (checked for the whole batch before the first inference call).
        batch_size : int, optional
            Items per inference call; default :attr:`batch_size`. Results do not depend on
            it beyond float32 rounding (a few ulp; the kernels may block differently).

        Returns
        -------
        dict name -> np.ndarray
            One array per model output (:attr:`output_names`), concatenated along its batch
            axis (0 unless set with ``output_batch_axes``).

        Raises
        ------
        RuntimeError
            If an output's batch axis does not have the length of the chunk (wrong
            ``output_batch_axes``), so that items are never silently mixed up.
        """
        bs = self.batch_size if batch_size is None else self._check_batch_size(batch_size)
        feed, dtypes, n = self._prepare(inputs)
        for i in range(0, n, bs):              # validate everything before any inference
            self._chunk(feed, dtypes, i, bs)
        chunks: list[list[np.ndarray]] = [[] for _ in self.output_names]
        for i in range(0, n, bs):
            m = min(bs, n - i)
            out = self._session.run(self.output_names, self._chunk(feed, dtypes, i, bs))
            for name, lst, o in zip(self.output_names, chunks, out):
                ax = self.output_batch_axes[name]
                if o.ndim <= ax or o.shape[ax] != m:
                    raise RuntimeError(
                        f"{self.name}: output {name!r} has shape {o.shape} for a chunk of {m} items; its "
                        f"batch axis {ax} does not have length {m}. Set output_batch_axes correctly.")
                lst.append(o)
        return {name: (lst[0] if len(lst) == 1 else np.concatenate(lst, axis=self.output_batch_axes[name]))
                for name, lst in zip(self.output_names, chunks)}

    def __repr__(self) -> str:
        return (f"OnnxModel({self.name!r}, device={self.device!r}, threads={self.threads}, "
                f"batch_size={self.batch_size})")
