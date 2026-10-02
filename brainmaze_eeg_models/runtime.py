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
  chunks of ``batch_size`` and concatenates the outputs, so memory stays bounded for long
  recordings.
- **integrity**: an optional SHA-256 of the model file is verified before loading, so a
  truncated or swapped model file fails loudly instead of producing wrong numbers.
- **inputs**: every input is cast to the dtype the model declares and must be finite (NaN or
  inf reaching a network silently corrupts its output; the detectors handle gaps before).

GPU installation
----------------
``onnxruntime`` (CPU) and ``onnxruntime-gpu`` install the **same** Python module
``onnxruntime``; with both installed, whichever was installed last wins and uninstalling one
breaks the other. For NVIDIA GPUs::

    pip install brainmaze-eeg-models
    pip uninstall -y onnxruntime
    pip install "onnxruntime-gpu[cuda,cudnn]"    # CUDA + cuDNN as pip packages

The CUDA / cuDNN major versions must match the ``onnxruntime-gpu`` build (see the
`ONNX Runtime CUDA requirements <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_);
:func:`device_report` prints what is installed and usable.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _md
import os
import warnings
from typing import Mapping, Sequence

import numpy as np

__all__ = ['OnnxModel', 'available_providers', 'device_report', 'DEVICES']

DEVICES = ('auto', 'cpu', 'cuda')

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
            "and conflict. Fix: pip uninstall -y onnxruntime onnxruntime-gpu && "
            "pip install \"onnxruntime-gpu[cuda,cudnn]\"")
    elif not has_gpu:
        lines.append(
            "The CPU build is installed. For NVIDIA GPUs: pip uninstall -y onnxruntime && "
            "pip install \"onnxruntime-gpu[cuda,cudnn]\" (or `pip install brainmaze-eeg-models[gpu]` "
            "followed by the uninstall/reinstall described in the README).")
    else:
        lines.append(
            "onnxruntime-gpu is installed but the CUDA execution provider could not be "
            "initialised: an NVIDIA driver, and CUDA + cuDNN libraries matching this "
            "onnxruntime-gpu build, are required (pip install \"onnxruntime-gpu[cuda,cudnn]\" "
            "installs matching CUDA/cuDNN wheels). See "
            "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements")
    lines.append("Use device='cpu' (or 'auto') to run on the CPU.")
    return "\n".join(lines)


def _preload_cuda_libraries() -> None:
    """Load CUDA/cuDNN from the nvidia-* pip wheels if present (onnxruntime >= 1.21)."""
    preload = getattr(_ort(), 'preload_dlls', None)
    if preload is None:
        return
    try:
        preload()
    except Exception:  # noqa: BLE001 - best effort; the session check below decides
        pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def device_report() -> str:
    """Human-readable summary of the installed onnxruntime distributions and providers."""
    ort = _ort()
    return "\n".join([
        f"onnxruntime module {ort.__version__} ({os.path.dirname(ort.__file__)})",
        f"installed distributions: {', '.join(_installed_ort_distributions()) or 'none'}",
        f"available providers: {', '.join(ort.get_available_providers())}",
        f"device='auto' would try: {'cuda, then cpu' if _CUDA in ort.get_available_providers() else 'cpu'}",
    ])


class OnnxModel:
    """One ONNX model on one device.

    Parameters
    ----------
    path : str
        Path to the ``.onnx`` file.
    device : {'auto', 'cpu', 'cuda'}
        ``'auto'``: CUDA when the CUDA execution provider is installed and initialises, else
        CPU (a warning is issued when CUDA is installed but unusable). ``'cpu'``: always CPU.
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
        rounding) on Ampere and newer GPUs; off, GPU and CPU outputs agree to float32 rounding.
    sha256 : str, optional
        Expected SHA-256 hex digest of the file; a mismatch raises :class:`RuntimeError`.
    name : str, optional
        Name used in messages.
    """

    def __init__(self, path: str, *, device: str = 'auto', threads: int | None = None,
                 batch_size: int = 32, cuda_device_id: int = 0, cuda_tf32: bool = False,
                 sha256: str | None = None,
                 name: str | None = None):
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
            self._session = session if session is not None else self._make_session([_CPU])
        self.device = 'cuda' if self._session.get_providers()[0] == _CUDA else 'cpu'

        self._inputs = {i.name: i for i in self._session.get_inputs()}
        self.input_names = [i.name for i in self._session.get_inputs()]
        self.output_names = [o.name for o in self._session.get_outputs()]

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

    def _prepare(self, inputs: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
        missing = [n for n in self.input_names if n not in inputs]
        extra = [n for n in inputs if n not in self._inputs]
        if missing or extra:
            raise ValueError(f"{self.name}: inputs {sorted(inputs)} do not match the model inputs "
                             f"{self.input_names} (missing {missing}, unexpected {extra})")
        feed, n = {}, None
        for k in self.input_names:
            meta = self._inputs[k]
            dtype = _ORT_DTYPES.get(meta.type)
            if dtype is None:
                raise TypeError(f"{self.name}: unsupported input type {meta.type} for {k!r}")
            v = np.ascontiguousarray(inputs[k], dtype=dtype)
            if v.ndim != len(meta.shape):
                raise ValueError(f"{self.name}: input {k!r} must have {len(meta.shape)} dimensions "
                                 f"{meta.shape}, got shape {v.shape}")
            for ax, (want, got) in enumerate(zip(meta.shape, v.shape)):
                if ax > 0 and isinstance(want, int) and want != got:
                    raise ValueError(f"{self.name}: input {k!r} must have shape {meta.shape}, got {v.shape}")
            if np.issubdtype(v.dtype, np.floating) and not np.isfinite(v).all():
                raise ValueError(f"{self.name}: input {k!r} contains NaN or inf; the model would "
                                 "return silently wrong values. Handle gaps before inference.")
            if n is None:
                n = v.shape[0]
            elif v.shape[0] != n:
                raise ValueError(f"{self.name}: all inputs must have the same batch size "
                                 f"(first axis); got {n} and {v.shape[0]} for {k!r}")
            feed[k] = v
        if not n:
            raise ValueError(f"{self.name}: empty batch (first axis has length 0)")
        return feed, n

    def run(self, inputs: Mapping[str, np.ndarray], batch_size: int | None = None) -> dict[str, np.ndarray]:
        """Run the model on a batch, in chunks along the first axis.

        Parameters
        ----------
        inputs : mapping name -> array
            One array per model input (:attr:`input_names`), all with the same length of the
            first (batch) axis, at least 1. Cast to the model's dtype; must be finite.
        batch_size : int, optional
            Items per inference call; default :attr:`batch_size`. Results do not depend on
            it beyond float32 rounding (a few ulp; the kernels may block differently).

        Returns
        -------
        dict name -> np.ndarray
            One array per model output (:attr:`output_names`), concatenated along axis 0.
        """
        bs = self.batch_size if batch_size is None else self._check_batch_size(batch_size)
        feed, n = self._prepare(inputs)
        chunks: list[list[np.ndarray]] = [[] for _ in self.output_names]
        for i in range(0, n, bs):
            out = self._session.run(self.output_names, {k: v[i:i + bs] for k, v in feed.items()})
            for lst, o in zip(chunks, out):
                lst.append(o)
        return {name: (lst[0] if len(lst) == 1 else np.concatenate(lst, axis=0))
                for name, lst in zip(self.output_names, chunks)}

    def __repr__(self) -> str:
        return (f"OnnxModel({self.name!r}, device={self.device!r}, threads={self.threads}, "
                f"batch_size={self.batch_size})")
