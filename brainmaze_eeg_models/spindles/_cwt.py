"""
Vectorised, batched continuous wavelet transform used by the spindle detector.

It computes, in float64, what OpenSpindleNet 0.1.1 computes at inference time with
PyWavelets::

    np.abs(pywt.cwt(x, np.geomspace(135, 270, num=15), 'shan6-13', sampling_period=1/250)[0])

with the wavelet sampled at ``precision = 10`` (2**10 points on [-20, 20]), which is what
``pywt.cwt`` hard-coded up to PyWavelets 1.8. PyWavelets >= 1.9 samples it at 2**12 points by
default, which changes the scalogram substantially for this wavelet (13 cycles per unit are
aliased at 2**10 points over 40 units); the models were trained on the precision-10 version,
so it is pinned here and PyWavelets is not needed at runtime.

**Training used pywt's float32 path.** The training code (mayo_spindles) fed float32 signals,
and ``pywt.cwt`` then builds the kernel index grid with a float32 step, so 1-102 kernel
indices per scale differ from the float64 kernel used here (and by OpenSpindleNet's own
inference on float64 input). Measured on the two OpenSpindleNet sample windows: the
normalised scalogram differs by up to 0.57 SD (eeg) / 5.4 SD (ieeg) at single samples, the
model outputs by <= 0.006, and the detected intervals are identical. That is much smaller
than the precision-12 vs precision-10 difference; this module deliberately reproduces the
float64 path (the original package's runtime behaviour, pinned by the golden tests).

Instead of one ``np.convolve`` per scale and window (about 1 s per 30 s window), the
convolution of every window with every scale is done with FFTs in one batch. The result
equals ``pywt.cwt(..., method='conv', precision=10)`` up to floating-point rounding
(tests: relative error < 1e-10).

pywt computes, per scale ``s`` (in samples):

- ``int_psi``: the running integral of the conjugated wavelet, ``cumsum(conj(psi)) * dx``;
- ``k = int_psi[j][::-1]`` with ``j = floor(arange(s * 40 + 1) / (s * dx))`` (indices beyond
  the grid dropped);
- ``coef = -sqrt(s) * diff(convolve(x, k))``, trimmed symmetrically to ``len(x)``
  (``floor(d)`` samples at the start, ``ceil(d)`` at the end, ``d = (len(k) - 2) / 2``).

``diff(convolve(x, k)) == convolve(x, diff([0, *k, 0]))[1:-1]``, so every output sample is a
single (FFT) convolution with the differenced kernel. Zero padding at the window edges is the
same as in ``np.convolve`` (linear, not circular, convolution).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import fft as _fft

__all__ = ['SCALES', 'FS', 'scalogram', 'shan_wavefun']

FS = 250.0
SCALES = np.geomspace(135, 270, num=15)

# 'shan6-13': complex Shannon wavelet, bandwidth B = 6, centre frequency C = 13.
_B, _C = 6.0, 13.0
_LOWER, _UPPER = -20.0, 20.0     # pywt's support for 'shan'
_PRECISION = 10                  # pywt <= 1.8 hard-coded value (see the module docstring)


def shan_wavefun(precision: int = _PRECISION) -> tuple[np.ndarray, np.ndarray]:
    """``psi, x`` of pywt's ``ContinuousWavelet('shan6-13').wavefun(precision)``."""
    x = np.linspace(_LOWER, _UPPER, 2 ** precision)
    psi = np.sqrt(_B) * np.sinc(_B * x) * np.exp(2j * np.pi * _C * x)
    return psi, x


def _scale_kernel(scale: float, int_psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    step = x[1] - x[0]
    j = (np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)).astype(int)
    j = j[j < int_psi.size]
    return int_psi[j][::-1]


@lru_cache(maxsize=8)
def _plan(n: int, scales: tuple[float, ...], precision: int):
    """FFT length, kernel spectra (n_scales, nfft) and output offsets for windows of n samples."""
    psi, x = shan_wavefun(precision)
    int_psi = np.conj(np.cumsum(psi) * (x[1] - x[0]))
    kernels = [_scale_kernel(s, int_psi, x) for s in scales]
    lmax = max(k.size for k in kernels)
    nfft = _fft.next_fast_len(n + lmax, real=False)
    spectra = np.empty((len(scales), nfft), dtype=np.complex128)
    offsets = []
    for i, (s, k) in enumerate(zip(scales, kernels)):
        dk = np.diff(np.concatenate(([0.0], k, [0.0])))          # length L + 1
        spectra[i] = -np.sqrt(s) * _fft.fft(dk, nfft)
        d = (k.size - 2) / 2.0                                    # pywt's trim
        if d < 0:
            raise ValueError(f"Selected scale of {s} too small.")
        offsets.append(int(np.floor(d)) + 1)
    spectra.setflags(write=False)
    return nfft, spectra, tuple(offsets)


def scalogram(x: np.ndarray, *, scales=SCALES, precision: int = _PRECISION,
              workers: int | None = None) -> np.ndarray:
    """Magnitude of the 'shan6-13' CWT of each window, as OpenSpindleNet computes it.

    Parameters
    ----------
    x : array_like, shape (..., n)
        Real windows (time on the last axis), finite. Each window is transformed
        independently with zero padding at its edges (as ``pywt.cwt`` does).
    scales : array_like
        Scales in samples (default: OpenSpindleNet's 15 scales, 135-270).
    precision : int
        Wavelet sampling, 2**precision points (default 10 = what the models were trained on).
    workers : int, optional
        Threads for :mod:`scipy.fft` (default: scipy's default, 1).

    Returns
    -------
    np.ndarray, shape (..., n_scales, n), float64
        ``|coef|``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError(f"x must have time on its last axis, got shape {x.shape}")
    lead, n = x.shape[:-1], x.shape[-1]
    xb = x.reshape(-1, n)
    nfft, spectra, offsets = _plan(n, tuple(float(s) for s in np.atleast_1d(scales)), int(precision))
    out = np.empty((xb.shape[0], len(offsets), n), dtype=np.float64)
    X = _fft.fft(xb, nfft, axis=-1, workers=workers)
    for i, off in enumerate(offsets):
        y = _fft.ifft(X * spectra[i], axis=-1, workers=workers)
        np.abs(y[:, off:off + n], out=out[:, i, :])
    return out.reshape(*lead, len(offsets), n)
