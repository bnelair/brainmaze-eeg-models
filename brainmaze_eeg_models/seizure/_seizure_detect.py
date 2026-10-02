# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Seizure-probability inference (spectrogram CNN + BiLSTM) for iEEG signals, on ONNX Runtime.

Ported from brainmaze-torch 0.2.0 (``brainmaze_torch.seizure_detection._seizure_detect``): the
preprocessing, windowing, gap/flat masking and combination code is unchanged; only the model
runs on ONNX Runtime instead of PyTorch (see :mod:`brainmaze_eeg_models.seizure` for the
export and the parity with PyTorch).

Time grid convention used throughout this module
------------------------------------------------
The model consumes spectrograms computed with 1 s segments (``nperseg = fs``)
advanced by 0.5 s (``hop = fs // 2``), keeping the first 100 frequency bins
(0-99 Hz at 1 Hz resolution). Every spectrogram column -- and therefore every
probability value -- describes the 1 s of signal *centred* on its time stamp.
For a signal (or window) starting at sample 0, column ``j`` covers samples
``[j * hop, j * hop + fs)`` and is stamped ``(j + 1) * 0.5`` s.
"""
import warnings

import numpy as np
from scipy.signal import spectrogram

from ..runtime import OnnxModel
from ._models import load_trained_model

#: Number of spectrogram frequency bins (1 Hz each, 0..99 Hz) the model expects.
N_FREQ_BINS = 100
#: Minimum supported sampling rate in Hz (needs >= 100 one-Hz bins below Nyquist).
MIN_FS = 200
#: Index of the seizure class in the model's 4-class softmax output.
SEIZURE_CLASS_INDEX = 3


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _validate_fs(fs):
    """Return ``fs`` as an ``int`` or raise ``ValueError``.

    ``fs`` must be a whole, even number of Hz and at least :data:`MIN_FS`.
    Float values that are whole numbers (e.g. ``500.0`` read from a .mat or
    MEF header) are accepted and cast to ``int``.
    """
    if isinstance(fs, (bool, np.bool_)):
        raise ValueError(f"fs must be a number of Hz, got {fs!r}")
    try:
        fs_f = float(fs)
    except (TypeError, ValueError):
        raise ValueError(f"fs must be a number of Hz, got {fs!r}") from None
    if not np.isfinite(fs_f) or not fs_f.is_integer():
        raise ValueError(
            f"fs must be a whole number of Hz, got {fs!r}. The model works on a 1 Hz / 0.5 s "
            "spectrogram grid that requires an integer number of samples per second; resample "
            "the signal to an integer rate (e.g. with scipy.signal.resample_poly) first."
        )
    fs_i = int(fs_f)
    if fs_i < MIN_FS:
        raise ValueError(
            f"fs must be >= {MIN_FS} Hz, got {fs!r}. The model needs {N_FREQ_BINS} spectrogram "
            f"bins (0-{N_FREQ_BINS - 1} Hz at 1 Hz resolution), i.e. a Nyquist frequency of at "
            "least 100 Hz."
        )
    if fs_i % 2:
        raise ValueError(
            f"fs must be an even number of Hz, got {fs_i}. With an odd rate the 0.5 s spectrogram "
            "hop (fs/2 samples) is not an integer number of samples, so the output would drift off "
            "the 0.5 s output grid. Resample to an even rate first, e.g. "
            "scipy.signal.resample_poly(x, up=2, down=1) or to a standard rate such as 500 Hz."
        )
    return fs_i


def _to_half_seconds(value, name, minimum):
    """Convert a duration in seconds to an integer count of 0.5 s steps."""
    try:
        v2 = float(value) * 2.0
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number of seconds, got {value!r}") from None
    if not np.isfinite(v2) or abs(v2 - round(v2)) > 1e-9:
        raise ValueError(f"{name} must be a multiple of 0.5 s (the model's time step), got {value!r}")
    v2 = int(round(v2))
    if v2 < minimum:
        raise ValueError(f"{name} must be >= {minimum / 2} s, got {value!r}")
    return v2


def _resolve_model(model, use_cuda, cuda_number, device):
    """``model`` (name or loaded :class:`OnnxModel`) + device arguments -> an OnnxModel.

    Device: ``device`` if given; else ``use_cuda=True`` -> 'cuda' (GPU ``cuda_number``),
    ``use_cuda=False`` -> 'cpu'; both None -> 'auto'. A loaded model keeps its device; asking
    for a different one raises instead of silently running elsewhere.
    """
    if device is not None and use_cuda is not None and (device == 'cuda') != bool(use_cuda):
        raise ValueError(f"device={device!r} contradicts use_cuda={use_cuda!r}; pass only one of them")
    if device is None:
        device = 'auto' if use_cuda is None else ('cuda' if use_cuda else 'cpu')
    if isinstance(model, str):
        return load_trained_model(model, device=device, cuda_device_id=cuda_number)
    if not isinstance(model, OnnxModel):
        raise TypeError(
            "model must be 'modelA', 'modelB' or a model from load_trained_model(); PyTorch models "
            "are not supported (brainmaze-eeg-models runs on ONNX Runtime)")
    if device != 'auto' and model.device != device:
        raise ValueError(
            f"the loaded model runs on {model.device!r} but {device!r} was requested; load it with "
            f"load_trained_model(..., device={device!r})")
    return model


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def preprocess_input(x, fs, return_axes=False):
    """Convert raw signal windows into normalised spectrograms for the model.

    Each row of ``x`` is processed independently:

    1. z-score the raw signal (NaN-aware mean/std over the row; a row with zero
       or undefined std is only mean-centred, never divided by 0),
    2. replace NaN / +-inf samples by 0,
    3. spectrogram with ``nperseg = fs`` (1 s, 1 Hz bins) and a 0.5 s hop
       (``noverlap = fs // 2``), keeping the first 100 bins (0-99 Hz),
    4. z-score every frequency bin over time, using only columns with non-zero
       total power. Columns with zero power (e.g. a flat or zero-filled gap) are
       left at 0. If a bin has zero variance, or fewer than two columns have
       power, that bin is only mean-centred (no division by 0).

    Parameters
    ----------
    x : array_like, shape (n_samples,) or (batch_size, n_samples)
        Raw signal. A 1D input is treated as a single window. Rows are windows
        (or channels), columns are samples. Must contain at least ``fs``
        samples (1 s).
    fs : int or float
        Sampling rate in Hz. Must be a whole, even number >= 200; a float such
        as ``500.0`` is accepted. Odd or fractional rates raise ``ValueError``
        -- resample first (see Notes).
    return_axes : bool, optional
        If True, also return the time and frequency axes. Default False.

    Returns
    -------
    t : numpy.ndarray, shape (n_times,)
        Only if ``return_axes``. Time stamp in seconds of every column,
        relative to the first sample of the window: ``(j + 1) * 0.5`` -- the
        centre of the 1 s segment ``[j * 0.5, j * 0.5 + 1)``.
    f : numpy.ndarray, shape (100,)
        Only if ``return_axes``. Frequencies in Hz (0, 1, ..., 99).
    sxx : numpy.ndarray, shape (batch_size, 100, n_times)
        Normalised spectrograms, ``n_times = floor((n_samples - fs) / (fs // 2)) + 1``
        (= ``2 * duration_s - 1`` for whole-second inputs). Always finite.

    Raises
    ------
    ValueError
        If ``fs`` is invalid, ``x`` is not 1D/2D, or shorter than 1 s.

    Notes
    -----
    NaN samples are zero-filled *before* the spectrogram, so the model still
    sees a value there. This low-level function does not mark such columns;
    use :func:`predict_channel_seizure_probability`, which reports any 1 s
    segment containing NaN/inf (or a flat signal) as NaN in its output.

    Where the ``fs`` rules come from: they follow from the spectrogram grid,
    not from the training data (whose sampling rate is not documented in this
    repository). ``nperseg = fs`` needs a whole number of samples per second;
    the 0.5 s hop (``fs / 2`` samples) needs an even rate, otherwise the output
    would drift off the 0.5 s grid; and 100 one-Hz bins (0-99 Hz) need a
    Nyquist frequency of at least 100 Hz, i.e. ``fs >= 200``.

    Anti-aliasing caveat at 200-256 Hz: acquisition (or resampling) anti-alias
    filters usually start attenuating at about 0.4 * fs, i.e. 80-100 Hz at
    these rates. The upper spectrogram bins then hold less power than in
    recordings with a higher native rate, which the model never saw in that
    form: a silent domain shift. Prefer a native rate >= 250-500 Hz, and when
    you resample, keep the new rate high enough that the anti-alias filter's
    transition band lies above 100 Hz.
    """
    fs = _validate_fs(fs)
    x = np.array(x, dtype=np.float64)  # always a copy; never modifies caller data
    if x.ndim == 1:
        x = x.reshape((1, -1))
    if x.ndim != 2:
        raise ValueError(f"x must be 1D (n_samples,) or 2D (batch_size, n_samples), got shape {x.shape}")
    if x.shape[1] < fs:
        raise ValueError(f"x must contain at least fs={fs} samples (1 s), got {x.shape[1]}")

    hop = fs // 2
    x[~np.isfinite(x)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows / ddof warnings
        mu = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
    mu = np.nan_to_num(mu, nan=0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    x = np.nan_to_num((x - mu) / std, nan=0.0, posinf=0.0, neginf=0.0)

    f, _, sxx = spectrogram(x, fs, nperseg=fs, noverlap=fs - hop, axis=1)
    sxx = np.ascontiguousarray(sxx[:, :N_FREQ_BINS, :])

    for kk in range(sxx.shape[0]):
        xx = sxx[kk]
        valid = np.sum(xx, axis=0) != 0
        if not valid.any():
            continue
        sub = xx[:, valid]
        mu_f = sub.mean(axis=1, keepdims=True)
        sd_f = sub.std(axis=1, keepdims=True)
        sd_f = np.where(sd_f > 0, sd_f, 1.0)
        xx[:, valid] = (sub - mu_f) / sd_f

    if return_axes:
        t = (np.arange(sxx.shape[2]) + 1) * 0.5
        return t, f[:N_FREQ_BINS], sxx
    return sxx


def infer_seizure_probability(x, model='modelA', use_cuda=None, cuda_number=0, *, device=None):
    """Run the seizure model on a batch of preprocessed spectrograms.

    Parameters
    ----------
    x : array_like, shape (batch_size, 100, n_times)
        Output of :func:`preprocess_input`. Must be finite and have
        ``n_times >= 2``. Recommended window length is 300 s (599 columns).
    model : str or OnnxModel, optional
        ``'modelA'`` (default), ``'modelB'``, or a model from
        :func:`load_trained_model` (then the device arguments must agree with the
        device it was loaded on).
    use_cuda : bool, optional
        ``True``: run on the GPU ``cuda_number`` (error if CUDA is unusable);
        ``False``: run on the CPU. Default None: ``device``, else 'auto'
        (GPU if usable, else CPU). Kept for compatibility with brainmaze-torch.
    cuda_number : int, optional
        CUDA device index. Default 0.
    device : {'auto', 'cpu', 'cuda'}, optional
        Alternative to ``use_cuda`` (keyword only).

    Returns
    -------
    numpy.ndarray, shape (batch_size, n_times), float32
        Seizure probability (softmax probability of class index 3 of the
        model's 4 output classes) for every spectrogram column. Column ``j``
        corresponds to the time stamp ``t[j]`` returned by
        :func:`preprocess_input` (``(j + 1) * 0.5`` s from window start).
        Values are in [0, 1].

    Raises
    ------
    ValueError
        If ``x`` has the wrong shape or contains non-finite values.

    Notes
    -----
    The ONNX model gives the same probabilities as the PyTorch model of brainmaze-torch
    0.2.0 to within float32 rounding (parity tests: max abs difference ~1e-6).
    """
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1] != N_FREQ_BINS or x.shape[2] < 2:
        raise ValueError(
            f"x must have shape (batch_size, {N_FREQ_BINS}, n_times >= 2) as returned by "
            f"preprocess_input, got {x.shape}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN/inf; preprocess_input always returns finite spectrograms")

    model = _resolve_model(model, use_cuda, cuda_number, device)
    probs = model.run({'x': np.ascontiguousarray(x, dtype=np.float32)})['probs']  # (n_times, batch, 4)
    return np.ascontiguousarray(probs[:, :, SEIZURE_CLASS_INDEX].T)


def predict_channel_seizure_probability(
        x, fs, model='modelA', use_cuda=None, cuda_number=0, n_batch=128,
        window_s=300, step_s=20, discard_edges_s=10,
        min_valid_fraction=0.0, fill_recording_edges=True, *, device=None,
):
    """Continuous seizure-probability trace for one (long) iEEG channel.

    The signal is cut into overlapping windows of ``window_s`` seconds every
    ``step_s`` seconds (the "regular" windows of the published method). Each
    window is converted to a spectrogram and run through the model;
    ``discard_edges_s`` seconds are dropped at both window edges (the BiLSTM
    has little context there), and overlapping windows are combined with a
    NaN-ignoring maximum. If the recording end does not fall on the step
    grid, one extra window aligned to the recording end covers the tail; it
    only fills time that no regular window covers and never changes a value
    produced by the regular windows. With the default parameters the output
    is therefore identical to the published method (brainmaze-torch <= 0.1.1)
    wherever that method produced an estimate, except that invalid (gap /
    flat) segments are NaN instead of a number.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Single-channel raw signal (1D). Lists and integer arrays are accepted.
        NaN / +-inf mark missing data (gaps).
    fs : int or float
        Sampling rate in Hz. Must be a whole, even number >= 200 Hz; a float
        such as ``500.0`` is accepted. Resample other rates first.
    model : str or OnnxModel, optional
        ``'modelA'`` (published model), ``'modelB'`` (extended training set)
        or a model from :func:`load_trained_model`. Default ``'modelA'``.
    use_cuda : bool, optional
        ``True``: GPU ``cuda_number`` (error if CUDA is unusable); ``False``: CPU.
        Default None: ``device``, else 'auto' (GPU if usable, else CPU).
    cuda_number : int, optional
        CUDA device index. Default 0.
    n_batch : int, optional
        Windows per inference batch (memory/speed trade-off only; does not
        change the result). Default 128.
    window_s : float, optional
        Window length in seconds; multiple of 0.5, >= 2. Default 300 (the
        length the model was designed for). The recording must be at least
        this long.
    step_s : float, optional
        Step between window starts in seconds; multiple of 0.5. Must satisfy
        ``step_s <= window_s - 2 * discard_edges_s - 0.5`` so that the kept
        (non-discarded) parts of consecutive windows leave no hole (a window
        has ``2 * window_s - 1`` half-second columns). Default 20.
    discard_edges_s : float, optional
        Seconds dropped at each window edge; multiple of 0.5, may be 0.
        Default 10.
    min_valid_fraction : float, optional
        Window preference near gaps. A window is *preferred* if at least this
        fraction of its 1 s segments (spectrogram columns) are valid (finite
        samples, not flat). A valid segment gets the maximum over the
        preferred windows covering it; only if none of them covers it, the
        maximum over all windows covering it is used. This never turns a valid
        segment into NaN; it only lets gap-heavy (mostly zero-filled) windows
        be ignored where a better window exists. Default 0.0: every window is
        preferred, i.e. the published method. Values > 0 change probabilities
        next to long gaps compared with the published method (not validated).
        The choice is made per time point, so the result is **not monotone in
        this value**: a point switches from "preferred windows" to the
        fallback (all windows) as soon as the last preferred window covering
        it drops below the threshold, and different points switch at different
        values. If the value exceeds the valid fraction of every window, no
        window is preferred and the result equals that of 0.0. Choose one
        value for a study and do not interpolate between values.
    fill_recording_edges : bool, optional
        The first and last ``discard_edges_s`` seconds of the *recording*
        lie in no window's kept interior. If True (default), they are taken
        from the edge of the first/last window (shorter model context, hence
        less reliable). If False, they are NaN.
    device : {'auto', 'cpu', 'cuda'}, optional
        Alternative to ``use_cuda`` (keyword only).

    Returns
    -------
    t : numpy.ndarray, shape (n_out,)
        Time in seconds from the first sample: ``t[k] = k * 0.5`` (so the
        value for time ``t`` is at index ``2 * t``),
        ``n_out = n_samples // (fs // 2)``. The last stamp is the centre of
        the last complete 1 s segment.
    prob : numpy.ndarray, shape (n_out,), float64
        Seizure probability in [0, 1] for the 1 s segment centred on
        ``t[k]``, or **NaN where no estimate exists**. NaN is returned for:

        * ``t = 0`` (a 1 s segment centred at 0 s would start before the data),
        * every 1 s segment containing at least one NaN/inf sample (gaps),
        * every 1 s segment over which the signal is constant (flat line,
          e.g. disconnected or saturated channel),
        * the first/last ``discard_edges_s`` s if ``fill_recording_edges`` is
          False.

        NaN therefore always means "not evaluated", never "no seizure"; a
        value is never 0 merely because a time point was not covered. Every
        other (valid) segment always gets a value, whatever
        ``min_valid_fraction`` is.

    Raises
    ------
    ValueError
        If ``x`` is not 1D, ``fs`` is invalid, the window parameters are
        inconsistent, or the recording is shorter than ``window_s``.

    Notes
    -----
    * Gaps are not interpolated. Within an evaluated window the gap samples
      are zero-filled for the model (as in the original implementation),
      which can influence the probability of valid segments next to a gap via
      the BiLSTM context; only the gap segments themselves are masked to NaN.
      To fill short gaps instead (e.g. by interpolation), do so before calling
      this function; filled samples are then treated as valid data. See
      ``min_valid_fraction`` to ignore mostly-gap windows where possible.
    * Probabilities of overlapping windows are combined with a NaN-ignoring
      maximum (``np.fmax``), as in the published method.

    Examples
    --------
    >>> import numpy as np
    >>> from brainmaze_eeg_models.seizure import predict_channel_seizure_probability
    >>> fs = 500
    >>> x = np.random.randn(fs * 600)          # 10 min, single channel
    >>> t, p = predict_channel_seizure_probability(x, fs, model='modelA')
    >>> t.shape == p.shape, float(t[1] - t[0])
    (True, 0.5)
    """
    fs = _validate_fs(fs)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"input x should be single channel signal (1D array), got shape {x.shape}")

    hop = fs // 2
    w2 = _to_half_seconds(window_s, 'window_s', minimum=4)       # >= 2 s
    s2 = _to_half_seconds(step_s, 'step_s', minimum=1)
    e2 = _to_half_seconds(discard_edges_s, 'discard_edges_s', minimum=0)
    n_cols = w2 - 1                       # spectrogram columns per window
    n_kept = n_cols - 2 * e2
    if n_kept < 1:
        raise ValueError(
            f"discard_edges_s={discard_edges_s} leaves nothing of window_s={window_s}; need "
            f"2 * discard_edges_s <= window_s - 1"
        )
    if s2 > n_kept:
        raise ValueError(
            f"step_s={step_s} is too large for window_s={window_s} and discard_edges_s="
            f"{discard_edges_s}: the kept parts of consecutive windows would leave uncovered "
            f"time. Need step_s <= window_s - 2 * discard_edges_s - 0.5 = {n_kept / 2} s."
        )
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError(f"min_valid_fraction must be within [0, 1], got {min_valid_fraction!r}")
    n_batch = int(n_batch)
    if n_batch < 1:
        raise ValueError(f"n_batch must be >= 1, got {n_batch}")

    n = x.shape[0]
    w = w2 * hop                          # window length in samples (= window_s * fs)
    if n < w:
        raise ValueError(
            f"recording is shorter ({n / fs:.2f} s) than window_s={window_s} s; pass a smaller "
            f"window_s (the model was designed for 300 s windows)"
        )

    # ---- per-1-s-segment validity on the output grid --------------------
    # Grid index k (t = k / 2) <-> segment covering samples [(k-1)*hop, (k+1)*hop),
    # i.e. half-second blocks k-1 and k.
    n_out = n // hop
    blocks = x[:n_out * hop].reshape(n_out, hop)
    blk_finite = np.isfinite(blocks).all(axis=1)
    blk_max = np.fmax.reduce(blocks, axis=1)
    blk_min = np.fmin.reduce(blocks, axis=1)
    seg_valid = np.zeros(n_out, dtype=bool)
    seg_valid[1:] = (
        blk_finite[:-1] & blk_finite[1:]
        & (np.fmax(blk_max[:-1], blk_max[1:]) > np.fmin(blk_min[:-1], blk_min[1:]))
    )

    # ---- window starts in half-second blocks ---------------------------
    # Regular windows start every step_s (exactly the windows of the published
    # method). If the recording end is not on that grid, one extra "tail" window
    # aligned to the recording end is added; it is only used to fill time that no
    # regular window covers, so it never changes a value the regular windows give.
    last_start = (n - w) // hop
    reg_starts = np.arange(0, last_start + 1, s2)
    has_tail = reg_starts[-1] != last_start
    starts = np.append(reg_starts, last_start) if has_tail else reg_starts
    is_tail = np.zeros(starts.size, dtype=bool)
    is_tail[-1] = has_tail

    # Output slice of every window: its kept (non-discarded) columns.
    lo_arr = np.where(fill_recording_edges & (starts == 0), 0, e2)
    hi_arr = np.where(fill_recording_edges & (starts == last_start), n_cols, n_cols - e2)
    frac_valid = np.array([seg_valid[b0 + 1:b0 + 1 + n_cols].mean() for b0 in starts])
    # A window whose kept part holds no valid segment can only contribute NaN: skip it.
    useful = np.array([seg_valid[b0 + 1 + lo:b0 + 1 + hi].any()
                       for b0, lo, hi in zip(starts, lo_arr, hi_arr)])
    preferred = frac_valid >= float(min_valid_fraction)

    t_out = np.arange(n_out) * 0.5
    prob_pref = np.full(n_out, np.nan)   # max over regular windows with enough valid data
    prob_all = np.full(n_out, np.nan)    # max over all regular windows
    prob_tail = np.full(n_out, np.nan)   # tail window
    # Resolve the model before any early return, so device errors never depend on the data.
    model = _resolve_model(model, use_cuda, cuda_number, device)
    sel = np.flatnonzero(useful)
    if sel.size == 0:
        return t_out, prob_pref

    for b in range(0, sel.size, n_batch):
        batch = sel[b:b + n_batch]
        xb = np.stack([x[starts[i] * hop:starts[i] * hop + w] for i in batch])
        pxx = preprocess_input(xb, fs)
        prob = infer_seizure_probability(pxx, model)
        if prob.shape[1] != n_cols:  # pragma: no cover - internal consistency check
            raise RuntimeError(f"unexpected number of model outputs {prob.shape[1]} != {n_cols}")

        for i, p in zip(batch, prob):
            b0, lo, hi = starts[i], lo_arr[i], hi_arr[i]
            idx = slice(b0 + 1 + lo, b0 + 1 + hi)
            p = p[lo:hi].astype(np.float64)
            p[~seg_valid[idx]] = np.nan
            if is_tail[i]:
                prob_tail[idx] = p
                continue
            prob_all[idx] = np.fmax(prob_all[idx], p)
            if preferred[i]:
                prob_pref[idx] = np.fmax(prob_pref[idx], p)

    # Priority: preferred regular windows > any regular window > tail window.
    # With min_valid_fraction=0 every regular window is preferred, so the result
    # equals the published method wherever a regular window's kept part lies.
    prob_out = np.where(np.isnan(prob_pref), prob_all, prob_pref)
    prob_out = np.where(np.isnan(prob_out), prob_tail, prob_out)
    return t_out, prob_out
