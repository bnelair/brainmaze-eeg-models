"""Sleep-spindle detection with OpenSpindleNet on long, multichannel, gapped recordings."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from brainmaze_utils.gaps import fill_gaps, gap_intervals, mask_in_gaps

from ..runtime import OnnxModel
from ._cwt import scalogram
from ._decode import SEGMENTS, decode_window, interval_nms, sigmoid

__all__ = ['SpindleDetector', 'SpindleDetections', 'MODELS', 'FS', 'WINDOW', 'DEMEAN_AUTO']

FS = 250.0            # model sampling rate (Hz)
WINDOW = 7500         # model input length (samples) = 30 s at 250 Hz
WINDOW_S = WINDOW / FS

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_models')
#: Bundled OpenSpindleNet models: name -> (file, SHA-256). 'eeg': scalp EEG, 'ieeg': intracranial EEG.
MODELS = {
    'eeg': ('spindle-detector-eeg.onnx', '8368ef11554dda32178d65320c5b4690456c5c6cebd207d4ba0e68d3d5179e81'),
    'ieeg': ('spindle-detector-ieeg.onnx', '75291ee435ff08360f869ee7279f00d61de557d2a37c57d860f1257e228ba0e8'),
}

_RESAMPLE_RTOL = 1e-4     # max relative error of an approximate rational resampling ratio
_RESAMPLE_MAX = 10 ** 5   # max up/down factor of resample_poly (its FIR has ~20 * max taps)
FS_MIN = 50.0             # lowest accepted sampling rate (DREAMS excerpt 3, in the eeg training set)
#: demean='auto': per model, what matches its training data (see SpindleDetector, ``demean``).
DEMEAN_AUTO = {'eeg': True, 'ieeg': False}
_OFFSET_WARN_SD = 5.0     # warn (demean off) when a window's |mean| exceeds this many SDs


@dataclass(frozen=True)
class SpindleDetections:
    """Spindles found in a recording, one row per spindle, times in seconds from the first sample.

    Attributes
    ----------
    start, end : np.ndarray, float64
        Interval of each spindle (s). Sorted by channel, then start.
    confidence : np.ndarray, float64
        Detection confidence in [0, 1] (the model's sigmoid output).
    channel : np.ndarray, int64
        Channel index (row of the input; 0 for 1-D input).
    not_evaluated : tuple of np.ndarray
        Per channel, ``(k, 2)`` ``[start, stop)`` intervals (s) where **no detection is
        possible**: NaN/inf gaps and flat segments widened by ``gap_margin_s``, plus the time
        owned by windows that were skipped (too little valid data, or flat). An empty result
        there means "not evaluated", never "no spindles".
    n_channels : int
    duration_s : float
        Recording duration ``n_samples / fs``.
    model : str
    """
    start: np.ndarray
    end: np.ndarray
    confidence: np.ndarray
    channel: np.ndarray
    not_evaluated: tuple
    n_channels: int
    duration_s: float
    model: str
    params: dict = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return int(self.start.size)

    def to_array(self) -> np.ndarray:
        """``(n, 4)`` float64 array ``[start, end, confidence, channel]``."""
        return np.column_stack([self.start, self.end, self.confidence, self.channel.astype(np.float64)])

    def channel_intervals(self, channel: int = 0) -> np.ndarray:
        """``(n, 3)`` ``[start, end, confidence]`` of one channel, sorted by start."""
        m = self.channel == channel
        return np.column_stack([self.start[m], self.end[m], self.confidence[m]])

    def evaluated_fraction(self, channel: int = 0) -> float:
        """Fraction of the recording of ``channel`` that was evaluated."""
        ne = self.not_evaluated[channel]
        return float(1.0 - (ne[:, 1] - ne[:, 0]).sum() / self.duration_s) if self.duration_s > 0 else 0.0

    def to_dataframe(self):
        """pandas DataFrame with columns start, end, duration, confidence, channel."""
        import pandas as pd
        return pd.DataFrame({'start': self.start, 'end': self.end, 'duration': self.end - self.start,
                             'confidence': self.confidence, 'channel': self.channel})


def _merge_intervals(iv: np.ndarray) -> np.ndarray:
    iv = np.asarray(iv, dtype=np.float64).reshape(-1, 2)
    iv = iv[iv[:, 1] > iv[:, 0]]
    if iv.size == 0:
        return iv
    iv = iv[np.argsort(iv[:, 0], kind='stable')]
    out = [iv[0].copy()]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append(np.array([s, e]))
    return np.array(out)


def _flat_runs(x: np.ndarray, fs: float, flat_s: float) -> np.ndarray:
    """``[start, stop)`` sample indices of constant runs (one finite value repeated) lasting >= flat_s.

    A run is a stretch of *pairwise* equal consecutive samples: ``x[i] == x[i + 1]`` for every
    ``i`` in it. Adjacent plateaus of different values (a staircase, e.g. sample-and-hold
    upsampling or coarse quantisation) are therefore separate runs, never one.
    """
    n_min = max(2, int(np.ceil(flat_s * fs)))
    if x.size < n_min:
        return np.empty((0, 2), dtype=np.int64)
    eq = (x[1:] == x[:-1]) & np.isfinite(x[1:])        # eq[i]: x[i] == x[i + 1]
    d = np.diff(np.concatenate(([0], eq.astype(np.int8), [0])))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)   # eq[a:b] all True
    stops = b + 1                                       # constant samples x[a:b + 1]
    keep = (stops - a) >= n_min
    return np.column_stack([a[keep], stops[keep]]).astype(np.int64)


def _resample_ratio(fs: float) -> tuple[int, int, float]:
    """``up, down, fs_out`` for :func:`scipy.signal.resample_poly` from ``fs`` to ~250 Hz.

    The exact ratio is used whenever it is small enough (``max(up, down) <= 1e5``; e.g.
    256 Hz -> 125/128, 2048 Hz -> 125/1024, 32556 Hz -> 125/16278), so ``fs_out == 250``.
    Otherwise (e.g. a measured rate such as 499.907 Hz) the smallest-denominator ratio within
    1e-4 (relative) of 250 Hz is used and ``fs_out`` is the resulting exact effective rate
    (the time axis uses it, so there is no drift). Raises :class:`ValueError` if no usable
    ratio exists (absurdly high ``fs``).
    """
    fr = Fraction(fs).limit_denominator(10 ** 6)
    if float(fr) == fs:                          # fs is (a float of) a small rational number
        exact = Fraction(FS) / fr
        if 1 <= max(exact.numerator, exact.denominator) <= _RESAMPLE_MAX:
            return exact.numerator, exact.denominator, FS
    target = Fraction(FS) / Fraction(fs)
    for limit in (100, 1000, 10000, _RESAMPLE_MAX):
        r = target.limit_denominator(limit)
        if r.numerator >= 1 and r.numerator <= _RESAMPLE_MAX and abs(float(r) * fs - FS) / FS <= _RESAMPLE_RTOL:
            return r.numerator, r.denominator, fs * r.numerator / r.denominator
    raise ValueError(f"cannot resample fs={fs} Hz to 250 Hz with a rational ratio of at most "
                     f"{_RESAMPLE_MAX}; downsample the recording first (e.g. to 250-2000 Hz, with "
                     "an anti-aliasing filter)")


def _normalize(a: np.ndarray) -> np.ndarray:
    """Per-row z-score over the last axis, as OpenSpindleNet (flat rows -> 0)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (a - a.mean(axis=-1, keepdims=True)) / a.std(axis=-1, keepdims=True)
    return np.nan_to_num(z, nan=0.0)


class SpindleDetector:
    """Sleep-spindle detector (OpenSpindleNet) for long, multichannel recordings with gaps.

    The models were trained on 30 s windows at 250 Hz (Seják et al. 2025, see *Credits*).
    :meth:`detect` takes a whole recording at any sampling rate and

    1. treats NaN/inf samples, and runs of identical values lasting >= ``flat_s`` (a
       disconnected, clipped or zero-padded channel), as gaps;
    2. fills the gaps (:func:`brainmaze_utils.gaps.fill_gaps`, spectral noise) so that the
       filters and the network see a plausible background;
    3. resamples to 250 Hz with :func:`scipy.signal.resample_poly` (its polyphase FIR is the
       anti-aliasing low-pass; cut-off at the lower Nyquist rate, 125 Hz when downsampling;
       exact ratio where one exists, e.g. 256 Hz -> 125/128). The training data were
       resampled with ``np.interp`` (linear interpolation, no anti-aliasing); the better filter
       does not change the results measurably: on a 6.8 h 500 Hz night 854 vs 861 spindles
       with 93-94 % mutual agreement (the same order as step 20 s vs 30 s), on DREAMS F1 0.570
       vs 0.572;
    4. cuts sliding 30 s windows every ``step_s`` seconds (the last one aligned to the end),
       computes the model inputs exactly as OpenSpindleNet (z-scored signal + z-scored
       'shan6-13' scalogram, see :func:`brainmaze_eeg_models.spindles.scalogram`) and runs
       the network in batches;
    5. decodes each window as OpenSpindleNet does (confidence threshold, per-window NMS);
       each window keeps only the spindles whose centre lies in the part of the recording it
       *owns* (the middle of the window; window boundaries are half-way between window
       centres), so the network's poorer view at window edges is not used; overlapping
       detections of neighbouring windows are then merged by NMS;
    6. drops every spindle that overlaps a gap widened by ``gap_margin_s`` and reports the
       unevaluated time per channel (:attr:`SpindleDetections.not_evaluated`).

    Parameters
    ----------
    model : {'eeg', 'ieeg'}
        ``'eeg'``: scalp EEG model; ``'ieeg'``: intracranial EEG model.
    device : {'auto', 'cpu', 'cuda'}
        See :class:`brainmaze_eeg_models.runtime.OnnxModel`.
    threads : int, optional
        CPU threads for the network and the FFTs (None: all cores).
    batch_size : int
        Windows per inference call.
    confidence_threshold : float
        Minimum confidence of a detection (OpenSpindleNet default 0.5).
    nms_iou_threshold : float
        Overlapping detections with IoU >= this are merged, keeping the most confident one
        (OpenSpindleNet default 0.3).
    step_s : float
        Hop between windows, in (0, 30] s. Default 20 s: 10 s overlap, every window keeps the
        spindles centred in its middle ~20 s (>= 5 s away from its edges, except at the
        recording edges). Smaller steps cost proportionally more time.
    min_valid_fraction : float
        A window with less than this fraction of valid (non-gap) samples is not evaluated.
    gap_margin_s : float
        Spindles overlapping a gap widened by this many seconds on each side are dropped.
    flat_s : float or None
        Runs of identical consecutive values at least this long (s) are treated as gaps.
        ``None`` disables the check.
    demean : {'auto', True, False}
        Subtract each window's mean before the scalogram. The scalogram kernels (up to 43 s)
        are longer than the window, so a DC offset leaks into every scalogram sample (the
        z-scored raw-signal input is not affected). ``'auto'`` (default) does what matches
        each model's training data: **on for 'eeg'**, **off for 'ieeg'**
        (:data:`DEMEAN_AUTO`). Measured effects:

        - 'eeg' (scalp data have small offsets; DREAMS <= 0.02 SD): demeaning changes nothing
          measurable (DREAMS F1 identical, the 6.8 h demo night gives the same 854 spindles)
          and protects against offsets, which without it change the confidences on the
          bundled eeg sample by up to 0.01 at 2 SD and 0.13 at 10 SD, and suppress most
          detections at 100 SD.
        - 'ieeg': the training data were raw MEF signals (no filtering, no demeaning), so DC
          offsets were part of the training distribution, and this model is much more
          sensitive to them: on the bundled iEEG sample (offset -1.89 SD) an added offset of
          0.5 / 1 / 3 / 5 SD changes the confidences by up to 0.07 / 0.10 / 0.32 / 0.57, and
          demeaning that sample changes them by up to 0.055 (segmentation 0.25, interval
          edges by up to 9 samples). Nobody has validated either choice on labelled iEEG, so
          the default keeps the training behaviour.

        **Limitation:** with demeaning off, the outputs depend on the recording's DC offset
        (amplifier offset, slow drift, reference choice). :meth:`detect` records the largest
        ``|window mean| / window SD`` of the evaluated windows in
        ``params['max_abs_offset_sd']`` and warns when it exceeds 5 (the bundled iEEG sample,
        from the training domain, has 1.9; 3.1 after its zero-filled dropout is filled, so a
        few SD are in-distribution; how far the training data went is not known). For such data, high-pass filter beforehand (e.g.
        0.3-0.5 Hz) or decide explicitly with ``demean=True``. ``False`` reproduces the
        original package exactly.

    Notes
    -----
    Differences from the ``openspindlenet`` package (0.1.1), all deliberate:

    - The wavelet is sampled as in PyWavelets <= 1.8 (``precision=10``), the version the
      models were trained with. ``openspindlenet`` with PyWavelets >= 1.9 computes a
      different scalogram (different detections) without any warning.
    - ``openspindlenet`` silently drops a spindle whose interval starts exactly at the first
      sample of the 30 s input; here it is kept.
    - NaN input is not z-scored to zeros silently: gaps are filled, then the spindles in and
      next to them are dropped and the time is reported as not evaluated. Flat runs
      (zero-filled dropouts, disconnected or saturated channels) are treated as gaps.
    - With the 'eeg' model each window is demeaned before the scalogram (``demean='auto'``).
    - Any length >= 30 s and any sampling rate >= 50 Hz is accepted (the original: exactly
      7500 samples, assumed to be at 250 Hz). Rates below 250 Hz are upsampled; spindles
      (11-16 Hz) are below the Nyquist rate, but the scalogram's top scales (up to 24 Hz)
      lose energy near the original Nyquist rate. The 'eeg' model's training data included
      50, 100 and 200 Hz DREAMS recordings upsampled to 250 Hz (by linear interpolation).

    With ``flat_s=None, demean=False`` a 250 Hz input gives the same intervals as
    ``openspindlenet`` run window by window with ``step_s=30`` (pinned by golden tests on
    single windows and on a 6 min multi-window recording), apart from the
    spindle-at-sample-0 case. The 'ieeg' model's defaults (``demean='auto'`` = off) give the
    same; the 'eeg' defaults differ only through demeaning (tested: same spindles, confidence
    differences < 0.01).

    .. rubric:: Memory

    Channels are processed one at a time, but each whole channel is held at its original rate
    in float64, in a few copies (the input cast, flat-run masking, gap filling and the
    resampler's working arrays): about 8 bytes x samples per copy, e.g. 22 GB per copy for
    24 h at 32 kHz. For long high-rate recordings, downsample beforehand (with an
    anti-aliasing filter, to e.g. 250-1000 Hz) or call :meth:`detect` on segments (overlap
    them by >= 30 s and keep each segment's spindles away from its edges).

    .. rubric:: Credits

    Models and the decoding are from OpenSpindleNet (https://github.com/CaptainTrojan/openspindlenet,
    MIT License, Copyright (c) 2025 CaptainTrojan; full text in NOTICE.md). If you use the
    detector, please cite: M. Seják, F. Mivalt, V. Sladký, V. Všianský, D. Z. Carvalho,
    E. K. St Louis, G. A. Worrell, V. Křemen, "OpenSpindleNet: An open-source deep learning
    network for reliable sleep spindle detection in scalp and intracranial EEG",
    Computers in Biology and Medicine 197 (2025) 110854.
    """

    def __init__(self, model: str = 'eeg', *, device: str = 'auto', threads: int | None = None,
                 batch_size: int = 32, confidence_threshold: float = 0.5,
                 nms_iou_threshold: float = 0.3, step_s: float = 20.0,
                 min_valid_fraction: float = 0.5, gap_margin_s: float = 0.5,
                 flat_s: float | None = 0.5, demean: bool | str = 'auto'):
        if model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 < nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in (0, 1]")
        if not 0.0 < step_s <= WINDOW_S:
            raise ValueError(f"step_s must be in (0, {WINDOW_S:g}] s, got {step_s}")
        if round(step_s * FS) < 1:
            raise ValueError("step_s is shorter than one sample at 250 Hz")
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        if gap_margin_s < 0:
            raise ValueError("gap_margin_s must be >= 0")
        if flat_s is not None and not flat_s > 0:
            raise ValueError("flat_s must be > 0 or None")
        self.model_name = model
        fname, sha = MODELS[model]
        self._model = OnnxModel(os.path.join(_MODEL_DIR, fname), device=device, threads=threads,
                                batch_size=batch_size, sha256=sha, name=f"OpenSpindleNet-{model}")
        self.threads = threads
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.step_s = float(step_s)
        self.min_valid_fraction = float(min_valid_fraction)
        self.gap_margin_s = float(gap_margin_s)
        self.flat_s = None if flat_s is None else float(flat_s)
        if isinstance(demean, str):
            if demean != 'auto':
                raise ValueError(f"demean must be 'auto', True or False, got {demean!r}")
            demean = DEMEAN_AUTO[model]
        elif not isinstance(demean, (bool, np.bool_)):
            raise ValueError(f"demean must be 'auto', True or False, got {demean!r}")
        #: resolved setting (bool): whether each window is demeaned before the scalogram
        self.demean = bool(demean)

    @property
    def device(self) -> str:
        """Device the network runs on ('cpu' or 'cuda')."""
        return self._model.device

    def __repr__(self) -> str:
        return (f"SpindleDetector(model={self.model_name!r}, device={self.device!r}, "
                f"step_s={self.step_s:g}, confidence_threshold={self.confidence_threshold:g})")

    # -- low level ----------------------------------------------------------------------
    def predict_windows(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        """Run the network on 30 s windows at 250 Hz, preprocessed as OpenSpindleNet does.

        Parameters
        ----------
        windows : array_like, shape (n_windows, 7500) or (7500,)
            Finite signal windows sampled at 250 Hz (any units; each window is z-scored, and
            demeaned before the scalogram if ``demean``).

        Returns
        -------
        dict
            ``'detection'``: ``(n_windows, 30, 3)`` sigmoid outputs (confidence, centre offset,
            duration / 2 s) per 1 s segment; ``'segmentation'``: ``(n_windows, 7500)``
            per-sample spindle probability.
        """
        w = np.asarray(windows, dtype=np.float64)
        if w.ndim == 1:
            w = w[None]
        if w.ndim != 2 or w.shape[1] != WINDOW:
            raise ValueError(f"windows must have shape (n, {WINDOW}), got {np.shape(windows)}")
        if not np.isfinite(w).all():
            raise ValueError("windows contain NaN/inf; use detect() for gapped recordings")
        out = {'detection': [], 'segmentation': []}
        workers = -1 if self.threads is None else self.threads
        bs = self._model.batch_size
        for i in range(0, len(w), bs):
            chunk = w[i:i + bs]
            raw = _normalize(chunk[:, None, :]).astype(np.float32)
            if self.demean:
                chunk = chunk - chunk.mean(axis=-1, keepdims=True)
            spec = _normalize(scalogram(chunk, workers=workers)).astype(np.float32)
            res = self._model.run({'raw_signal': raw, 'spectrogram': spec})
            out['detection'].append(sigmoid(res['detection']))
            out['segmentation'].append(sigmoid(res['segmentation'])[:, :WINDOW, 0])
        return {k: np.concatenate(v, axis=0) for k, v in out.items()}

    # -- high level ---------------------------------------------------------------------
    def detect(self, x, fs: float) -> SpindleDetections:
        """Detect spindles in a recording.

        Parameters
        ----------
        x : array_like, shape (n_samples,) or (n_channels, n_samples)
            Signal (any units, e.g. uV); NaN/inf mark missing data. At least 30 s long.
            Channels are processed independently (time on the last axis).
        fs : float
            Sampling rate (Hz), >= 50 (see the class notes for rates below 250 Hz).

        Returns
        -------
        SpindleDetections
        """
        x = np.asarray(x)
        if x.ndim not in (1, 2):
            raise ValueError(f"x must be 1-D (n_samples,) or 2-D (n_channels, n_samples), got shape {x.shape}")
        if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
            raise ValueError(f"x must be a real numeric array, got dtype {x.dtype}")
        fs = float(fs)
        if not np.isfinite(fs) or fs < FS_MIN:
            raise ValueError(f"fs must be a finite sampling rate >= {FS_MIN:g} Hz, got {fs}")
        xs = x[None] if x.ndim == 1 else x
        n_ch, n = xs.shape
        duration = n / fs
        if duration < WINDOW_S:
            raise ValueError(f"the recording is {duration:.2f} s long; at least {WINDOW_S:g} s are needed")
        up, down, fs_eff = _resample_ratio(fs)
        # Window grid at fs_eff, shared by all channels.
        m = max(WINDOW, int(np.ceil(n * up / down)))
        step = int(round(self.step_s * FS))
        starts = np.arange(0, m - WINDOW + 1, step)
        if starts[-1] != m - WINDOW:
            starts = np.append(starts, m - WINDOW)
        centres = starts + WINDOW / 2
        bounds = np.concatenate(([0.0], (centres[:-1] + centres[1:]) / 2, [float(m)]))

        # Channels one at a time: memory stays at a few copies of ONE channel.
        out_s, out_e, out_c, out_ch, not_eval, offsets = [], [], [], [], [], []
        for ch in range(n_ch):
            s_, e_, c_, ne, off = self._detect_channel(np.asarray(xs[ch], dtype=np.float64), fs, duration,
                                                       up, down, fs_eff, m, starts, bounds)
            out_s.append(s_); out_e.append(e_); out_c.append(c_); not_eval.append(ne)
            out_ch.append(np.full(s_.size, ch, dtype=np.int64))
            offsets.append(off)
        max_off = float(max(offsets))
        if not self.demean and max_off > _OFFSET_WARN_SD:
            warnings.warn(
                f"SpindleDetector({self.model_name!r}, demean=False): an evaluated window has a mean of "
                f"{max_off:.1f} SD of the window (channels {[c for c, o in enumerate(offsets) if o > _OFFSET_WARN_SD]}). "
                "Without demeaning the outputs depend on the DC offset (see the 'demean' documentation); "
                "high-pass filter the recording (e.g. 0.3-0.5 Hz) or pass demean=True explicitly.",
                RuntimeWarning, stacklevel=2)
        return SpindleDetections(
            start=np.concatenate(out_s), end=np.concatenate(out_e), confidence=np.concatenate(out_c),
            channel=np.concatenate(out_ch), not_evaluated=tuple(not_eval), n_channels=n_ch,
            duration_s=duration, model=self.model_name,
            params=dict(fs=fs, fs_model=fs_eff, resample=(up, down), step_s=self.step_s,
                        confidence_threshold=self.confidence_threshold,
                        nms_iou_threshold=self.nms_iou_threshold, min_valid_fraction=self.min_valid_fraction,
                        gap_margin_s=self.gap_margin_s, flat_s=self.flat_s, demean=self.demean,
                        n_windows=len(starts), max_abs_offset_sd=max_off))

    def _detect_channel(self, x, fs, duration, up, down, fs_eff, m, starts, bounds):
        """One channel: start, end, confidence (s), not-evaluated intervals, max |window mean| / SD."""
        empty = np.empty(0)
        # 1. gaps: NaN/inf and flat runs, in seconds (on the original signal)
        gaps = np.asarray(gap_intervals(x, fs), dtype=np.float64).reshape(-1, 2)
        if self.flat_s is not None:
            fr = _flat_runs(x, fs, self.flat_s)
            if len(fr):
                x = x.copy()
                for a, b in fr:
                    x[a:b] = np.nan
                gaps = _merge_intervals(np.vstack([gaps, fr / fs]))
        widened = gaps + np.array([-self.gap_margin_s, self.gap_margin_s])
        if not np.isfinite(x).any():
            return empty, empty, empty, np.array([[0.0, duration]]), 0.0
        # 2. fill the gaps so filters and the network see a plausible background
        if len(gaps):
            x = fill_gaps(x, fs)
        # 3. resample to ~250 Hz (polyphase FIR = anti-aliasing low-pass)
        z = x if up == down else resample_poly(x, up, down, padtype='line')
        if z.size < m:                          # rounding at exactly 30 s
            z = np.concatenate([z, np.repeat(z[-1:], m - z.size)])
        z = z[:m]
        # 4. windows to evaluate: enough valid data and not flat
        cbad = np.concatenate(([0], np.cumsum(_bad_mask(gaps, m, fs_eff))))
        valid_frac = 1.0 - (cbad[starts + WINDOW] - cbad[starts]) / WINDOW
        skipped, sel = [], []
        for k, s0 in enumerate(starts):
            if valid_frac[k] < self.min_valid_fraction or not np.ptp(z[s0:s0 + WINDOW]) > 0:
                skipped.append([bounds[k] / fs_eff, bounds[k + 1] / fs_eff])
            else:
                sel.append(k)
        # DC offset of the evaluated windows (relevant when demean is off)
        max_off = 0.0
        for k in sel:
            w = z[starts[k]:starts[k] + WINDOW]
            max_off = max(max_off, abs(w.mean()) / w.std())
        # 5. inference in batches, decoding, ownership
        rows = []
        bs = self._model.batch_size
        last = len(starts) - 1
        for i in range(0, len(sel), bs):
            ks = sel[i:i + bs]
            det = self.predict_windows(np.stack([z[starts[k]:starts[k] + WINDOW] for k in ks]))['detection']
            for k, d in zip(ks, det):
                iv = interval_nms(decode_window(d, WINDOW, self.confidence_threshold), self.nms_iou_threshold)
                if iv.size == 0:
                    continue
                iv[:, :2] += starts[k]
                c = (iv[:, 0] + iv[:, 1]) / 2
                own = (c >= bounds[k]) & ((c < bounds[k + 1]) | (k == last))
                rows.append(iv[own])
        r = np.vstack(rows) if rows else np.empty((0, 3))
        # 6. merge neighbouring windows' detections, seconds, drop spindles in/near gaps
        r = _nms_clusters(r, self.nms_iou_threshold)
        s, e, c = r[:, 0] / fs_eff, np.minimum(r[:, 1] / fs_eff, duration), r[:, 2]
        if len(gaps) and s.size:
            keep = ~mask_in_gaps(s, gaps, fs, units='seconds', gap_units='seconds',
                                 margin_s=self.gap_margin_s, end=e)
            s, e, c = s[keep], e[keep], c[keep]
        order = np.argsort(s, kind='stable')
        ne = _merge_intervals(np.vstack([widened, np.array(skipped).reshape(-1, 2)]))
        return s[order], e[order], c[order], np.clip(ne, 0.0, duration), max_off


def _bad_mask(gaps_s: np.ndarray, m: int, fs_eff: float) -> np.ndarray:
    """Samples ``i`` of the 250 Hz signal with ``i / fs_eff`` inside a gap ``[a, b)``."""
    d = np.zeros(m + 1, dtype=np.int64)
    if len(gaps_s):
        t = np.arange(m) / fs_eff
        a = np.searchsorted(t, gaps_s[:, 0], side='left')
        b = np.searchsorted(t, gaps_s[:, 1], side='left')
        np.add.at(d, a, 1)
        np.add.at(d, b, -1)
    return np.cumsum(d[:m]) > 0


def _nms_clusters(iv: np.ndarray, thr: float) -> np.ndarray:
    """NMS applied separately to each group of mutually overlapping intervals (fast for long recordings)."""
    if iv.size == 0:
        return iv.reshape(0, 3)
    iv = iv[np.argsort(iv[:, 0], kind='stable')]
    run_end = np.maximum.accumulate(iv[:, 1])
    new = np.concatenate(([True], iv[1:, 0] >= run_end[:-1]))
    groups = np.split(iv, np.flatnonzero(new)[1:])
    return np.vstack([g if len(g) == 1 else interval_nms(g, thr) for g in groups])
