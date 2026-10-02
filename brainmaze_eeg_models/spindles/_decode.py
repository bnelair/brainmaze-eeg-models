"""
Decoding of the OpenSpindleNet detection head and interval non-maximum suppression.

Ported from OpenSpindleNet (``openspindlenet/evaluator.py``: ``Evaluator.sigmoid_to_true_duration``,
``detections_to_intervals``, ``intervals_nms``), Copyright (c) 2025 CaptainTrojan (Michal Seják),
MIT License (full text in NOTICE.md at the repository root and in the wheel's licence files).

The detection head predicts, for each of its 30 one-second segments of a 30 s window,
``(confidence, centre offset within the segment, duration / (2 s))``, all after a sigmoid.

Differences from the original, on purpose (documented in the detector):

- The original NMS drops every interval whose start is exactly 0 (it uses ``start != 0`` to
  strip padding rows), i.e. a spindle that starts at the very first sample of a window (after
  clipping) was silently discarded. Here no interval is dropped for its position; intervals
  of zero length after clipping are dropped instead (they cannot be a detection).
- Intervals are computed in float64 (the original stores them in a float32 array).
- The IoU of two zero-length intervals is defined as 0 (the original divides by zero).
"""
from __future__ import annotations

import numpy as np

__all__ = ['sigmoid', 'decode_window', 'interval_nms']

SEGMENTS = 30          # detection grid of the model (one per second of a 30 s window)
FS = 250.0             # model sampling rate
MAX_DURATION_S = 2.0   # sigmoid(duration) * 2 s


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function (same formula as OpenSpindleNet; float32 in, float32 out)."""
    return 1 / (1 + np.exp(-x))


def decode_window(det: np.ndarray, seq_len: int = 7500, confidence_threshold: float = 0.5,
                  fs: float = FS) -> np.ndarray:
    """Turn one window's detection head output (after sigmoid) into intervals.

    Parameters
    ----------
    det : np.ndarray, shape (n_segments, 3)
        ``(confidence, centre offset, sigmoided duration)`` per segment, values in [0, 1].
    seq_len : int
        Window length in samples (7500).
    confidence_threshold : float
        Segments with ``confidence >= confidence_threshold`` are kept (as in the original).
    fs : float
        Sampling rate of the window (250 Hz); the duration unit is ``2 * fs`` samples.

    Returns
    -------
    np.ndarray, shape (n, 3), float64
        ``(start, end, confidence)`` in samples of the window, clipped to ``[0, seq_len]``,
        in segment order; zero-length intervals are dropped.
    """
    det = np.asarray(det, dtype=np.float64)
    seg = seq_len / det.shape[0]
    idx = np.flatnonzero(det[:, 0] >= confidence_threshold)
    conf, off, dur = det[idx, 0], det[idx, 1], det[idx, 2] * MAX_DURATION_S * fs
    centre = (idx + off) * seg
    start = np.clip(centre - dur / 2, 0, seq_len)
    end = np.clip(centre + dur / 2, 0, seq_len)
    keep = end > start
    return np.column_stack([start[keep], end[keep], conf[keep]])


def interval_nms(intervals: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """Greedy non-maximum suppression of ``(start, end, confidence, ...)`` rows.

    The most confident interval is kept and every remaining interval whose IoU with it is
    ``>= iou_threshold`` is removed; repeat. Extra columns are carried along.

    Returns
    -------
    np.ndarray
        The kept rows, most confident first (ties: lower index first).
    """
    iv = np.asarray(intervals, dtype=np.float64)
    if iv.size == 0:
        return iv.reshape(0, iv.shape[1] if iv.ndim == 2 else 3)
    order = np.argsort(-iv[:, 2], kind='stable')
    iv = iv[order]
    keep = []
    alive = np.ones(len(iv), dtype=bool)
    for i in range(len(iv)):
        if not alive[i]:
            continue
        keep.append(i)
        rest = np.flatnonzero(alive[i + 1:]) + i + 1
        if rest.size == 0:
            break
        s0, e0 = iv[i, 0], iv[i, 1]
        inter = np.maximum(0.0, np.minimum(e0, iv[rest, 1]) - np.maximum(s0, iv[rest, 0]))
        union = (e0 - s0) + (iv[rest, 1] - iv[rest, 0]) - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.where(union > 0, inter / union, 0.0)
        alive[rest[iou >= iou_threshold]] = False
    return iv[keep]
