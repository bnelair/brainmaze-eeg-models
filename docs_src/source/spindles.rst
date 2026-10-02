Sleep spindles (OpenSpindleNet)
===============================

.. automodule:: brainmaze_eeg_models.spindles

Interpreting the output
-----------------------

- Times are seconds from the first sample of the input. Each spindle has a ``start``, an
  ``end`` (0.1-2 s apart; the model's maximum duration is 2 s) and a ``confidence`` (the
  model's sigmoid output, >= ``confidence_threshold``).
- ``not_evaluated[ch]`` lists, per channel, the time where no spindle can be reported: gaps
  (NaN/inf and flat runs) widened by ``gap_margin_s``, and windows skipped for too little valid
  data. **No spindle there means "not evaluated", not "no spindle".** Densities (spindles per
  minute) must be computed over the evaluated time only
  (:meth:`~brainmaze_eeg_models.spindles.SpindleDetections.evaluated_fraction`).
- The models detect spindle-like activity in any state; in wake many detections are alpha
  bursts. Restrict analyses to NREM sleep with a hypnogram.

Validation against the original package
---------------------------------------

With ``flat_s=None, demean=False`` a 30 s, 250 Hz input gives the same network outputs
(within 1e-5) and the same intervals (within 0.01 samples) as ``openspindlenet`` 0.1.1 on its
two sample windows (golden tests, also run live against the installed original). The one
deliberate difference there: ``openspindlenet`` drops a spindle whose interval starts exactly at
the first sample of the window (the iEEG sample has one, 0-0.87 s, confidence 0.6); it is kept
here. The batched wavelet transform equals ``pywt.cwt`` with the training-time wavelet sampling
to a relative error below 1e-14.

Performance
-----------

The network costs about 0.12-0.2 s per 30 s window on a 2013 6-core Xeon (CPU), the wavelet
transform about 6 ms (``pywt``: 1.2 s). With the default 20 s step a channel runs at about 110x
real time on that CPU (a 6.8 h night in ~3.5 min); ``step_s=30`` (no overlap, as the original)
is 1.5x faster but less stable at window edges (90.7 % of the spindles agree with the 20 s
step; 10 s vs 20 s: 94.6 %). A GPU (``device='cuda'``) removes most of the network cost.

API
---

.. autoclass:: brainmaze_eeg_models.spindles.SpindleDetector
   :members: detect, predict_windows, device

.. autoclass:: brainmaze_eeg_models.spindles.SpindleDetections
   :members: to_array, channel_intervals, evaluated_fraction, to_dataframe

.. autofunction:: brainmaze_eeg_models.spindles.scalogram
