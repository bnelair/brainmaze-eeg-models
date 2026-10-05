Sleep spindles (OpenSpindleNet)
===============================

.. automodule:: brainmaze_eeg_models.spindles

Interpreting the output
-----------------------

- Times are seconds from the first sample of the input. Each spindle has a ``start``, an
  ``end`` (at most 2 s apart: the model's maximum duration; **no minimum duration is
  enforced**, and intervals clipped at a window or recording edge can be very short, so filter
  short events yourself if your definition needs one) and a ``confidence`` (the model's
  sigmoid output, >= ``confidence_threshold``).
- ``not_evaluated[ch]`` lists, per channel, the time where no spindle can be reported: gaps
  (NaN/inf and flat runs) widened by ``gap_margin_s``, and windows skipped for too little valid
  data. **No spindle there means "not evaluated", not "no spindle".** Densities (spindles per
  minute) must be computed over the evaluated time only
  (:meth:`~brainmaze_eeg_models.spindles.SpindleDetections.evaluated_fraction`).
- The models detect spindle-like activity in any state; in wake many detections are alpha
  bursts. Restrict analyses to NREM sleep with a hypnogram.

Sampling rate and preprocessing
-------------------------------

Any rate >= 50 Hz is accepted, but **the spindle counts depend on the band above ~25-60 Hz**,
not only on the spindle band (11-16 Hz): the network sees the whole z-scored signal. On the
same hour of a scalp night (demo night, 'eeg' model, defaults; the 250 Hz signal presented at
other rates with an anti-aliased resampler):

- 250 Hz: 231 spindles; 512-4096 Hz: the same 231; 256 Hz: 232 (1 extra);
- 200 Hz: 234 (recall 0.996, precision 0.983 vs 250 Hz);
- 128 Hz: 247 (+7 %); 100 Hz: 291 (+26 %); 50 Hz: 313 (+35 %);
- 250 Hz after a 60 Hz notch filter alone: 273 (+18 %).

Low-pass filtering the 250 Hz signal at 25 / 50 Hz gives 315 / 290 spindles, matching the
50 / 100 Hz runs at 99 %, so the resampler is consistent and the difference is the missing band
(in this recording, line noise at 58-62 Hz holds 2.9 % of the power, the spindle band 3.5 %).
:meth:`~brainmaze_eeg_models.spindles.SpindleDetector.detect` warns when ``fs`` < 200 Hz; it
cannot tell whether a higher-rate recording was notch or low-pass filtered before. **Use the
same sampling rate and the same preprocessing for every recording of a study** (or compare
only recordings processed alike).

These measurements show only that the counts differ. Which rate or preprocessing agrees best
with expert scoring was **not measured** (the counts are relative to the full-band 250 Hz run,
not to a ground truth); that the full band at >= 200 Hz is closest to the training data is a
plausible but untested hypothesis (whether the training recordings were notch filtered is not
documented).

Validation against the original package
---------------------------------------

- **Single windows:** with ``flat_s=None, demean=False`` a 30 s, 250 Hz input gives the same
  network outputs (within 1e-5) and the same intervals (within 0.01 samples) as
  ``openspindlenet`` 0.1.1 on its two sample windows (golden tests, also run live against the
  installed original). With the defaults, the eeg sample gives the same spindles (confidences
  within 0.01).
- **Multiple windows:** on 6 min (12 windows) of a scalp night, ``step_s=30`` with the other
  settings at their defaults gives the same intervals as ``openspindlenet`` run window by window
  (eeg: within 0.5 samples and 0.01 confidence, the difference being demeaning; ieeg:
  identical); with the default 20 s step, 94-96 % (eeg) / 89 % (ieeg) of the spindles agree.
  An independent 1 h check gave 204 vs 204 spindles (``step_s=30``, ``demean=False``).
- The one deliberate difference: ``openspindlenet`` drops a spindle whose interval starts
  exactly at the first sample of a window (the iEEG sample has one, 0-0.87 s, confidence 0.6;
  about 2 % of the detections on a long recording); it is kept here.
- The batched wavelet transform equals ``pywt.cwt`` (float64, the training-time wavelet
  sampling) to a relative error below 1e-14. Training computed it on float32 data, which moves
  the model outputs by <= 0.006 (see :func:`~brainmaze_eeg_models.spindles.scalogram`).
- Training resampled with ``np.interp`` (no anti-aliasing), the port with
  ``scipy.signal.resample_poly``: 854 vs 861 spindles on a 6.8 h night (93-94 % agreement), F1
  0.570 vs 0.572 on DREAMS.

Performance
-----------

Measured on a Xeon E5-1650 v2 (2013, 6 cores / 12 threads, all threads used; CPU only): the
wavelet transform takes about 6 ms per 30 s window (18 ms with one thread; ``pywt``: 1.2 s), the
network 0.12-0.2 s per window. With the default 20 s step a channel runs at 90-96x real
time (the 6.8 h demo night in 253-271 s; more on a loaded machine). ``step_s=30`` (no
overlap, as the original) is 1.5x faster but less stable at window edges (90.7 % of the
spindles agree with the 20 s step; 10 s vs 20 s: 94.6 %). A GPU (``device='cuda'``) should
remove most of the network cost (not yet measured).

Memory: channels are processed one at a time, but each channel is held whole at its original
rate in float64, in a few copies (e.g. 22 GB per copy for 24 h at 32 kHz). Downsample long
high-rate recordings beforehand or process them in overlapping segments (see
:class:`~brainmaze_eeg_models.spindles.SpindleDetector`, *Memory*).

API
---

.. autoclass:: brainmaze_eeg_models.spindles.SpindleDetector
   :members: detect, predict_windows, device

.. autoclass:: brainmaze_eeg_models.spindles.SpindleDetections
   :members: to_array, channel_intervals, evaluated_fraction, to_dataframe

.. autofunction:: brainmaze_eeg_models.spindles.scalogram
