"""
Sleep-spindle detection with OpenSpindleNet (scalp EEG and intracranial EEG models).

.. code-block:: python

    from brainmaze_eeg_models.spindles import SpindleDetector

    det = SpindleDetector(model='eeg')            # 'ieeg' for intracranial EEG
    res = det.detect(x, fs)                       # x: (n_samples,) or (n_channels, n_samples)
    res.channel_intervals(0)                      # [start_s, end_s, confidence] per spindle
    res.not_evaluated[0]                          # [start_s, stop_s) with no detection possible

See :class:`SpindleDetector` for the pipeline (gaps, resampling to 250 Hz, sliding windows)
and the differences from the ``openspindlenet`` package. The models and the decoding are
ported from OpenSpindleNet (https://github.com/CaptainTrojan/openspindlenet, MIT License,
Copyright (c) 2025 CaptainTrojan; full notice in NOTICE.md). Please cite
Seják et al., Computers in Biology and Medicine 197 (2025) 110854.
"""
from ._cwt import scalogram
from ._detector import DEMEAN_AUTO, FS, MODELS, WINDOW, SpindleDetections, SpindleDetector

__all__ = ['SpindleDetector', 'SpindleDetections', 'scalogram', 'MODELS', 'FS', 'WINDOW', 'DEMEAN_AUTO']
