"""
Seizure probability from intracranial EEG with a CNN + bidirectional LSTM applied to
spectrograms (Sladky et al. 2022, Brain Communications, doi:10.1093/braincomms/fcac115).

This is the brainmaze-torch 0.2.0 seizure pipeline with the model running on ONNX Runtime
(PyTorch is not needed). The public functions have the same names, arguments and outputs as
``brainmaze_torch.seizure_detection`` 0.2.0, so code moves over by changing the import:

.. code-block:: python

    # before: from brainmaze_eeg_models.seizure import predict_channel_seizure_probability
    from brainmaze_eeg_models.seizure import predict_channel_seizure_probability

Differences from brainmaze-torch 0.2.0:

- :func:`load_trained_model` returns an :class:`~brainmaze_eeg_models.runtime.OnnxModel`
  (not a ``torch.nn.Module``); passing a PyTorch model raises ``TypeError``.
- Device: ``use_cuda`` now defaults to None = ``'auto'`` (GPU if usable, else CPU);
  ``use_cuda=False`` forces the CPU and ``use_cuda=True`` the GPU ``cuda_number`` (an error if
  CUDA is unusable, never a silent CPU fallback). A keyword ``device='auto'|'cpu'|'cuda'`` is
  accepted as well. On the GPU, TF32 is disabled so results match the CPU.
- Numbers: the ONNX models (exported from the 0.2.0 weights, ``tools/export_seizure_onnx.py``
  in the repository) give the same probabilities as PyTorch within float32 rounding (parity
  tests: max abs difference about 1e-6, NaN masks identical); the brainmaze-torch golden
  tests pass unchanged.

Bundled models (:func:`load_trained_model`):

- ``'modelA'``: the model from the published work.
- ``'modelB'``: the same architecture trained on an extended data set.

The model was designed for 300 s inputs; its outputs near the ends of an input have
little LSTM context and are less reliable, which is why the high-level function drops
``discard_edges_s`` at each window edge.

Two ways to use it:

1. **High level** (recommended): :func:`predict_channel_seizure_probability` turns one
   long channel into a continuous probability trace on a 0.5 s grid. It handles the
   windowing, gaps, flat segments and the recording edges.
2. **Low level**: :func:`preprocess_input` + :func:`infer_seizure_probability` for
   batches of windows you cut yourself.

Conventions (both levels)
.........................

- ``fs`` must be a whole, even number of Hz, at least 200 Hz. A float such as ``500.0``
  (typical of .mat / MEF headers) is accepted. Odd or fractional rates raise
  ``ValueError``: resample first (anti-aliasing is the caller's job). These rules come from
  the spectrogram grid (whole samples per 1 s segment, an integer 0.5 s hop, 100 bins
  below Nyquist), not from the training data. At 200-256 Hz the anti-alias filter of the
  acquisition or resampling usually attenuates the upper bins (from ~0.4 fs = 80-100 Hz),
  which the model did not see in training: prefer higher rates (see
  :func:`preprocess_input`).
- Spectrogram: 1 s segments (``nperseg = fs``), 0.5 s hop, bins 0-99 Hz at 1 Hz.
  Column ``j`` of a window describes the 1 s of signal centred ``(j + 1) * 0.5`` s
  after the window's first sample.
- In the output of :func:`predict_channel_seizure_probability`, **NaN means "not
  evaluated", never "no seizure"**: ``t = 0`` and every 1 s segment containing NaN/inf or
  a flat signal are NaN; every other segment has a value.
  Combine channels or time with NaN-aware functions (``np.nanmax``) and never replace
  NaN by 0.
- The low-level :func:`preprocess_input` zero-fills NaN samples and does **not** mark
  them; masking is done by the high-level function.

Example (high level, one channel)
.................................

.. code-block:: python

    import numpy as np
    from brainmaze_eeg_models.seizure import predict_channel_seizure_probability

    fs = 500
    x = np.random.randn(fs * 600)                 # 10 min, one channel
    x[100 * fs:130 * fs] = np.nan                 # a 30 s gap
    t, p = predict_channel_seizure_probability(x, fs, model='modelA')
    # t = 0, 0.5, 1.0, ... s; p is NaN at t = 0 and over the gap

Example (low level, batch of 300 s windows)
...........................................

.. code-block:: python

    import numpy as np
    from brainmaze_eeg_models.seizure import (
        load_trained_model, preprocess_input, infer_seizure_probability)

    model = load_trained_model('modelA')
    fs = 500
    x = np.random.randn(3, fs * 300)              # 3 windows (rows) of 300 s
    t, f, sxx = preprocess_input(x, fs, return_axes=True)   # sxx: (3, 100, 599)
    y = infer_seizure_probability(sxx, model)     # (3, 599); y[:, j] belongs to t[j]
"""

from ._models import TRAINED_MODELS, load_trained_model
from ._seizure_detect import infer_seizure_probability, predict_channel_seizure_probability, preprocess_input

__all__ = [
    'TRAINED_MODELS',
    'load_trained_model',
    'preprocess_input',
    'infer_seizure_probability',
    'predict_channel_seizure_probability'
]


