Seizure probability (iEEG)
==========================

.. automodule:: brainmaze_eeg_models.seizure

Moving from brainmaze-torch
---------------------------

brainmaze-torch is retired; its last release will re-export these functions with a
``DeprecationWarning``. The public names are the same:

.. list-table::
   :header-rows: 1

   * - brainmaze-torch 0.2.0 (``brainmaze_torch.seizure_detection``)
     - brainmaze-eeg-models (``brainmaze_eeg_models.seizure``)
   * - ``predict_channel_seizure_probability(x, fs, model, use_cuda, cuda_number, ...)``
     - same signature and output; default device the CPU (as 0.2.0); ``use_cuda=True`` or the
       keyword ``device='cuda'`` / ``'auto'`` opt in to the GPU (both given: must agree)
   * - ``preprocess_input(x, fs, return_axes=False)``
     - identical code and output
   * - ``infer_seizure_probability(x, model, use_cuda, cuda_number)``
     - same; ``model`` may also be a name (default ``'modelA'``)
   * - ``load_trained_model('modelA' | 'modelB')`` -> ``torch.nn.Module``
     - -> :class:`~brainmaze_eeg_models.runtime.OnnxModel` (``device`` default ``'cpu'``, ``threads``,
       ``cuda_device_id``)

Validation
----------

- The brainmaze-torch golden tests (outputs of the published brainmaze-torch 0.1.1 code on a
  15 min iEEG seizure recording, incl. gaps and truncations) pass unchanged on the ONNX path
  (tolerance 1e-4; observed differences about 1e-6), as do its robustness tests.
- Parity with PyTorch (``tests/test_seizure_parity.py``, needs ``pip install -e ".[export]"``):
  end-to-end probabilities within 1e-4 (the golden tolerance) and identical NaN masks for both
  models on the fixture (with and without a gap) and on synthetic recordings at 200-1000 Hz
  with bursts, gaps and flat segments; preprocessing bit-identical. Observed differences are
  ~1e-6 typically and up to 2.1e-5 (modelB, a 400 Hz synthetic recording, at p = 0.49: the
  float32 BiLSTM over 599 steps accumulates rounding differently in the two runtimes, and the
  softmax is steepest near 0.5). These tests need PyTorch and are not run in CI; run them by
  hand when torch, onnx or onnxruntime change.
- GPU: not yet measured on a GPU (TF32 is off, so agreement to float32 rounding is expected).
- The ONNX files are produced by ``tools/export_seizure_onnx.py`` from the brainmaze-torch
  0.2.0 weights (reproducible byte for byte with the same PyTorch version) and their SHA-256
  is verified when they are loaded.

API
---

.. autofunction:: brainmaze_eeg_models.seizure.predict_channel_seizure_probability

.. autofunction:: brainmaze_eeg_models.seizure.preprocess_input

.. autofunction:: brainmaze_eeg_models.seizure.infer_seizure_probability

.. autofunction:: brainmaze_eeg_models.seizure.load_trained_model
