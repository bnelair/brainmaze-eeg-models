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
     - same signature and output; ``use_cuda`` default None = ``'auto'``; keyword ``device``
   * - ``preprocess_input(x, fs, return_axes=False)``
     - identical code and output
   * - ``infer_seizure_probability(x, model, use_cuda, cuda_number)``
     - same; ``model`` may also be a name (default ``'modelA'``)
   * - ``load_trained_model('modelA' | 'modelB')`` -> ``torch.nn.Module``
     - -> :class:`~brainmaze_eeg_models.runtime.OnnxModel` (``device``, ``threads``, ``cuda_device_id``)

Validation
----------

- The brainmaze-torch golden tests (outputs of the published brainmaze-torch 0.1.1 code on a
  15 min iEEG seizure recording, incl. gaps and truncations) pass unchanged on the ONNX path
  (tolerance 1e-4; observed differences about 1e-6), as do its robustness tests.
- Parity with PyTorch (``tests/test_seizure_parity.py``, needs ``pip install -e ".[export]"``):
  end-to-end probabilities within 1e-5 (observed about 1e-6) and identical NaN masks for both
  models on the fixture (with and without a gap) and on synthetic recordings at 200-1000 Hz
  with bursts, gaps and flat segments; preprocessing bit-identical.
- The ONNX files are produced by ``tools/export_seizure_onnx.py`` from the brainmaze-torch
  0.2.0 weights (reproducible byte for byte with the same PyTorch version) and their SHA-256
  is verified when they are loaded.

API
---

.. autofunction:: brainmaze_eeg_models.seizure.predict_channel_seizure_probability

.. autofunction:: brainmaze_eeg_models.seizure.preprocess_input

.. autofunction:: brainmaze_eeg_models.seizure.infer_seizure_probability

.. autofunction:: brainmaze_eeg_models.seizure.load_trained_model
