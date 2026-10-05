BrainMaze - EEG Models
======================

Ready-to-use trained models for brain electrophysiology (EEG / iEEG), part of the
`BrainMaze <https://github.com/bnelair>`_ family. Inference only: the models run on
`ONNX Runtime <https://onnxruntime.ai/>`_, on the CPU by default or on an NVIDIA GPU;
PyTorch is not needed.

Version |release|. Source: https://github.com/bnelair/brainmaze-eeg-models

.. list-table::
   :header-rows: 1

   * - model
     - module
     - output
   * - Seizure probability, CNN + BiLSTM (iEEG; successor of brainmaze-torch)
     - :mod:`brainmaze_eeg_models.seizure`
     - probability every 0.5 s; NaN where not evaluated
   * - OpenSpindleNet (scalp EEG / iEEG)
     - :mod:`brainmaze_eeg_models.spindles`
     - spindle intervals (s) + confidence, and the time that could not be evaluated

The package is built for long, multichannel recordings with gaps (NaN). Missing data never
turns into silent zeros or a silent "no event": every result states where the model could not
look at the data.

.. toctree::
   :maxdepth: 2

   installation
   seizure
   spindles
   demos
   runtime
   credits
