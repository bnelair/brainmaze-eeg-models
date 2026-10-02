Installation
============

Python >= 3.10. Dependencies: ``numpy``, ``scipy``, ``onnxruntime`` and
``brainmaze-utils >= 3.0.0``. No sample data and no PyTorch are installed.

CPU (default)
-------------

.. code-block:: bash

    pip install brainmaze-eeg-models

On Python 3.10 pip installs ``onnxruntime`` 1.23 (the last release with Python 3.10 wheels);
any ``onnxruntime >= 1.19`` works.

NVIDIA GPU
----------

``onnxruntime`` (CPU) and ``onnxruntime-gpu`` install **the same Python module**
(``onnxruntime``). With both installed, whichever was installed last wins, and uninstalling one
breaks the other. Install the package, then swap the runtime:

.. code-block:: bash

    pip install brainmaze-eeg-models
    pip uninstall -y onnxruntime
    pip install "onnxruntime-gpu[cuda,cudnn]"     # also installs matching CUDA + cuDNN wheels

``pip install "brainmaze-eeg-models[gpu]"`` adds ``onnxruntime-gpu`` but cannot remove the CPU
package; follow it with
``pip uninstall -y onnxruntime onnxruntime-gpu && pip install "onnxruntime-gpu[cuda,cudnn]"``.

Requirements: an NVIDIA driver and CUDA / cuDNN versions matching the ``onnxruntime-gpu``
build (current PyPI builds use CUDA 13 and cuDNN 9; the ``[cuda,cudnn]`` extras install them as
pip packages, which the package loads with ``onnxruntime.preload_dlls()``). See the
`ONNX Runtime CUDA requirements
<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_.

Since ``onnxruntime-gpu`` 1.27 the PyPI wheels are built for CUDA 13; builds for CUDA 12
(12.8 or newer) are published on a separate index (see the ONNX Runtime CUDA page above). Use
those if your driver or other packages (e.g. a PyTorch install sharing the CUDA libraries)
require CUDA 12.

On the GPU, TF32 matrix maths is **disabled** by default (ONNX Runtime enables it, which changes
results at the ~1e-3 level on Ampere and newer GPUs); with it off, GPU and CPU outputs agree to
float32 rounding. :class:`~brainmaze_eeg_models.runtime.OnnxModel` takes ``cuda_tf32=True`` to
allow it.

Check what is installed and usable:

.. code-block:: python

    from brainmaze_eeg_models.runtime import device_report
    print(device_report())

Every model takes ``device='auto' | 'cpu' | 'cuda'``. ``'cuda'`` never falls back to the CPU
silently (ONNX Runtime itself does when the CUDA libraries cannot be loaded): it raises an
error explaining what is missing. ``'auto'`` (default) uses CUDA when it initialises and the
CPU otherwise.
