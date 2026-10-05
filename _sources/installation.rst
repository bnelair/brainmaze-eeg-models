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

There is no ``[gpu]`` extra, on purpose. ``onnxruntime`` (CPU) and ``onnxruntime-gpu`` install
**the same Python module** (``onnxruntime``): with both installed the CPU build usually wins
without any error (``pip check`` passes), and uninstalling one breaks the other. Install the
package, then swap the runtime, and check:

.. code-block:: bash

    pip install brainmaze-eeg-models
    pip uninstall -y onnxruntime
    pip install "onnxruntime-gpu[cuda,cudnn]"     # also installs matching CUDA + cuDNN wheels
    python -c "import onnxruntime as ort; print(ort.get_available_providers())"
    # must list CUDAExecutionProvider

The ``[cuda,cudnn]`` extras install CUDA and cuDNN as pip packages, which the package loads with
``onnxruntime.preload_dlls()`` (once per process; its messages are shown by ``device_report()``
instead of being printed).

**Upgrading.** ``pip install -U brainmaze-eeg-models`` re-installs the CPU ``onnxruntime`` (a
dependency), which overwrites the GPU module again. In a GPU environment upgrade with

.. code-block:: bash

    pip install -U --no-deps brainmaze-eeg-models

or redo the swap afterwards. ``device='auto'`` warns when both packages are installed and the
CPU build is active, and ``device_report()`` prints a ``CONFLICT`` line.

**CUDA versions.** ``onnxruntime-gpu`` >= 1.27 on PyPI is built for **CUDA 13** + cuDNN 9 and
needs a driver that supports CUDA 13. For a CUDA 12 system (e.g. a cluster with an older driver,
or a PyTorch install sharing CUDA 12 libraries), use Microsoft's CUDA 12 feed. These builds need
**CUDA 12.8 or newer** (versions 1.27-1.29 for Python 3.11-3.14 as of 2026-10; use the newest one
listed on the feed):

.. code-block:: bash

    pip uninstall -y onnxruntime onnxruntime-gpu
    pip install --no-deps onnxruntime-gpu==1.29.0 \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
        nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12   # or a system CUDA >= 12.8 + cuDNN 9

``--no-deps`` with an exact version keeps pip from taking the same version from PyPI (the CUDA 13
build); the other dependencies are already installed with the CPU package.

**Python 3.10.** The newest ``onnxruntime-gpu`` with Python 3.10 wheels is 1.23.2, a CUDA 12
build, so on Python 3.10 the first recipe gives CUDA 12 (its ``[cuda,cudnn]`` extras install the
``-cu12`` wheels). See the `ONNX Runtime CUDA requirements
<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_.

On the GPU, TF32 matrix maths is **disabled** by default (ONNX Runtime enables it, which changes
results at the ~1e-3 level on Ampere and newer GPUs); with it off, GPU and CPU outputs are
expected to agree to float32 rounding. :class:`~brainmaze_eeg_models.runtime.OnnxModel` takes
``cuda_tf32=True`` to allow it.

.. note::

   **Limitation: the GPU path is not yet tested on a GPU.** The tests cover it with mocked ONNX
   Runtime sessions, and it was run with a real ``onnxruntime-gpu`` on a host without a GPU
   (``'cuda'`` raises, ``'auto'`` warns and uses the CPU). GPU-vs-CPU numbers for the bundled
   models have not been measured yet.

Check what is installed and usable:

.. code-block:: python

    from brainmaze_eeg_models.runtime import device_report
    print(device_report())

Every model takes ``device='auto' | 'cpu' | 'cuda'``. ``'cuda'`` never falls back to the CPU
silently (ONNX Runtime itself does when the CUDA libraries cannot be loaded): it raises an
error explaining what is missing. ``'auto'`` uses CUDA when it initialises and the CPU
otherwise, with a :class:`RuntimeWarning` when a GPU setup is installed but unusable.
The default is ``'auto'`` for the spindle detector and ``'cpu'`` for the seizure functions
(compatibility with brainmaze-torch 0.2.0).
