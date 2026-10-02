"""Bundled seizure models (ONNX), exported from brainmaze-torch 0.2.0 by tools/export_seizure_onnx.py."""
import os
from functools import lru_cache

from ...runtime import OnnxModel

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

#: name -> (file, SHA-256). 'modelA': published model (Sladky et al. 2022);
#: 'modelB': same architecture trained on an extended data set.
TRAINED_MODELS = {
    'modelA': ('modelA_paper.onnx', '088c815a10b54435432a0bef2629977d59e655a0c60b48a89e744aa52b6d48e3'),
    'modelB': ('modelB_full.onnx', '0fed4d9f9c11a97f9bd97307e8d97aed9216831bd3e6b15de81c0cd223979fcc'),
}


def load_trained_model(model_name, device='cpu', threads=None, cuda_device_id=0):
    """Load one of the bundled, pre-trained seizure-detection models.

    Parameters
    ----------
    model_name : {'modelA', 'modelB'}
        ``'modelA'``: the model from the published work (Sladky et al. 2022).
        ``'modelB'``: the same architecture trained on an extended data set.
    device : {'cpu', 'auto', 'cuda'}
        Default ``'cpu'``, as brainmaze-torch 0.2.0 (bit-reproducible across machines).
        ``'auto'`` (GPU if usable, else CPU) and ``'cuda'`` are opt-in; see
        :class:`brainmaze_eeg_models.runtime.OnnxModel` (``'cuda'`` never falls back to the
        CPU silently).
    threads : int, optional
        CPU threads per inference call (None: ONNX Runtime default).
    cuda_device_id : int
        GPU index for ``'cuda'`` / ``'auto'``.

    Returns
    -------
    OnnxModel
        Input ``'x'``: spectrograms ``(batch, 100, n_times)`` from
        :func:`~brainmaze_eeg_models.seizure.preprocess_input`; outputs ``'logits'`` and
        ``'probs'``, both ``(n_times, batch, 4)`` (class index 3 = seizure). Loaded models
        are cached (one session per name/device/threads/GPU).

    Raises
    ------
    KeyError
        If ``model_name`` is not one of the bundled models.
    """
    if model_name not in TRAINED_MODELS:
        raise KeyError(f"unknown trained model {model_name!r}; available {list(TRAINED_MODELS)}")
    return _load(model_name, device, threads, int(cuda_device_id))


@lru_cache(maxsize=16)
def _load(model_name, device, threads, cuda_device_id):
    fname, sha = TRAINED_MODELS[model_name]
    return OnnxModel(os.path.join(_MODEL_DIR, fname), device=device, threads=threads, batch_size=128,
                     cuda_device_id=cuda_device_id, output_batch_axes={'logits': 1, 'probs': 1},
                     sha256=sha, name=f"seizure-{model_name}")
