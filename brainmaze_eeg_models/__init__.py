"""
brainmaze_eeg_models: ready-to-use trained models for EEG / iEEG, running on ONNX Runtime
(inference only, no PyTorch).
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("brainmaze-eeg-models")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ['__version__']
