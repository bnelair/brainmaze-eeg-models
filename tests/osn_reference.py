"""Test-harness access to the ORIGINAL openspindlenet package (PyPI 0.1.1), for golden tests.

Two problems of the original are worked around HERE ONLY (never in the package):

- ``openspindlenet/__init__.py`` does ``import pkg_resources``, which fails on fresh
  environments (setuptools >= 81 no longer provides it). A minimal stub with
  ``resource_filename`` is installed in ``sys.modules`` if the real module is missing.
- ``pywt.cwt`` sampled the wavelet with ``precision=10`` up to PyWavelets 1.8 (what the
  models were trained with) but defaults to 12 from 1.9 on; the original then silently
  computes a different scalogram. :func:`pywt_precision10` pins 10 while it is active.
"""
import contextlib
import importlib
import inspect
import os
import sys
import types


def _install_pkg_resources_stub():
    try:
        import pkg_resources  # noqa: F401
        return
    except Exception:  # noqa: BLE001 - missing or broken
        pass
    stub = types.ModuleType("pkg_resources")

    def resource_filename(package, resource):
        mod = importlib.import_module(package)
        return os.path.join(os.path.dirname(mod.__file__), resource)

    stub.resource_filename = resource_filename
    sys.modules["pkg_resources"] = stub


def load_openspindlenet():
    """Import the original package (raises ImportError if it is not installed)."""
    _install_pkg_resources_stub()
    import openspindlenet  # noqa: F401
    from openspindlenet import evaluator, inference
    return openspindlenet, inference, evaluator


@contextlib.contextmanager
def pywt_precision10():
    import pywt
    orig = pywt.cwt
    if "precision" in inspect.signature(orig).parameters:
        def cwt(*args, **kwargs):
            kwargs.setdefault("precision", 10)
            return orig(*args, **kwargs)
        pywt.cwt = cwt
    try:
        yield
    finally:
        pywt.cwt = orig
