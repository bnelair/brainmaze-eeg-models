import re

import brainmaze_eeg_models


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", brainmaze_eeg_models.__version__)


def test_runtime_imports_are_light():
    # The package must not depend on PyTorch at runtime.
    import importlib.metadata as md
    req = md.requires("brainmaze-eeg-models") or []
    base = [r for r in req if "extra ==" not in r]
    assert not any(r.lower().startswith("torch") for r in base), base
    assert any(r.lower().startswith("onnxruntime") for r in base), base
