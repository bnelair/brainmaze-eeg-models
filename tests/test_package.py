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


# --- shipped models: every .onnx is registered with a matching SHA-256 (#2 R7) -------------
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402

PKG = os.path.dirname(os.path.abspath(brainmaze_eeg_models.__file__))
ROOT = os.path.dirname(PKG)

#: model subpackage -> "module:attribute" of its registry {name: (file, sha256)}. A new model
#: package must add its registry here (the test fails for an unregistered _models/ folder).
REGISTRIES: dict[str, str] = {
    "spindles": "brainmaze_eeg_models.spindles._detector:MODELS",
}


def _registered():
    out = {}
    for pkg, ref in REGISTRIES.items():
        mod, attr = ref.split(":")
        for name, (fname, sha) in getattr(importlib.import_module(mod), attr).items():
            rel = f"brainmaze_eeg_models/{pkg}/_models/{fname}"
            assert rel not in out, f"{rel} registered twice"
            out[rel] = sha
    return out


def _required_assets(workflow):
    text = open(os.path.join(ROOT, ".github", "workflows", workflow)).read()
    m = re.search(r"required-assets:\s*>-?\s*\n((?:[ \t]+\S.*\n?)+)", text)
    return set(m.group(1).split()) if m else set()


def test_every_shipped_model_is_registered_with_its_sha256():
    shipped = {os.path.relpath(p, ROOT).replace(os.sep, "/")
               for p in glob.glob(os.path.join(PKG, "*", "_models", "*.onnx"))}
    model_dirs = {os.path.basename(os.path.dirname(os.path.dirname(p))) for p in shipped}
    assert model_dirs <= set(REGISTRIES), f"model folders without a registry: {model_dirs - set(REGISTRIES)}"
    reg = _registered()
    assert set(reg) == shipped, (f"unregistered: {shipped - set(reg)}; registered but missing: "
                                 f"{set(reg) - shipped}")
    for rel, sha in reg.items():
        with open(os.path.join(ROOT, rel), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == sha.lower(), f"SHA-256 mismatch for {rel}"


def test_required_assets_list_every_shipped_model():
    if not os.path.isdir(os.path.join(ROOT, ".github", "workflows")):
        pytest.skip("not a source checkout (the package is not installed in editable mode)")
    shipped = {os.path.relpath(p, ROOT).replace(os.sep, "/")
               for p in glob.glob(os.path.join(PKG, "*", "_models", "*.onnx"))}
    assert _required_assets("ci.yml") == shipped
    assert _required_assets("release.yml") == shipped
