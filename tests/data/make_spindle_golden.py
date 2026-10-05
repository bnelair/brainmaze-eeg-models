"""Generate tests/data/spindle_golden.npz: the ORIGINAL openspindlenet 0.1.1 outputs on its own
two sample windows (eeg_sample.txt, ieeg_sample.txt; MIT, Copyright (c) 2025 CaptainTrojan).

    pip install openspindlenet==0.1.1 pywavelets
    python tests/data/make_spindle_golden.py      # from the repository root

The original runs with pywt's wavelet precision pinned to 10 (the training setting; see
tests/osn_reference.py) and its pkg_resources import worked around. Stored per model:
the input window (float64, as loaded by the original), the sigmoid detection head
(30, 3), the segmentation (7500,) and the original's NMS'd intervals (start, end,
confidence) in samples at 250 Hz, sorted by start.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from osn_reference import load_openspindlenet, pywt_precision10  # noqa: E402

osn, inference, _ = load_openspindlenet()
import importlib.metadata as md  # noqa: E402
import pywt  # noqa: E402

out = {"openspindlenet_version": md.version("openspindlenet"), "pywavelets_version": md.version("pywavelets"),
       "numpy_version": np.__version__}
for kind in ("eeg", "ieeg"):
    x = inference.SpindleInference(osn.get_model_path(kind)).load_data(osn.get_example_data_path(kind))
    with pywt_precision10():
        r = osn.detect(x, model_type=kind)
    iv = np.asarray(r["detection_intervals"], dtype=np.float64)
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    out[f"{kind}_x"] = np.asarray(x, dtype=np.float64)
    out[f"{kind}_detection"] = r["detection"].astype(np.float32)
    out[f"{kind}_segmentation"] = r["segmentation"][:, 0].astype(np.float32)
    out[f"{kind}_intervals"] = iv
    print(kind, np.round(iv, 2))
np.savez_compressed(os.path.join(HERE, "spindle_golden.npz"), **out)
print({k: v for k, v in out.items() if k.endswith("version")})
