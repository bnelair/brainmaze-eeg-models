"""Generate tests/data/spindle_snapshot.npz: OUR OWN full-defaults output (SpindleDetector('eeg')
and SpindleDetector('ieeg') with every default) on the inputs of tests/spindle_snapshot_cases.py.

    python tests/data/make_spindle_snapshot.py            # from the repository root

Regenerate ONLY after a deliberate change of the default pipeline (or of
brainmaze_utils.gaps.fill_gaps), and say in the PR why the numbers changed. The hour of the
demo night is included when a local copy is found ($BRAINMAZE_DEMO_NIGHT, or
demo/spindle_detection/patient_one_data.mat, where the spindle demo caches it); the tests skip
that part when the file is not available. Keys: ``<case>__<model>__<field>`` with the fields
of ``spindle_snapshot_cases.run``.
"""
import importlib.metadata as md
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import spindle_snapshot_cases as sc  # noqa: E402

from brainmaze_eeg_models.spindles import SpindleDetector  # noqa: E402

cases = sc.cases()
demo = sc.demo_night_path()
if demo:
    cases.update(sc.demo_cases(demo))
else:
    print("demo night not found: the snapshot has no demo-hour cases")
out = {"versions": np.array([f"{p} {md.version(p)}" for p in
                             ("brainmaze-eeg-models", "brainmaze-utils", "numpy", "scipy", "onnxruntime")])}
for kind in sc.KINDS:
    det = SpindleDetector(kind, device="cpu")
    for name, (x, fs) in cases.items():
        r = sc.run(det, x, fs)
        for k, v in r.items():
            out[f"{name}__{kind}__{k}"] = v
        n = [len(v) for k, v in r.items() if k.endswith("_iv")]
        ne = sum(float((v[:, 1] - v[:, 0]).sum()) for k, v in r.items() if k.endswith("_ne"))
        print(f"{kind:4s} {name:24s} spindles/channel {n}  not evaluated {ne:.2f} s  windows {int(r['n_windows'])}",
              flush=True)
np.savez_compressed(sc.SNAPSHOT, **out)
print("written", sc.SNAPSHOT, list(out["versions"]))
