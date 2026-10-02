"""Generate tests/data/spindle_golden_long.npz: the ORIGINAL openspindlenet 0.1.1 run window by
window (12 consecutive 30 s windows = 6 min) on a scalp EEG night, for the multi-window golden
tests (windowing offsets, ownership, cross-window NMS, the effective rate).

    pip install openspindlenet==0.1.1 pywavelets
    python tests/data/make_spindle_golden_long.py [patient_one_data.mat]   # from the repository root

Data: the public brainmaze-eeg demo night (Fz-Cz, 500 Hz, with a hypnogram;
https://github.com/bnelair/brainmaze-eeg/raw/main/demo/eeg_wave_detection/patient_one_data.mat,
downloaded if no path is given). The 6 min with the most N2 are taken, resampled to 250 Hz with
scipy.signal.resample_poly(x, 1, 2) and stored as float32 (the tests cast to float64, as the
reference run here did). Both models are run on the same signal (the 'ieeg' model on scalp
data only exercises the pipeline). The original runs with pywt's wavelet precision pinned to 10
(tests/osn_reference.py). Stored per model: intervals (start, end, confidence) in samples at
250 Hz from the start of the excerpt, sorted by start, with each window's offset added.
"""
import os
import sys
import tempfile
import urllib.request

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from osn_reference import load_openspindlenet, pywt_precision10  # noqa: E402

URL = "https://github.com/bnelair/brainmaze-eeg/raw/main/demo/eeg_wave_detection/patient_one_data.mat"
N_WIN, WIN = 12, 7500

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "patient_one_data.mat")
if not os.path.exists(path):
    urllib.request.urlretrieve(URL, path)
d = sio.loadmat(path)
x500, hyp = d["fzcz"][0].astype(np.float64), d["hypnogram"][0]
n500 = N_WIN * WIN * 2
step = 500 * 60
scores = [np.mean(np.ceil(hyp[i:i + n500]) == 2) for i in range(0, len(hyp) - n500, step)]
st = int(np.argmax(scores)) * step
x = resample_poly(x500[st:st + n500], 1, 2).astype(np.float32)
assert x.size == N_WIN * WIN and np.isfinite(x).all()

osn, inference, _ = load_openspindlenet()
import importlib.metadata as md  # noqa: E402

out = {"openspindlenet_version": md.version("openspindlenet"), "pywavelets_version": md.version("pywavelets"),
       "numpy_version": np.__version__, "source": URL, "start_sample_500hz": st, "x": x}
xf = x.astype(np.float64)
for kind in ("eeg", "ieeg"):
    ivs = []
    with pywt_precision10():
        for k in range(N_WIN):
            r = osn.detect(xf[k * WIN:(k + 1) * WIN], model_type=kind)
            iv = np.asarray(r["detection_intervals"], dtype=np.float64).reshape(-1, 3)
            iv[:, :2] += k * WIN
            ivs.append(iv)
    iv = np.vstack(ivs)
    out[f"{kind}_intervals"] = iv[np.argsort(iv[:, 0], kind="stable")]
    print(kind, len(iv), "spindles")
np.savez_compressed(os.path.join(HERE, "spindle_golden_long.npz"), **out)
print("N2 fraction", max(scores), {k: v for k, v in out.items() if k.endswith("version")})
