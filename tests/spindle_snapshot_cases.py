"""Inputs of the full-defaults spindle snapshot (tests/data/spindle_snapshot.npz).

The snapshot freezes OUR OWN output with every default setting (step_s=20, demean='auto',
flat_s=0.5, gap handling, ownership, cross-window NMS) on deterministic inputs, so that any
change of the default pipeline shows up (verification round 2, #3 V1: mutants of the window
ownership changed the counts by 5-15 % and passed every agreement test). It is a regression
test, not a validation: the reference for correctness are the golden tests against the
original openspindlenet (tests/test_spindles.py).

Shared by the generator (tests/data/make_spindle_snapshot.py) and the tests, so both build
exactly the same inputs. Every input is derived from the bundled test data, except the
optional demo-night hour (``demo_night()``), which is used only when the public demo night
is available locally (it is never downloaded by the tests).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
KINDS = ("eeg", "ieeg")
SNAPSHOT = os.path.join(DATA, "spindle_snapshot.npz")
DEMO_URL = "https://github.com/bnelair/brainmaze-eeg/raw/main/demo/eeg_wave_detection/patient_one_data.mat"
# Approximate (non-rational) rate: exercises the approximate resampling ratio and its time axis.
FS_APPROX = 499.907


def _golden():
    return np.load(os.path.join(DATA, "spindle_golden.npz"))


def _long():
    return np.load(os.path.join(DATA, "spindle_golden_long.npz"))["x"].astype(np.float64)


def synthetic(n_rep=6):
    """~3 min continuous signal from the bundled eeg window (reversed copies keep it continuous)."""
    a = _golden()["eeg_x"].astype(np.float64)
    return np.concatenate([a if i % 2 == 0 else a[::-1] for i in range(n_rep)])


def gappy(x, fs):
    """Gaps covering every rule: NaN gaps (short, long, at the start), +inf, a flat run, and
    two 9 s gaps around the middle of the window starting at 100 s, so that window has a valid
    fraction of 0.4 < min_valid_fraction and the time it owns ([105, 125) s with step 20 s)
    is reported as not evaluated although it lies outside the widened gaps."""
    y = np.array(x, dtype=np.float64, copy=True)

    def s(t):
        return int(round(t * fs))
    y[:s(0.8)] = np.nan                       # gap at the recording start
    y[s(40.0):s(40.05)] = np.nan              # 50 ms (interpolated by fill_gaps, still a gap)
    y[s(100.0):s(109.0)] = np.nan             # } window 100-130 s: valid fraction 0.4
    y[s(121.0):s(130.0)] = np.nan             # }
    y[s(200.0):s(201.5)] = y[s(200.0)]        # 1.5 s flat run (disconnected channel)
    y[s(250.0):s(252.0)] = np.inf             # +inf gap
    y[s(300.0):s(303.0)] = np.nan
    return y


def approx_rate(x250, fs=FS_APPROX):
    """The 250 Hz signal presented at a non-rational rate (linear interpolation, deterministic)."""
    t = np.arange(int(np.floor(x250.size / 250.0 * fs))) / fs
    return np.interp(t, np.arange(x250.size) / 250.0, x250)


def cases():
    """name -> (x, fs). Always available (bundled data only)."""
    g = _golden()
    long = _long()
    syn = synthetic()
    multi = np.stack([syn, -syn[::-1], gappy(np.concatenate([syn, syn[::-1]]), 250.0)[:syn.size]])
    return {
        "sample_eeg": (g["eeg_x"].astype(np.float64), 250.0),      # bundled 30 s windows
        "sample_ieeg": (g["ieeg_x"].astype(np.float64), 250.0),
        "synthetic": (syn, 250.0),                                  # 180 s, 8 windows
        "synthetic_multichannel": (multi, 250.0),
        "long": (long, 250.0),                                      # 6 min of a real scalp night
        "long_gappy": (gappy(long, 250.0), 250.0),
        "long_500hz_gappy": (gappy(approx_rate(long, 500.0), 500.0), 500.0),
        "long_approx_rate": (approx_rate(long), FS_APPROX),
    }


def demo_night_path():
    """Local copy of the demo night, or None. Looked for in $BRAINMAZE_DEMO_NIGHT, then where the
    spindle demo caches it (demo/spindle_detection/patient_one_data.mat)."""
    for p in (os.environ.get("BRAINMAZE_DEMO_NIGHT"),
              os.path.join(os.path.dirname(HERE), "demo", "spindle_detection", "patient_one_data.mat")):
        if p and os.path.isfile(p):
            return p
    return None


def demo_cases(path):
    """Hour 1-2 of the demo night (Fz-Cz, 500 Hz) as is, and with 25 random gaps of 1-25 s."""
    import scipy.io as sio
    d = sio.loadmat(path)
    x = d["fzcz"][0].astype(np.float64)[500 * 3600:500 * 7200]
    y = x.copy()
    rng = np.random.default_rng(1)
    for a in rng.uniform(0, 3500, 25):
        y[int(a * 500):int((a + rng.uniform(1, 25)) * 500)] = np.nan
    return {"demo_hour": (x, 500.0), "demo_hour_gappy": (y, 500.0)}


def run(det, x, fs):
    """Detections as stored in the snapshot: per channel (n, 3) [start_s, end_s, conf] and
    (k, 2) not_evaluated, plus the effective model rate and the number of windows."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = det.detect(x, fs)
    out = {"fs_model": np.float64(r.params["fs_model"]), "n_windows": np.int64(r.params["n_windows"])}
    for ch in range(r.n_channels):
        out[f"ch{ch}_iv"] = r.channel_intervals(ch)
        out[f"ch{ch}_ne"] = np.asarray(r.not_evaluated[ch], dtype=np.float64).reshape(-1, 2)
    return out
