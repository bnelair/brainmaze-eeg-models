"""
Sleep-spindle detection on one night of scalp EEG
=================================================

Runs :class:`brainmaze_eeg_models.spindles.SpindleDetector` (OpenSpindleNet, scalp model) on
the brainmaze-eeg demo night: ~6.8 h of Fz-Cz at 500 Hz with a per-sample hypnogram.

The recording is NOT part of this package. It is downloaded once (93 MB) from the
brainmaze-eeg repository into this folder:
https://github.com/bnelair/brainmaze-eeg/raw/main/demo/eeg_wave_detection/patient_one_data.mat

Usage::

    pip install brainmaze-eeg-models matplotlib
    python spindles_one_night.py            # add --plot for a figure

It prints the number of spindles and their density (per minute) in every sleep stage. Spindles
belong to NREM sleep (mostly N2); detections in wake are mostly alpha bursts and other
spindle-like activity: restrict the analysis to NREM epochs from a hypnogram.
"""
import argparse
import os
import time
import urllib.request

import numpy as np
from scipy.io import loadmat

from brainmaze_eeg_models.spindles import SpindleDetector

URL = "https://github.com/bnelair/brainmaze-eeg/raw/main/demo/eeg_wave_detection/patient_one_data.mat"
DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "patient_one_data.mat")
# hypnogram codes of the demo file
STAGES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}


def load():
    if not os.path.exists(DATA_PATH):
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, DATA_PATH)
    d = loadmat(DATA_PATH)
    x = d["fzcz"].ravel().astype(np.float64)
    fs = float(d["fsamp"].ravel()[0])
    hyp = np.ceil(d["hypnogram"].ravel()).astype(int)       # per-sample stage code
    present = d["data_present"].ravel() > 0
    x[~present] = np.nan                                       # missing data -> NaN gaps
    return x, fs, hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    x, fs, hyp = load()
    det = SpindleDetector(model="eeg", device=args.device)
    print(det)
    t0 = time.perf_counter()
    res = det.detect(x, fs)
    el = time.perf_counter() - t0
    dur_h = len(x) / fs / 3600
    print(f"{dur_h:.2f} h in {el:.0f} s ({dur_h * 3600 / el:.0f}x real time) on {det.device}; "
          f"{len(res)} spindles; evaluated {100 * res.evaluated_fraction(0):.1f} % of the recording")

    centre = ((res.start + res.end) / 2 * fs).astype(int)
    stage_of_spindle = hyp[np.clip(centre, 0, len(hyp) - 1)]
    print(f"{'stage':>6} {'minutes':>8} {'spindles':>9} {'per min':>8}")
    for code, name in STAGES.items():
        minutes = np.sum(hyp == code) / fs / 60
        n = int(np.sum(stage_of_spindle == code))
        print(f"{name:>6} {minutes:8.1f} {n:9d} {n / minutes if minutes else float('nan'):8.2f}")
    print("median duration %.2f s, median confidence %.2f" % (np.median(res.end - res.start), np.median(res.confidence)))

    if args.plot:
        import matplotlib.pyplot as plt
        from scipy.signal import butter, sosfiltfilt
        i = int(np.argmax(res.confidence * (stage_of_spindle == 2)))     # most confident N2 spindle
        a, b = res.start[i] - 5, res.end[i] + 5
        sl = slice(int(a * fs), int(b * fs))
        t = np.arange(sl.start, sl.stop) / fs
        sigma = sosfiltfilt(butter(4, [11, 16], "bandpass", fs=fs, output="sos"), np.nan_to_num(x[sl]))
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        ax[0].plot(t, x[sl], lw=0.7)
        ax[1].plot(t, sigma, lw=0.7, color="C1")
        for s, e in res.channel_intervals(0)[:, :2]:
            if e > a and s < b:
                for axx in ax:
                    axx.axvspan(s, e, color="C2", alpha=0.2)
        ax[0].set_ylabel("Fz-Cz")
        ax[1].set_ylabel("11-16 Hz")
        ax[1].set_xlabel("time (s)")
        fig.suptitle(f"spindle at {res.start[i]:.1f} s, confidence {res.confidence[i]:.2f}")
        plt.show()


if __name__ == "__main__":
    main()
