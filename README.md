# brainmaze-eeg-models

Ready-to-use trained models for brain electrophysiology (EEG / iEEG), part of the
[BrainMaze](https://github.com/bnelair) family. Inference only: the models run on
[ONNX Runtime](https://onnxruntime.ai/), on the CPU by default or on an NVIDIA GPU, and
**PyTorch is not needed**.

| model | module | input | output |
|---|---|---|---|
| Seizure probability, CNN + BiLSTM (iEEG; successor of [brainmaze-torch](https://github.com/bnelair/brainmaze-torch)) | `brainmaze_eeg_models.seizure` | one channel, even integer fs >= 200 Hz, >= 300 s, NaN gaps allowed | probability every 0.5 s, NaN where not evaluated |
| OpenSpindleNet (scalp EEG / iEEG) | `brainmaze_eeg_models.spindles` | 1-D or multichannel, any fs >= 100 Hz, >= 30 s, NaN gaps allowed | spindle intervals (s) + confidence, and the time that could not be evaluated |

The package is built for long, multichannel recordings with gaps: missing data never turns into
silent zeros or a silent "no event". Wherever a model could not look at the data, the result says so.

Documentation: <https://bnelair.github.io/brainmaze-eeg-models/>

## Installation

Python >= 3.10. Dependencies: `numpy`, `scipy`, `onnxruntime`, `brainmaze-utils >= 3.0.0`.

### CPU (default)

```bash
pip install brainmaze-eeg-models
```

### NVIDIA GPU

`onnxruntime` (CPU) and `onnxruntime-gpu` install **the same Python module** (`onnxruntime`).
With both installed, whichever was installed last wins, and uninstalling one breaks the other.
So install the package, then swap the CPU runtime for the GPU one:

```bash
pip install brainmaze-eeg-models
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]"     # also installs matching CUDA + cuDNN wheels
```

`pip install "brainmaze-eeg-models[gpu]"` adds `onnxruntime-gpu` but cannot remove the CPU package;
after it, run `pip uninstall -y onnxruntime onnxruntime-gpu && pip install "onnxruntime-gpu[cuda,cudnn]"`.

Requirements: an NVIDIA driver, and CUDA / cuDNN versions matching the `onnxruntime-gpu` build
(current PyPI builds: CUDA 13 + cuDNN 9; the `[cuda,cudnn]` extras of `onnxruntime-gpu` install
them as pip packages; see the
[ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)).
CUDA 12 users: `onnxruntime-gpu` >= 1.27 on PyPI targets CUDA 13; CUDA 12 builds are on a separate
index (see the ONNX Runtime page above). TF32 is disabled on the GPU by default, so GPU and CPU
results agree to float32 rounding. Check what is installed and usable:

```python
from brainmaze_eeg_models.runtime import device_report
print(device_report())
```

`device='cuda'` never falls back to the CPU silently: it raises an error that says what is missing.
`device='auto'` (the default) uses CUDA when it initialises and the CPU otherwise.

## Quick start: seizure probability

```python
from brainmaze_eeg_models.seizure import predict_channel_seizure_probability

t, p = predict_channel_seizure_probability(x, fs, model='modelA')   # x: one channel, NaN = missing
# t[k] = k * 0.5 s; p[k] = seizure probability of the 1 s centred on t[k]; NaN = not evaluated
```

This is the brainmaze-torch 0.2.0 pipeline, unchanged, with the model on ONNX Runtime: the same
function names and outputs (migrating = changing the import from
`brainmaze_torch.seizure_detection`), the brainmaze-torch golden tests pass, and the
probabilities match PyTorch within ~1e-6. NaN means "not evaluated" (t = 0, gaps, flat
segments), never "no seizure": use `np.nanmax` / `np.isfinite`, never `fillna(0)`. `fs` must be a
whole, even number >= 200 Hz (resample first, with anti-aliasing). `use_cuda` now defaults to
`'auto'`; `use_cuda=False` forces the CPU.

## Quick start: sleep spindles

```python
from brainmaze_eeg_models.spindles import SpindleDetector

det = SpindleDetector(model='eeg')         # scalp EEG; 'ieeg' for intracranial EEG
res = det.detect(x, fs)                    # x: (n_samples,) or (n_channels, n_samples), NaN = missing

res.channel_intervals(0)                   # (n, 3): start_s, end_s, confidence of channel 0
res.not_evaluated[0]                       # (k, 2): [start_s, stop_s) where nothing could be detected
res.to_dataframe()                         # start, end, duration, confidence, channel
```

What happens inside (details in the documentation):

- Gaps (NaN/inf, and flat runs >= 0.5 s such as zero-filled dropouts or disconnected channels) are
  filled with noise matching the neighbouring spectrum (`brainmaze_utils.gaps`); afterwards every
  spindle overlapping a gap ± 0.5 s is dropped and the time is listed in `not_evaluated`.
- The signal is resampled to 250 Hz with an anti-aliasing filter (`scipy.signal.resample_poly`).
- 30 s windows every 20 s; each window keeps the spindles centred in its middle, and duplicates
  across window boundaries are merged.
- The model inputs are computed as in OpenSpindleNet (z-scored signal + 'shan6-13' wavelet
  scalogram) with a ~70x faster batched CWT, the wavelet sampling the models were trained with,
  and each window demeaned before the scalogram (a large DC offset otherwise changes detections).

Detections in wake are mostly spindle-like alpha bursts: analyse NREM epochs. Speed (CPU only, a
2013 6-core Xeon): about 110x real time per channel, i.e. a 6.8 h night in about 3.5 min.

Demo: [`demo/spindle_detection/spindles_one_night.py`](demo/spindle_detection/spindles_one_night.py)
(downloads a 6.8 h scalp EEG night from the brainmaze-eeg repository; no data ships with this package).

## Low-level runtime

`brainmaze_eeg_models.runtime.OnnxModel` wraps one ONNX Runtime session (device selection,
threads, batching, model checksum, input validation); every model in the package uses it.

## Credits and citation

**Seizure detection.** V. Sladky et al., "Distributed brain co-processor for tracking spikes,
seizures and behaviour during electrical brain stimulation", *Brain Communications* 4(3), 2022,
doi:10.1093/braincomms/fcac115.

**OpenSpindleNet.** The spindle models and the decoding are from
[OpenSpindleNet](https://github.com/CaptainTrojan/openspindlenet) (MIT License,
Copyright (c) 2025 CaptainTrojan; full notice in [NOTICE.md](NOTICE.md)). Please cite:

> M. Seják, F. Mivalt, V. Sladký, V. Všianský, D. Z. Carvalho, E. K. St Louis, G. A. Worrell,
> V. Křemen, "OpenSpindleNet: An open-source deep learning network for reliable sleep spindle
> detection in scalp and intracranial EEG", *Computers in Biology and Medicine* 197 (2025) 110854.

## Contributing and releasing

Work on a feature branch and open a pull request into `main` (protected; a review is required).
Never change `[project].version` in a pull request: releases follow the BrainMaze family process,
see [RELEASING.md](RELEASING.md). Tests: `pip install -e '.[test]' && pytest`.

Documentation (Sphinx, published to GitHub Pages by the Docs workflow):

```bash
pip install -r docs_src/requirements.txt -e .
sphinx-build -b html docs_src/source docs
```

## License

BSD 3-Clause, see [LICENSE](LICENSE). Third-party material (the OpenSpindleNet models and ported
code, MIT) is listed in [NOTICE.md](NOTICE.md).

## Funding

Parts of the BrainMaze toolbox were developed under projects including:

- NIH Brain Initiative UH2&3 NS095495 - *Neurophysiologically-Based Brain State Tracking & Modulation in Focal Epilepsy*,
- NIH U01-NS128612 - *An Ecosystem of Technology and Protocols for Adaptive Neuromodulation Research in Humans*,
- DARPA - HR0011-20-2-0028 *Manipulating and Optimizing Brain Rhythms for Enhancement of Sleep (Morpheus)*.
