# brainmaze-eeg-models

Ready-to-use trained models for brain electrophysiology (EEG / iEEG), part of the
[BrainMaze](https://github.com/bnelair) family. Inference only: the models run on
[ONNX Runtime](https://onnxruntime.ai/), on the CPU by default or on an NVIDIA GPU, and
**PyTorch is not needed**.

| model | module | input | output |
|---|---|---|---|
| Seizure probability, CNN + BiLSTM (iEEG; successor of [brainmaze-torch](https://github.com/bnelair/brainmaze-torch)) | `brainmaze_eeg_models.seizure` | one channel, even integer fs >= 200 Hz, >= 300 s, NaN gaps allowed | probability every 0.5 s, NaN where not evaluated |
| OpenSpindleNet (scalp EEG / iEEG) | `brainmaze_eeg_models.spindles` | 1-D or multichannel, fs >= 50 Hz (>= 200 Hz recommended, see below), >= 30 s, NaN gaps allowed | spindle intervals (s) + confidence, and the time that could not be evaluated |

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

There is no `[gpu]` extra, on purpose. `onnxruntime` (CPU) and `onnxruntime-gpu` install **the
same Python module** (`onnxruntime`): with both installed the CPU build usually wins without any
error (`pip check` passes), and uninstalling one breaks the other. So install the package, then
swap the CPU runtime for the GPU one, and check:

```bash
pip install brainmaze-eeg-models
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]"     # also installs matching CUDA + cuDNN wheels
python -c "import brainmaze_eeg_models as bm; print(bm.check_gpu())"           # runs a tiny model on the GPU
```

`check_gpu()` creates a CUDA session and runs a small inference on the GPU; it raises with the
reason and what to install if CUDA does not work. (`CUDAExecutionProvider` in
`onnxruntime.get_available_providers()` is not enough: a GPU build without usable CUDA libraries
lists it too and then runs everything on the CPU.) `SpindleDetector('eeg', device='cuda')` also
raises if its CUDA session cannot be created.

**Upgrading:** `pip install -U brainmaze-eeg-models` re-installs the CPU `onnxruntime` (a
dependency), which overwrites the GPU module again. In a GPU environment upgrade with

```bash
pip install -U --no-deps brainmaze-eeg-models
```

(or redo the swap afterwards). `device='auto'` warns when it finds both packages installed.

**CUDA versions.** `onnxruntime-gpu` >= 1.27 on PyPI is built for **CUDA 13** + cuDNN 9 and needs
a driver that supports CUDA 13. For a CUDA 12 system (e.g. a cluster with an older driver), the
simplest route is the last CUDA 12 builds on PyPI, 1.24-1.26 (Python >= 3.11), whose
`[cuda,cudnn]` extras install the `-cu12` CUDA and cuDNN wheels:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu[cuda,cudnn]<1.27"
```

For a newer CUDA 12 build use Microsoft's CUDA 12 feed; these builds need **CUDA 12.8 or newer**
(versions 1.27-1.29 for Python 3.11-3.14 as of 2026-10; use the newest one listed there):

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --no-deps onnxruntime-gpu==1.29.0 \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
    nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12   # or a system CUDA >= 12.8 + cuDNN 9
```

`--no-deps` because the feed also mirrors other packages in old versions (without it pip would
install e.g. numpy 2.1.2 and protobuf 5.28.3 from the feed); the dependencies are already
installed with the CPU package. (With `--index-url` pip does not look at PyPI at all.)
**Python 3.10:** the newest
`onnxruntime-gpu` for Python 3.10 is 1.23.2, a CUDA 12 build, so there the first recipe gives
CUDA 12 (its `[cuda,cudnn]` extras install the `-cu12` wheels). See the
[ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

TF32 is disabled on the GPU by default, so GPU and CPU results are expected to agree to float32
rounding. **The GPU path has not been run on a GPU in this package's tests yet** (mocked sessions,
plus a real `onnxruntime-gpu` on a GPU-less host, where it raises / warns as documented). Check
what is installed (`device_report()`) and whether CUDA really works (`check_gpu()`):

```python
import brainmaze_eeg_models as bm
print(bm.device_report())
print(bm.check_gpu())      # raises RuntimeError with the reason if CUDA is unusable
```

`device='cuda'` never falls back to the CPU silently: it raises an error that says what is missing.
`device='auto'` (the default of `SpindleDetector` and `OnnxModel`; the seizure functions default to
the CPU, as brainmaze-torch 0.2.0 did) uses CUDA when it initialises and the CPU otherwise, with a warning
when a GPU setup is installed but unusable.

## Quick start: seizure probability

```python
from brainmaze_eeg_models.seizure import predict_channel_seizure_probability

t, p = predict_channel_seizure_probability(x, fs, model='modelA')   # x: one channel, NaN = missing
# t[k] = k * 0.5 s; p[k] = seizure probability of the 1 s centred on t[k]; NaN = not evaluated
```

This is the brainmaze-torch 0.2.0 pipeline, unchanged, with the model on ONNX Runtime: the same
function names and outputs (migrating = changing the import from
`brainmaze_torch.seizure_detection`), the brainmaze-torch golden tests pass, and the
probabilities match PyTorch to float32 rounding (typically ~1e-6, at most ~2e-5). NaN means "not evaluated" (t = 0, gaps, flat
segments), never "no seizure": use `np.nanmax` / `np.isfinite`, never `fillna(0)`. `fs` must be a
whole, even number >= 200 Hz (resample first, with anti-aliasing). As in brainmaze-torch 0.2.0 it
runs on the CPU by default; `use_cuda=True` or `device='cuda'` / `'auto'` opt in to the GPU.
One 0.2.0 pattern now raises `ValueError`: `m = load_trained_model(name)` followed by
`use_cuda=True` (an ONNX session cannot move to the GPU). Load the model on the GPU instead,
`load_trained_model(name, device='cuda', cuda_device_id=k)`, or pass the model name with
`use_cuda=True` (see the [migration notes](https://bnelair.github.io/brainmaze-eeg-models/seizure.html)).

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
- The signal is resampled to 250 Hz with an anti-aliasing filter (`scipy.signal.resample_poly`,
  exact ratio where one exists).
- 30 s windows every 20 s; each window keeps the spindles centred in its middle, and duplicates
  across window boundaries are merged.
- The model inputs are computed as in OpenSpindleNet (z-scored signal + 'shan6-13' wavelet
  scalogram) with a batched FFT wavelet transform and the wavelet sampling the models were
  trained with. With the scalp model each window is demeaned before the scalogram (a large DC
  offset otherwise changes detections); the iEEG model keeps its training behaviour (no
  demeaning; see `demean` in the documentation for the DC-offset limitation).

Detections in wake are mostly spindle-like alpha bursts: analyse NREM epochs.

**Sampling rate and preprocessing change the counts.** Any fs >= 50 Hz is accepted, but the
model's output depends on the band above ~25-60 Hz, not only on the spindle band. On the same
hour of a scalp night: 231 spindles at 250 Hz and at 512-4096 Hz (identical detections), 232 at
256 Hz (1 extra), 234 at 200 Hz,
247 at 128 Hz (+7 %), 291 at 100 Hz (+26 %), 313 at 50 Hz (+35 %), and 273 (+18 %) at 250 Hz
after a 60 Hz notch filter alone. The resampler is not the cause (low-pass filtering the 250 Hz
signal gives the same counts as the low rates). `detect` warns below 200 Hz. **Use the same
sampling rate and preprocessing (notch, low-pass, referencing) for every recording of a
study.** These numbers show only that the counts differ; which setting agrees best with expert
scoring was not measured.

Speed (CPU only; Xeon E5-1650 v2 from 2013, 6 cores / 12 threads, all threads used): the wavelet
transform takes about 6 ms per 30 s window (PyWavelets: 1.2 s, ~200x slower; about 18 ms with
one thread), the network about 0.12-0.2 s per window. The 6.8 h demo night takes 253-271 s
(90-96x real time per channel) on that machine; expect more on a loaded machine.

Demo: [`demo/spindle_detection/spindles_one_night.py`](https://github.com/bnelair/brainmaze-eeg-models/blob/main/demo/spindle_detection/spindles_one_night.py)
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
Copyright (c) 2025 CaptainTrojan; full notice in [NOTICE.md](https://github.com/bnelair/brainmaze-eeg-models/blob/main/NOTICE.md)). Please cite:

> M. Seják, F. Mivalt, V. Sladký, V. Všianský, D. Z. Carvalho, E. K. St Louis, G. A. Worrell,
> V. Křemen, "OpenSpindleNet: An open-source deep learning network for reliable sleep spindle
> detection in scalp and intracranial EEG", *Computers in Biology and Medicine* 197 (2025) 110854.

## Contributing and releasing

Work on a feature branch and open a pull request into `main` (protected; a review is required).
Never change `[project].version` in a pull request: releases follow the BrainMaze family process,
see [RELEASING.md](https://github.com/bnelair/brainmaze-eeg-models/blob/main/RELEASING.md). Tests: `pip install -e '.[test]' && pytest`.

Documentation (Sphinx, published to GitHub Pages by the Docs workflow):

```bash
pip install -r docs_src/requirements.txt -e .
sphinx-build -b html docs_src/source docs
```

## License

BSD 3-Clause, see [LICENSE](https://github.com/bnelair/brainmaze-eeg-models/blob/main/LICENSE). Third-party material (the OpenSpindleNet models and ported
code, MIT) is listed in [NOTICE.md](https://github.com/bnelair/brainmaze-eeg-models/blob/main/NOTICE.md).

## Funding

Parts of the BrainMaze toolbox were developed under projects including:

- NIH Brain Initiative UH2&3 NS095495 - *Neurophysiologically-Based Brain State Tracking & Modulation in Focal Epilepsy*,
- NIH U01-NS128612 - *An Ecosystem of Technology and Protocols for Adaptive Neuromodulation Research in Humans*,
- DARPA - HR0011-20-2-0028 *Manipulating and Optimizing Brain Rhythms for Enhancement of Sleep (Morpheus)*.
