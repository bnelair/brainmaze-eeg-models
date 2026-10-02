# Third-party notices

brainmaze-eeg-models is distributed under the BSD 3-Clause License (see `LICENSE`). It contains
the following third-party material, which remains under its own licence.

## OpenSpindleNet

- **What:** the trained sleep-spindle models
  `brainmaze_eeg_models/spindles/_models/spindle-detector-eeg.onnx` and
  `spindle-detector-ieeg.onnx` (unchanged copies), and code ported from OpenSpindleNet 0.1.1
  (`brainmaze_eeg_models/spindles/_decode.py`: detection decoding and interval NMS;
  `brainmaze_eeg_models/spindles/_detector.py` / `_cwt.py`: the input preprocessing, i.e. the
  z-scoring and the 'shan6-13' scalogram).
- **Source:** https://github.com/CaptainTrojan/openspindlenet (training code:
  https://github.com/CaptainTrojan/mayo_spindles).
- **Reference:** M. Seják, F. Mivalt, V. Sladký, V. Všianský, D. Z. Carvalho, E. K. St Louis,
  G. A. Worrell, V. Křemen, "OpenSpindleNet: An open-source deep learning network for reliable
  sleep spindle detection in scalp and intracranial EEG", *Computers in Biology and Medicine*
  197 (2025) 110854.
- **Licence:** MIT

```text
MIT License

Copyright (c) 2025 CaptainTrojan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The test suite (repository only, never in the release artifacts) also contains OpenSpindleNet's
two 30 s sample windows and the original package's outputs on them (`tests/data/spindle_golden.npz`),
under the same MIT licence.
