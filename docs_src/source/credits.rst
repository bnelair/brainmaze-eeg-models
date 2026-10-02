Credits and licences
====================

brainmaze-eeg-models is licensed under the BSD 3-Clause License.

OpenSpindleNet
--------------

The spindle models (``spindle-detector-eeg.onnx``, ``spindle-detector-ieeg.onnx``, unchanged)
and the decoding / preprocessing code ported in :mod:`brainmaze_eeg_models.spindles` are from
`OpenSpindleNet <https://github.com/CaptainTrojan/openspindlenet>`_ (training code:
`mayo_spindles <https://github.com/CaptainTrojan/mayo_spindles>`_), MIT License,
Copyright (c) 2025 CaptainTrojan. The full licence text is in ``NOTICE.md`` (repository root,
and the ``licenses`` folder of the installed package's metadata). If you use the spindle
detector, please cite:

    M. Seják, F. Mivalt, V. Sladký, V. Všianský, D. Z. Carvalho, E. K. St Louis,
    G. A. Worrell, V. Křemen, "OpenSpindleNet: An open-source deep learning network for
    reliable sleep spindle detection in scalp and intracranial EEG",
    *Computers in Biology and Medicine* 197 (2025) 110854.
