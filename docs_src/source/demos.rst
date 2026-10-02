Demos
=====

The demos live in the repository (``demo/``), not in the installed package, and download their
data on first use. No recording ships with brainmaze-eeg-models.

Sleep spindles on one night of scalp EEG
----------------------------------------

`demo/spindle_detection/spindles_one_night.py
<https://github.com/bnelair/brainmaze-eeg-models/blob/main/demo/spindle_detection/spindles_one_night.py>`_
runs :class:`~brainmaze_eeg_models.spindles.SpindleDetector` on the brainmaze-eeg demo night
(about 6.8 h of Fz-Cz at 500 Hz with a hypnogram, downloaded once, 93 MB, from the
`brainmaze-eeg repository
<https://github.com/bnelair/brainmaze-eeg/tree/main/demo/eeg_wave_detection>`_), and prints the
number and density of spindles per sleep stage; ``--plot`` shows the most confident N2 spindle
with its 11-16 Hz band.

.. code-block:: bash

    pip install brainmaze-eeg-models matplotlib
    python demo/spindle_detection/spindles_one_night.py --plot

Output (CPU only: Xeon E5-1650 v2 from 2013, 6 cores / 12 threads, all threads used; the run
time varies with machine load, 253-271 s in our runs):

.. code-block:: text

    6.76 h in 271 s (90x real time) on cpu; 854 spindles; evaluated 99.4 % of the recording
     stage  minutes evaluated  spindles  per min
      Wake    151.6     151.4       211     1.39
        N1     50.5      50.5        48     0.95
        N2     84.5      84.5       431     5.10
        N3     27.5      27.5        89     3.24
       REM     71.5      71.5        50     0.70
    median duration 0.99 s, median confidence 0.81

Densities are computed over the *evaluated* minutes of each stage (gaps and skipped windows,
``res.not_evaluated``, are excluded from the denominator), as the demo script shows.
Spindles concentrate in N2 (about 5 per minute) and N3; the detections in wake and REM are
spindle-like activity (e.g. alpha bursts) that a hypnogram must exclude.
