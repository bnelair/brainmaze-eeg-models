import os
import warnings

import numpy as np
import pytest

from brainmaze_eeg_models.spindles import FS, WINDOW, SpindleDetector, scalogram
from brainmaze_eeg_models.spindles._cwt import SCALES, shan_wavefun
from brainmaze_eeg_models.spindles._decode import decode_window, interval_nms
from brainmaze_eeg_models.spindles._detector import _flat_runs, _resample_ratio

from . import osn_reference

GOLDEN = np.load(os.path.join(os.path.dirname(__file__), "data", "spindle_golden.npz"))
KINDS = ("eeg", "ieeg")


@pytest.fixture(scope="module")
def detectors():
    return {k: SpindleDetector(k, device="cpu") for k in KINDS}


def _sorted(iv):
    iv = np.asarray(iv, dtype=np.float64).reshape(-1, 3)
    return iv[np.argsort(iv[:, 0], kind="stable")]


def _assert_same_as_original(ours_samples, ref):
    """Same intervals as openspindlenet, except the ones the original drops because they start
    exactly at sample 0 (its `start != 0` padding filter; documented difference)."""
    extra = ours_samples[ours_samples[:, 0] == 0]
    ours = ours_samples[ours_samples[:, 0] != 0]
    assert ours.shape == ref.shape, (ours, ref)
    np.testing.assert_allclose(ours[:, :2], ref[:, :2], atol=1e-2)      # samples (float32 in the original)
    np.testing.assert_allclose(ours[:, 2], ref[:, 2], atol=1e-6)
    return extra


# --- CWT ------------------------------------------------------------------------------

def test_wavefun_matches_pywt():
    pywt = pytest.importorskip("pywt")
    psi, x = pywt.ContinuousWavelet("shan6-13").wavefun(10)
    psi2, x2 = shan_wavefun(10)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_allclose(psi2, psi, rtol=0, atol=1e-14)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_scalogram_matches_pywt_precision10_float64_path(dtype):
    # float32 input is compared with pywt's FLOAT64 path (x cast before pywt.cwt): the port always
    # computes in float64. pywt's own float32 path (used in training) differs slightly; see _cwt.py.
    pywt = pytest.importorskip("pywt")
    rng = np.random.default_rng(3)
    xs = np.stack([GOLDEN["eeg_x"], GOLDEN["ieeg_x"], 40 * rng.standard_normal(WINDOW)]).astype(dtype)
    ours = scalogram(xs)
    assert ours.shape == (3, 15, WINDOW) and ours.dtype == np.float64
    with osn_reference.pywt_precision10():
        for x, o in zip(xs, ours):
            ref = np.abs(pywt.cwt(x.astype(np.float64), SCALES, "shan6-13", sampling_period=1 / 250)[0])
            assert np.abs(o - ref).max() / np.abs(ref).max() < 1e-6
            assert np.abs(o - ref).max() / np.abs(ref).max() < 1e-10   # in practice: rounding only


def test_scalogram_shapes_and_validation():
    x = np.random.default_rng(0).standard_normal((2, 3, 1000))
    s = scalogram(x)
    assert s.shape == (2, 3, 15, 1000)
    np.testing.assert_allclose(s[1, 2], scalogram(x[1, 2]))      # batching is transparent
    with pytest.raises(ValueError):
        scalogram(np.zeros((2, 0)))


# --- golden: the original openspindlenet ------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_network_outputs_match_golden(detectors, kind):
    out = SpindleDetector(kind, device="cpu", demean=False).predict_windows(GOLDEN[f"{kind}_x"])
    np.testing.assert_allclose(out["detection"][0], GOLDEN[f"{kind}_detection"], atol=1e-5)
    np.testing.assert_allclose(out["segmentation"][0], GOLDEN[f"{kind}_segmentation"], atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_detect_matches_golden(kind):
    # flat_s=None: the iEEG sample contains 4.6 s of exact zeros (a zero-filled dropout) that the
    # original treats as data; here we compare the pure pipeline.
    det = SpindleDetector(kind, device="cpu", flat_s=None, demean=False)
    res = det.detect(GOLDEN[f"{kind}_x"], 250.0)
    assert res.params["n_windows"] == 1 and res.not_evaluated[0].size == 0
    ours = res.channel_intervals(0)
    ours[:, :2] *= FS
    extra = _assert_same_as_original(ours, GOLDEN[f"{kind}_intervals"])
    if kind == "eeg":
        assert extra.size == 0
    else:   # the iEEG sample has one spindle cut by the window start: kept here, dropped by the original
        assert extra.shape == (1, 3) and extra[0, 1] == pytest.approx(217.8, abs=0.1)


@pytest.mark.parametrize("kind", KINDS)
def test_detect_matches_live_original(kind):
    try:
        osn, inference, _ = osn_reference.load_openspindlenet()
    except ImportError:
        pytest.skip("openspindlenet (the original package) is not installed")
    pytest.importorskip("pywt")
    x = GOLDEN[f"{kind}_x"]
    with osn_reference.pywt_precision10():
        ref = _sorted(osn.detect(x, model_type=kind)["detection_intervals"])
    ours = SpindleDetector(kind, device="cpu", flat_s=None, demean=False).detect(x, 250).channel_intervals(0)
    ours[:, :2] *= FS
    _assert_same_as_original(ours, ref)


@pytest.mark.parametrize("kind", KINDS)
def test_default_settings_find_the_same_spindles_on_the_samples(detectors, kind):
    # demean (default) + flat runs as gaps: on the eeg sample (offset 0.007 SD) nothing changes;
    # the ieeg sample (offset -1.9 SD, 4.6 s zero-filled dropout) keeps every spindle outside the dropout
    res = detectors[kind].detect(GOLDEN[f"{kind}_x"], 250.0)
    ours = res.channel_intervals(0)
    ours[:, :2] *= FS
    ref = GOLDEN[f"{kind}_intervals"]
    ne = res.not_evaluated[0] * FS
    if ne.size:
        ref = ref[~((ref[:, 0] < ne[:, 1].max()) & (ref[:, 1] > ne[:, 0].min()))]
    m = _iou_matrix(ref, ours)
    assert (m.max(axis=1) >= 0.8).all()
    if kind == "eeg":   # #3 R3: defaults (demean on) = the original: no extra detections, conf within 0.01
        assert ours.shape == ref.shape
        np.testing.assert_allclose(ours[:, :2], ref[:, :2], atol=0.5)
        np.testing.assert_allclose(ours[:, 2], ref[:, 2], atol=0.01)


def test_dc_offset_does_not_change_detections(detectors):
    x = _long_signal(4)
    a = detectors["eeg"].detect(x, 250).channel_intervals(0)
    b = detectors["eeg"].detect(x + 100 * x.std(), 250).channel_intervals(0)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_decode_and_nms_match_original_on_random_heads():
    try:
        _, _, evaluator = osn_reference.load_openspindlenet()
    except ImportError:
        pytest.skip("openspindlenet (the original package) is not installed")
    E = evaluator.Evaluator
    rng = np.random.default_rng(7)
    for _ in range(200):
        det = rng.uniform(0, 1, (30, 3)).astype(np.float32)
        det[0, 1:] = [0.9, 0.05]                     # keep away from the start==0 quirk
        ref = E.intervals_nms(E.detections_to_intervals(det, 7500, 0.5), iou_threshold=0.3)
        ours = interval_nms(decode_window(det, 7500, 0.5), 0.3)
        ours = ours[ours[:, 0] != 0]
        assert len(ours) == len(ref)
        np.testing.assert_allclose(_sorted(ours), _sorted(ref), atol=1e-2)


def test_original_quirk_drops_spindle_at_window_start():
    det = np.zeros((30, 3), dtype=np.float32)
    det[0] = [0.9, 0.1, 0.5]                          # centre 25 samples, 1 s long -> clipped start 0
    iv = interval_nms(decode_window(det))
    assert iv.shape == (1, 3) and iv[0, 0] == 0 and iv[0, 1] == pytest.approx(150.0)


# --- golden, multi-window (#3 R3): 6 min of a scalp night, original run window by window -----

LONG = np.load(os.path.join(os.path.dirname(__file__), "data", "spindle_golden_long.npz"))


def _long_ours(kind, **kw):
    res = SpindleDetector(kind, device="cpu", **kw).detect(LONG["x"].astype(np.float64), 250.0)
    assert res.evaluated_fraction(0) == 1.0
    iv = res.channel_intervals(0)
    iv[:, :2] *= FS
    return iv


def _split_window_starts(iv):
    """The original drops spindles starting exactly at a window's first sample (documented)."""
    ws = iv[:, 0] % WINDOW == 0
    return iv[~ws], iv[ws]


@pytest.mark.parametrize("kind", KINDS)
def test_long_recording_matches_original_window_by_window(kind):
    # pure pipeline (step = window, no demean, no flat check) == openspindlenet on each window
    ours, extra = _split_window_starts(_long_ours(kind, step_s=30, demean=False, flat_s=None))
    ref = LONG[f"{kind}_intervals"]
    assert len(ref) > 20 and len(extra) <= 2
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours[:, :2], ref[:, :2], atol=1e-2)
    np.testing.assert_allclose(ours[:, 2], ref[:, 2], atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_long_recording_default_settings_match_original(kind):
    # every default except the window step (the original tiles windows without overlap):
    # demean='auto' (on for eeg, off for ieeg), flat_s=0.5, gap handling, ownership, NMS
    ours, extra = _split_window_starts(_long_ours(kind, step_s=30))
    ref = LONG[f"{kind}_intervals"]
    assert ours.shape == ref.shape and len(extra) <= 2
    if kind == "eeg":     # demeaning: same spindles, edges within half a sample, conf within 0.01
        np.testing.assert_allclose(ours[:, :2], ref[:, :2], atol=0.5)
        np.testing.assert_allclose(ours[:, 2], ref[:, 2], atol=0.01)
    else:                 # 'ieeg' defaults = the original pipeline
        np.testing.assert_allclose(ours[:, :2], ref[:, :2], atol=1e-2)
        np.testing.assert_allclose(ours[:, 2], ref[:, 2], atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_long_recording_full_defaults_agree_with_original(kind):
    # step_s=20 (overlapping windows) sees different context at window edges, so not identical;
    # measured recall/precision vs the original (IoU >= 0.3): eeg 0.96/0.94, ieeg 0.89/0.89
    ours = _long_ours(kind)
    ref = LONG[f"{kind}_intervals"]
    m = _iou_matrix(ref, ours)
    assert (m.max(axis=1) >= 0.3).mean() >= 0.85 and (m.max(axis=0) >= 0.3).mean() >= 0.85


def test_long_recording_matches_live_original():
    try:
        osn, inference, _ = osn_reference.load_openspindlenet()
    except ImportError:
        pytest.skip("openspindlenet (the original package) is not installed")
    pytest.importorskip("pywt")
    x = LONG["x"].astype(np.float64)
    ref = []
    with osn_reference.pywt_precision10():
        for k in range(x.size // WINDOW):
            iv = _sorted(osn.detect(x[k * WINDOW:(k + 1) * WINDOW], model_type="eeg")["detection_intervals"])
            iv[:, :2] += k * WINDOW
            ref.append(iv)
    ref = _sorted(np.vstack(ref))
    np.testing.assert_allclose(ref, LONG["eeg_intervals"], atol=1e-5)     # the frozen file is current
    ours, _ = _split_window_starts(_long_ours("eeg", step_s=30, demean=False, flat_s=None))
    np.testing.assert_allclose(ours[:, :2], ref[:, :2], atol=1e-2)


# --- long recordings: windows, resampling, channels ------------------------------------

def _long_signal(n_rep=4):
    """Continuous ~2 min signal from the two sample windows (reversed copies keep it continuous)."""
    a = GOLDEN["eeg_x"]
    parts = [a if i % 2 == 0 else a[::-1] for i in range(n_rep)]
    return np.concatenate(parts)


def _iou_matrix(a, b):
    s = np.maximum(a[:, None, 0], b[None, :, 0])
    e = np.minimum(a[:, None, 1], b[None, :, 1])
    inter = np.maximum(0, e - s)
    return inter / ((a[:, None, 1] - a[:, None, 0]) + (b[None, :, 1] - b[None, :, 0]) - inter)


def test_sliding_windows_no_duplicates_and_sorted(detectors):
    x = _long_signal(6)
    for step in (5.0, 20.0, 30.0):
        res = SpindleDetector("eeg", device="cpu", step_s=step).detect(x, 250)
        iv = res.channel_intervals(0)
        assert len(iv) > 5
        assert np.all(np.diff(iv[:, 0]) >= 0)
        m = _iou_matrix(iv, iv)
        np.fill_diagonal(m, 0)
        assert m.max() < 0.3
        assert np.all((iv[:, 0] >= 0) & (iv[:, 1] <= res.duration_s) & (iv[:, 1] > iv[:, 0]))


def test_step_sizes_agree_away_from_edges():
    x = _long_signal(6)
    a = SpindleDetector("eeg", device="cpu", step_s=20).detect(x, 250).channel_intervals(0)
    b = SpindleDetector("eeg", device="cpu", step_s=10).detect(x, 250).channel_intervals(0)
    m = _iou_matrix(a, b)
    # most spindles found with both hops (the network sees different context, so not all)
    assert (m.max(axis=1) >= 0.3).mean() > 0.8


def test_multichannel_equals_single_channel(detectors):
    x = _long_signal(4)
    y = np.stack([x, -x[::-1], 0.5 * x])
    det = detectors["eeg"]
    res = det.detect(y, 250)
    assert res.n_channels == 3
    for ch in range(3):
        single = det.detect(y[ch], 250).channel_intervals(0)
        np.testing.assert_allclose(res.channel_intervals(ch), single)
    arr = res.to_array()
    assert arr.shape == (len(res), 4) and set(np.unique(arr[:, 3])) <= {0.0, 1.0, 2.0}


def test_scale_invariance(detectors):
    x = _long_signal(4)
    a = detectors["eeg"].detect(x, 250).channel_intervals(0)
    b = detectors["eeg"].detect(1e-6 * x, 250).channel_intervals(0)   # volts instead of uV
    np.testing.assert_allclose(a[:, :2], b[:, :2], atol=1e-3)
    np.testing.assert_allclose(a[:, 2], b[:, 2], atol=1e-4)


@pytest.mark.parametrize("fs", [500.0, 512.0, 1000.0])
def test_resampled_input_gives_same_spindles(detectors, fs):
    from scipy.signal import resample_poly
    from fractions import Fraction
    x = _long_signal(4)
    r = Fraction(fs / 250).limit_denominator(1000)
    xf = resample_poly(x, r.numerator, r.denominator)
    a = detectors["eeg"].detect(x, 250).channel_intervals(0)
    b = detectors["eeg"].detect(xf, fs).channel_intervals(0)
    m = _iou_matrix(a, b)
    assert (m.max(axis=1) >= 0.5).mean() > 0.9 and (m.max(axis=0) >= 0.5).mean() > 0.9


@pytest.mark.parametrize("fs", [250.0, 256.0, 500.0, 512.0, 1000.0, 1024.0, 200.0, 499.907, 32556.0, 30000.0,
                                2048.0, 50.0, 100.0, 1e6, 30000.5])
def test_resample_ratio_is_accurate(fs):
    up, down, fs_out = _resample_ratio(fs)
    assert up >= 1 and down >= 1 and max(up, down) <= 10 ** 5
    assert abs(fs_out - 250) / 250 <= 1e-4 and fs_out == pytest.approx(fs * up / down)
    if fs == 250:
        assert up == down == 1


@pytest.mark.parametrize("fs, ratio", [(256.0, (125, 128)), (2048.0, (125, 1024)), (512.0, (125, 256)),
                                       (32556.0, (125, 16278)), (32768.0, (125, 16384)), (200.0, (5, 4)),
                                       (50.0, (5, 1)), (30000.0, (1, 120)), (1e6, (1, 4000))])
def test_resample_ratio_is_exact_when_possible(fs, ratio):
    # #3 R4: 256 Hz used to give 83/85 and 2048 Hz 99/811 (within 1e-4, but not exact)
    up, down, fs_out = _resample_ratio(fs)
    assert (up, down) == ratio and fs_out == 250.0


@pytest.mark.parametrize("fs", [5e7, 1e9])
def test_resample_ratio_unrepresentable_raises(fs):
    # #3 R4: 5e7 Hz used to give up == 0 and an opaque resample_poly error
    with pytest.raises(ValueError, match="downsample the recording first"):
        _resample_ratio(fs)


def test_low_rates_down_to_50_hz_are_accepted(detectors):
    # #3 R6: 50 Hz DREAMS recordings were in the eeg model's training data (upsampled to 250 Hz)
    from scipy.signal import resample_poly
    x = _long_signal(4)
    a = detectors["eeg"].detect(x, 250).channel_intervals(0)
    for fs, (up, down) in ((100.0, (2, 5)), (50.0, (1, 5))):
        res = detectors["eeg"].detect(resample_poly(x, up, down), fs)
        assert res.params["resample"] == ((5, 2) if fs == 100 else (5, 1)) and res.params["fs_model"] == 250.0
        b = res.channel_intervals(0)
        assert len(b) > 0.5 * len(a)                       # spindles (11-16 Hz) survive 50 Hz sampling
        m = _iou_matrix(a, b)
        assert (m.max(axis=0) >= 0.3).mean() > 0.7         # and are mostly the same ones


# --- demean (#3 R2) --------------------------------------------------------------------

def test_demean_auto_follows_each_models_training():
    assert SpindleDetector("eeg", device="cpu").demean is True
    assert SpindleDetector("ieeg", device="cpu").demean is False
    assert SpindleDetector("ieeg", device="cpu", demean=True).demean is True
    assert SpindleDetector("eeg", device="cpu", demean=False).demean is False
    for bad in ("yes", None, 1, "on"):
        with pytest.raises(ValueError, match="demean"):
            SpindleDetector("eeg", device="cpu", demean=bad)


def test_large_offset_without_demean_warns_and_is_recorded():
    x = _long_signal(4)
    det = SpindleDetector("ieeg", device="cpu")                # demean off (auto)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = det.detect(x, 250)                               # ~0 SD offset: no warning
    assert res.params["demean"] is False and res.params["max_abs_offset_sd"] < 1
    with pytest.warns(RuntimeWarning, match="DC offset"):
        res = det.detect(x + 10 * x.std(), 250)
    assert res.params["max_abs_offset_sd"] > 5
    with warnings.catch_warnings():                            # demean on: offsets are harmless
        warnings.simplefilter("error")
        SpindleDetector("ieeg", device="cpu", demean=True).detect(x + 10 * x.std(), 250)


# --- gaps -------------------------------------------------------------------------------

def test_nan_gap_drops_spindles_and_is_reported(detectors):
    x = _long_signal(4)
    det = detectors["eeg"]
    base = det.detect(x, 250).channel_intervals(0)
    s0, e0 = base[1, 0], base[1, 1]
    g0, g1 = (s0 + e0) / 2 - 0.5, (s0 + e0) / 2 + 0.5        # 1 s gap inside spindle #1
    y = x.copy()
    y[int(g0 * 250):int(g1 * 250)] = np.nan
    res = det.detect(y, 250)
    iv = res.channel_intervals(0)
    margin = det.gap_margin_s
    assert not np.any((iv[:, 0] < g1 + margin) & (iv[:, 1] > g0 - margin))
    ne = res.not_evaluated[0]
    assert ne.shape == (1, 2) and ne[0, 0] == pytest.approx(g0 - margin, abs=0.01) and ne[0, 1] == pytest.approx(g1 + margin, abs=0.01)
    far = base[(base[:, 1] < g0 - 20) | (base[:, 0] > g1 + 20)]
    m = _iou_matrix(far, iv)
    assert (m.max(axis=1) >= 0.5).all()                       # spindles far from the gap survive


def test_mostly_gap_window_is_skipped_and_reported(detectors):
    x = _long_signal(4)                                       # 120 s
    y = x.copy()
    y[30 * 250:80 * 250] = np.nan                             # 50 s gap
    res = detectors["eeg"].detect(y, 250)
    ne = res.not_evaluated[0]
    assert ne[:, 0].min() <= 30 - 0.5 and ne[:, 1].max() >= 80 + 0.5
    iv = res.channel_intervals(0)
    assert not np.any((iv[:, 0] < 80.5) & (iv[:, 1] > 29.5))
    assert 0 < res.evaluated_fraction(0) < 0.6


def test_all_nan_and_flat_channels_are_not_evaluated(detectors):
    x = _long_signal(4)
    y = np.stack([x, np.full_like(x, np.nan), np.full_like(x, 7.0), x])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # no stray warnings
        res = detectors["eeg"].detect(y, 250)
    for ch in (1, 2):
        assert np.sum(res.channel == ch) == 0
        np.testing.assert_allclose(res.not_evaluated[ch], [[0.0, res.duration_s]])
        assert res.evaluated_fraction(ch) == 0.0
    np.testing.assert_allclose(res.channel_intervals(0), res.channel_intervals(3))


def test_flat_runs_are_gaps(detectors):
    x = GOLDEN["ieeg_x"]
    runs = _flat_runs(x, 250, 0.5)
    np.testing.assert_array_equal(runs, [[2444, 3603]])       # the zero-filled dropout of the sample
    res = SpindleDetector("ieeg", device="cpu").detect(x, 250)
    ne = res.not_evaluated[0]
    assert ne.shape == (1, 2)
    assert ne[0, 0] == pytest.approx(2444 / 250 - 0.5) and ne[0, 1] == pytest.approx(3603 / 250 + 0.5)
    iv = res.channel_intervals(0)
    assert not np.any((iv[:, 0] < ne[0, 1]) & (iv[:, 1] > ne[0, 0]))


def test_flat_runs_are_constant_runs_not_staircases():
    # #3 R1: adjacent plateaus of different values are separate runs, never one long "flat" run
    stair = np.repeat(np.arange(10.0), 2)                          # 0,0,1,1,2,2,...
    runs = _flat_runs(stair, 4.0, 0.5)                             # n_min = 2 samples
    np.testing.assert_array_equal(runs, np.column_stack([np.arange(0, 20, 2), np.arange(2, 21, 2)]))
    assert _flat_runs(stair, 4.0, 0.75).size == 0                  # n_min = 3: no run is that long
    x = np.array([1, 2, 2, 2, 3, 3, 4, np.nan, np.nan, np.nan, 5, 5, 5], dtype=float)
    np.testing.assert_array_equal(_flat_runs(x, 1.0, 3.0), [[1, 4], [10, 13]])   # NaN runs are not flat runs
    assert all(np.ptp(x[a:b]) == 0 for a, b in _flat_runs(x, 1.0, 2.0))


def test_sample_repeated_recording_is_not_a_gap(detectors):
    # #3 R1: a 250 Hz signal stored at 500 Hz by repeating every sample (sample-and-hold export)
    # used to be one "flat run" (0 spindles, evaluated 0.0)
    x = _long_signal(4)
    a = detectors["eeg"].detect(x, 250)
    b = detectors["eeg"].detect(np.repeat(x, 2), 500)
    assert b.evaluated_fraction(0) == 1.0 and b.not_evaluated[0].size == 0
    ia, ib = a.channel_intervals(0), b.channel_intervals(0)
    assert len(ia) > 10
    m = _iou_matrix(ia, ib)                                        # same spindles, up to resampling
    assert (m.max(axis=1) >= 0.5).mean() > 0.9 and (m.max(axis=0) >= 0.5).mean() > 0.9


@pytest.mark.parametrize("fs2, lsb", [(2048, 50), (8192, 10), (32768, 2)])
def test_oversampled_quantised_signal_has_no_false_flat_runs(fs2, lsb):
    # #3 R1: slowly varying, oversampled, coarsely quantised iEEG-like data (std ~300)
    from scipy.signal import resample_poly
    rng = np.random.default_rng(11)
    x = np.cumsum(rng.standard_normal(2048 * 60))
    x = 300 * (x - x.mean()) / x.std()                             # brown noise, 60 s at 2048 Hz
    y = resample_poly(x, fs2 // 2048, 1) if fs2 != 2048 else x
    q = np.round(y / lsb) * lsb
    for a, b in _flat_runs(q, fs2, 0.5):
        assert np.ptp(q[a:b]) == 0                                 # only truly constant runs
    assert _flat_runs(q, fs2, 0.5).size == 0
    q[fs2 * 10:fs2 * 12] = q[fs2 * 10]                             # a real 2 s dropout is still found
    np.testing.assert_array_equal(_flat_runs(q, fs2, 0.5)[:, 1], [fs2 * 12])


def test_inf_is_a_gap(detectors):
    x = _long_signal(4)
    y = x.copy()
    y[1000] = np.inf
    res = detectors["eeg"].detect(y, 250)
    assert res.not_evaluated[0].shape == (1, 2)


# --- validation ---------------------------------------------------------------------------

def test_input_validation(detectors):
    det = detectors["eeg"]
    with pytest.raises(ValueError, match="at least 30"):
        det.detect(np.zeros(250 * 29), 250)
    with pytest.raises(ValueError, match=">= 50"):
        det.detect(np.zeros(5000), 49.9)
    with pytest.raises(ValueError):
        det.detect(np.zeros((2, 2, 8000)), 250)
    with pytest.raises(ValueError, match="NaN"):
        det.predict_windows(np.full(WINDOW, np.nan))
    with pytest.raises(ValueError, match="shape"):
        det.predict_windows(np.zeros(100))


@pytest.mark.parametrize("kw", [dict(model="scalp"), dict(step_s=0), dict(step_s=31), dict(confidence_threshold=2),
                                dict(nms_iou_threshold=0), dict(min_valid_fraction=1.5), dict(gap_margin_s=-1),
                                dict(flat_s=0), dict(device="tpu")])
def test_bad_parameters(kw):
    with pytest.raises(ValueError):
        SpindleDetector(**kw)


def test_exactly_30_s_at_other_rates(detectors):
    for fs in (256.0, 500.0):
        x = np.random.default_rng(0).standard_normal(int(30 * fs))
        res = detectors["eeg"].detect(x, fs)
        assert res.params["n_windows"] == 1 and res.duration_s == pytest.approx(30.0)


def test_dataframe_and_repr(detectors):
    res = detectors["eeg"].detect(_long_signal(2), 250)
    df = res.to_dataframe()
    assert list(df.columns) == ["start", "end", "duration", "confidence", "channel"] and len(df) == len(res)
    assert "SpindleDetector" in repr(detectors["eeg"])
