"""Robustness tests for the seizure-probability pipeline.

Ported from brainmaze-torch 0.2.0 (tests/test_robustness.py). The PyTorch-specific tests
(model train/eval mode and device restoration, inference_mode) are replaced by their ONNX
counterparts at the end (device arguments, PyTorch models rejected, model cache).

These pin down the output contract of ``predict_channel_seizure_probability``:
complete coverage wherever data exists, NaN (never 0) for anything that was not
evaluated, and explicit errors for unsupported inputs instead of silent garbage.
"""
import numpy as np
import pytest

from brainmaze_eeg_models import runtime
from brainmaze_eeg_models.runtime import OnnxModel
from brainmaze_eeg_models.seizure import (
    infer_seizure_probability,
    load_trained_model,
    predict_channel_seizure_probability,
    preprocess_input,
)

FS = 256
HOP = FS // 2
# Small windows keep the tests fast; window/step/edge logic is identical to the defaults.
KW = dict(window_s=10, step_s=1, discard_edges_s=0.5, n_batch=16)
# float32 inference is not bit-reproducible across BLAS/oneDNN kernels, batch sizes
# or platforms; compare model outputs with an absolute tolerance (NaN masks exactly).
ATOL = 1e-5


def assert_prob_close(a, b, atol=ATOL):
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, rtol=0, atol=atol, equal_nan=True)


@pytest.fixture(scope="module")
def model():
    return load_trained_model('modelA', device='cpu')


def _expected_nan_idx(x, n_out):
    """Grid indices whose 1 s segment contains a non-finite sample, plus t=0."""
    bad_blocks = np.unique(np.where(~np.isfinite(x))[0] // HOP)
    idx = np.unique(np.r_[0, bad_blocks, bad_blocks + 1])
    return idx[idx < n_out]


# --------------------------------------------------------------------------- #
# Fix 1: coverage -- no uncovered time reported as probability 0
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("extra_samples", [0, 1, 127, 128, 129, 255, 3 * FS + 77])
def test_full_coverage_including_tail(model, extra_samples):
    rng = np.random.default_rng(extra_samples)
    x = rng.standard_normal(60 * FS + extra_samples)
    t, p = predict_channel_seizure_probability(x, FS, model, **KW)

    assert t.shape == p.shape == (x.size // HOP,)
    np.testing.assert_allclose(t, np.arange(t.size) * 0.5)
    # last stamp = centre of the last complete 1 s segment
    assert t[-1] + 0.5 <= x.size / FS < t[-1] + 1.0
    # only t=0 is undefined; everything else is a genuine model output
    assert np.isnan(p[0])
    assert np.all(np.isfinite(p[1:]))
    assert np.all((p[1:] >= 0) & (p[1:] <= 1))


def test_full_coverage_default_parameters(model):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(700 * FS + 123)  # 700 s, not a multiple of step_s
    t, p = predict_channel_seizure_probability(x, FS, model)
    assert np.all(np.isfinite(p[1:]))
    assert t[-1] == 699.5


def test_recording_edges_nan_when_not_filled(model):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(60 * FS)
    _, p = predict_channel_seizure_probability(x, FS, model, fill_recording_edges=False, **KW)
    e = int(KW['discard_edges_s'] * 2)
    assert np.all(np.isnan(p[:1 + e]))
    assert np.all(np.isnan(p[-e:]))
    assert np.all(np.isfinite(p[1 + e:-e]))


def test_n_batch_does_not_change_result(model):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(40 * FS + 50)
    _, p1 = predict_channel_seizure_probability(x, FS, model, **{**KW, 'n_batch': 1})
    _, p2 = predict_channel_seizure_probability(x, FS, model, **{**KW, 'n_batch': 1000})
    # identical up to float32 batched-matmul rounding
    assert_prob_close(p1, p2)


@pytest.mark.parametrize("window_s, step_s, discard_edges_s", [(10, 9.5, 0), (10, 8.5, 0.5), (300, 279.5, 10)])
def test_largest_valid_step_has_no_holes(model, window_s, step_s, discard_edges_s):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int(2.7 * window_s * FS))
    _, p = predict_channel_seizure_probability(
        x, FS, model, window_s=window_s, step_s=step_s, discard_edges_s=discard_edges_s,
        fill_recording_edges=False)
    e = int(discard_edges_s * 2)
    inner = p[1 + e:p.size - e] if e else p[1:]
    assert np.all(np.isfinite(inner))


@pytest.mark.parametrize("window_s, step_s, discard_edges_s", [(10, 10, 0), (10, 9, 0.5), (300, 280, 10), (10, 1, 5)])
def test_step_leaving_holes_raises(window_s, step_s, discard_edges_s):
    x = np.random.default_rng(0).standard_normal(1000 * FS)
    with pytest.raises(ValueError):
        predict_channel_seizure_probability(x, FS, 'modelA', window_s=window_s, step_s=step_s,
                                            discard_edges_s=discard_edges_s)


# --------------------------------------------------------------------------- #
# Fix 2: NaN gaps, flat channels, z-score guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gap_start_s, gap_end_s", [(15, 45), (15.1, 44.9), (20.25, 20.26), (0, 3.3), (57.7, 60)])
def test_nan_gap_is_nan_consistently(model, gap_start_s, gap_end_s):
    rng = np.random.default_rng(4)
    x = rng.standard_normal(60 * FS)
    x[int(gap_start_s * FS):int(gap_end_s * FS)] = np.nan
    _, p = predict_channel_seizure_probability(x, FS, model, min_valid_fraction=0.0, **KW)
    expected = _expected_nan_idx(x, p.size)
    np.testing.assert_array_equal(np.where(np.isnan(p))[0], expected)


def test_inf_treated_like_nan(model):
    rng = np.random.default_rng(5)
    x = rng.standard_normal(30 * FS)
    x[10 * FS + 3] = np.inf
    _, p = predict_channel_seizure_probability(x, FS, model, **KW)
    np.testing.assert_array_equal(np.where(np.isnan(p))[0], _expected_nan_idx(x, p.size))


def test_gap_never_reported_as_zero_probability(model):
    rng = np.random.default_rng(6)
    x = rng.standard_normal(1500 * FS)
    x[400 * FS:1000 * FS] = np.nan
    t, p = predict_channel_seizure_probability(x, FS, model)
    gap = (t >= 400) & (t <= 1000)
    assert np.all(np.isnan(p[gap]))
    assert np.all(np.isfinite(p[(t > 0) & ~gap]))


@pytest.mark.parametrize("mvf", [0.0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("gap", [(0, 25), (5, 25), (35, 55), (40, 60)])
def test_min_valid_fraction_never_nans_valid_data(model, mvf, gap):
    # long gaps next to the recording edges: no window there has mostly valid data,
    # yet every valid segment must still get a value (review R2)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(60 * FS)
    x[gap[0] * FS:gap[1] * FS] = np.nan
    _, p = predict_channel_seizure_probability(x, FS, model, min_valid_fraction=mvf, **KW)
    np.testing.assert_array_equal(np.where(np.isnan(p))[0], _expected_nan_idx(x, p.size))


def test_min_valid_fraction_default_is_published_method(model):
    rng = np.random.default_rng(9)
    x = rng.standard_normal(60 * FS)
    x[30 * FS:50 * FS] = np.nan
    _, p_def = predict_channel_seizure_probability(x, FS, model, **KW)
    _, p_0 = predict_channel_seizure_probability(x, FS, model, min_valid_fraction=0.0, **KW)
    np.testing.assert_array_equal(p_def, p_0)
    # a preference > 0 may only change values, never the NaN set
    _, p_pref = predict_channel_seizure_probability(x, FS, model, min_valid_fraction=0.9, **KW)
    np.testing.assert_array_equal(np.isnan(p_pref), np.isnan(p_0))


def test_tail_window_never_changes_regular_values(model):
    # the extra end-aligned window may only fill time no regular window covers (review R1):
    # a recording that ends off the step grid gives the same values as the same signal
    # cut back onto the grid, wherever the latter has regular-window coverage
    rng = np.random.default_rng(10)
    x = rng.standard_normal(80 * FS)
    kw = dict(window_s=10, step_s=4, discard_edges_s=0.5, n_batch=16)
    _, p_on = predict_channel_seizure_probability(x[:74 * FS], FS, model, fill_recording_edges=False, **kw)
    _, p_off = predict_channel_seizure_probability(x[:77 * FS + 50], FS, model, **kw)
    covered = np.isfinite(p_on)
    assert covered.sum() > 100
    # same regular windows; batch composition differs -> float tolerance
    np.testing.assert_allclose(p_off[:p_on.size][covered], p_on[covered], rtol=0, atol=ATOL)
    assert np.all(np.isfinite(p_off[1:]))


def test_all_nan_input_is_all_nan(model):
    x = np.full(30 * FS, np.nan)
    t, p = predict_channel_seizure_probability(x, FS, model, **KW)
    assert t.size == p.size and np.all(np.isnan(p))


def test_flat_channel_is_nan_not_zero(model):
    for value in (0.0, 3.7):
        x = np.full(30 * FS, value)
        _, p = predict_channel_seizure_probability(x, FS, model, **KW)
        assert np.all(np.isnan(p))


def test_flat_segment_is_nan(model):
    rng = np.random.default_rng(8)
    x = rng.standard_normal(60 * FS)
    x[20 * FS:23 * FS] = 0.0   # 3 s flat line (e.g. amplifier dropout)
    t, p = predict_channel_seizure_probability(x, FS, model, **KW)
    flat = (t >= 20.5) & (t <= 22.5)   # 1 s segments entirely inside the flat part
    assert np.all(np.isnan(p[flat]))
    assert np.all(np.isfinite(p[(t > 0) & ~flat]))


def test_preprocess_guards_against_zero_std():
    # constant input, single-column input and all-NaN input must not create NaN/inf
    for x in (np.zeros((2, 10 * FS)), np.ones(FS), np.random.randn(FS), np.full(5 * FS, np.nan)):
        with np.errstate(all='raise'):
            sxx = preprocess_input(x, FS)
        assert np.all(np.isfinite(sxx))


def test_infer_rejects_non_finite(model):
    sxx = np.zeros((1, 100, 20))
    sxx[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        infer_seizure_probability(sxx, model)


# --------------------------------------------------------------------------- #
# Fix 3: sampling-rate handling
# --------------------------------------------------------------------------- #
def test_float_fs_equivalent_to_int(model):
    rng = np.random.default_rng(9)
    x = rng.standard_normal(30 * 500)
    _, p_int = predict_channel_seizure_probability(x, 500, model, **KW)
    _, p_flt = predict_channel_seizure_probability(x, 500.0, model, **KW)
    _, p_np = predict_channel_seizure_probability(x, np.float32(500), model, **KW)
    assert_prob_close(p_int, p_flt)
    assert_prob_close(p_int, p_np)


@pytest.mark.parametrize("fs", [128, 198, 199, 255, 499.9, 1e3 + 0.5, float('nan'), 'abc', None, True])
def test_invalid_fs_raises_value_error(fs):
    x = np.random.default_rng(0).standard_normal(30 * 512)
    with pytest.raises(ValueError):
        predict_channel_seizure_probability(x, fs, 'modelA', **KW)
    with pytest.raises(ValueError):
        preprocess_input(x[:10 * 512], fs)


@pytest.mark.parametrize("fs", [200, 250, 500, 512, 1000, 1024])
def test_supported_fs_are_on_half_second_grid(model, fs):
    x = np.random.default_rng(fs).standard_normal(int(20.3 * fs))
    t, p = predict_channel_seizure_probability(x, fs, model, **KW)
    assert t.size == x.size // (fs // 2)
    np.testing.assert_allclose(np.diff(t), 0.5)
    assert np.all(np.isfinite(p[1:]))
    tt, ff, sxx = preprocess_input(x[:10 * fs], fs, return_axes=True)
    np.testing.assert_allclose(tt, (np.arange(19) + 1) * 0.5)
    np.testing.assert_allclose(ff, np.arange(100))


def test_window_alignment_on_grid(model):
    """A window placed at t0 must reproduce the low-level result shifted by 2*t0."""
    rng = np.random.default_rng(10)
    x = rng.standard_normal(10 * FS)
    t, p = predict_channel_seizure_probability(x, FS, model, window_s=10, step_s=1, discard_edges_s=0)
    y = infer_seizure_probability(preprocess_input(x, FS), model)[0]
    np.testing.assert_allclose(p[1:], y, rtol=0, atol=ATOL)


# --------------------------------------------------------------------------- #
# LOW items
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("discard", [0, 0.5, 1])
def test_small_discard_edges_work(model, discard):
    x = np.random.default_rng(11).standard_normal(30 * FS)
    _, p = predict_channel_seizure_probability(x, FS, model, window_s=10, step_s=1, discard_edges_s=discard)
    assert np.all(np.isfinite(p[1:]))


@pytest.mark.parametrize("bad", [dict(discard_edges_s=0.25), dict(step_s=0.3), dict(window_s=10.2),
                                 dict(discard_edges_s=-1), dict(step_s=0), dict(window_s=1),
                                 dict(discard_edges_s=5), dict(min_valid_fraction=1.5), dict(n_batch=0)])
def test_invalid_window_parameters_raise(bad):
    x = np.random.default_rng(0).standard_normal(60 * FS)
    with pytest.raises(ValueError):
        predict_channel_seizure_probability(x, FS, 'modelA', **{**KW, **bad})


def test_discard_edges_equal_window_raises():
    x = np.random.default_rng(0).standard_normal(60 * FS)
    with pytest.raises(ValueError):
        predict_channel_seizure_probability(x, FS, 'modelA', window_s=10, step_s=1, discard_edges_s=10)


def test_recording_shorter_than_window_raises():
    x = np.random.default_rng(0).standard_normal(200 * FS)
    with pytest.raises(ValueError, match="shorter"):
        predict_channel_seizure_probability(x, FS, 'modelA')


def test_list_and_integer_inputs(model):
    rng = np.random.default_rng(12)
    x = rng.standard_normal(20 * FS)
    _, p_ref = predict_channel_seizure_probability(x, FS, model, **KW)
    _, p_list = predict_channel_seizure_probability(list(x), FS, model, **KW)
    assert_prob_close(p_ref, p_list)
    xi = (x * 1000).astype(np.int16)
    _, p_int = predict_channel_seizure_probability(xi, FS, model, **KW)
    assert np.all(np.isfinite(p_int[1:]))


def test_caller_data_not_modified(model):
    x = np.random.default_rng(13).standard_normal(20 * FS)
    x[100:200] = np.nan
    x_copy = x.copy()
    predict_channel_seizure_probability(x, FS, model, **KW)
    preprocess_input(x, FS)
    np.testing.assert_array_equal(x, x_copy)


def test_infer_output_layout(model):
    rng = np.random.default_rng(15)
    x = rng.standard_normal((3, 10 * FS))
    sxx = preprocess_input(x, FS)
    y = infer_seizure_probability(sxx, model)
    assert y.shape == (3, sxx.shape[2])
    # each batch row must equal running that row alone (no batch/time transposition)
    for i in range(3):
        np.testing.assert_allclose(y[i], infer_seizure_probability(sxx[i:i + 1], model)[0], atol=ATOL)


def test_infer_rejects_wrong_shape(model):
    with pytest.raises(ValueError):
        infer_seizure_probability(np.zeros((100, 20)), model)
    with pytest.raises(ValueError):
        infer_seizure_probability(np.zeros((1, 99, 20)), model)


def test_inference_does_not_depend_on_batch_split(model):
    sxx = preprocess_input(np.random.default_rng(16).standard_normal((5, 10 * FS)), FS)
    y = infer_seizure_probability(sxx, model)
    small = OnnxModel(model.path, device='cpu', batch_size=2, output_batch_axes={'logits': 1, 'probs': 1})
    assert_prob_close(infer_seizure_probability(sxx, small), y)


# --------------------------------------------------------------------------- #
# ONNX runtime: device arguments, model objects
# --------------------------------------------------------------------------- #
def test_names_and_loaded_models_give_the_same_result(model):
    x = np.random.default_rng(17).standard_normal(20 * FS)
    _, p1 = predict_channel_seizure_probability(x, FS, 'modelA', use_cuda=False, **KW)
    _, p2 = predict_channel_seizure_probability(x, FS, model, **KW)
    _, p3 = predict_channel_seizure_probability(x, FS, 'modelA', device='cpu', **KW)
    assert_prob_close(p1, p2)
    assert_prob_close(p1, p3)


def test_loaded_models_are_cached():
    assert load_trained_model('modelA', device='cpu') is load_trained_model('modelA', device='cpu')
    assert load_trained_model('modelA', device='cpu') is not load_trained_model('modelB', device='cpu')


def test_device_conflicts_raise(model, monkeypatch):
    x = np.random.default_rng(18).standard_normal(20 * FS)
    with pytest.raises(ValueError, match="contradicts"):
        predict_channel_seizure_probability(x, FS, 'modelA', use_cuda=True, device='cpu', **KW)
    with pytest.raises(ValueError, match="runs on 'cpu'"):
        predict_channel_seizure_probability(x, FS, model, use_cuda=True, **KW)
    with pytest.raises(ValueError, match="runs on 'cpu'"):
        infer_seizure_probability(np.zeros((1, 100, 5)), model, device='cuda')
    # no CUDA provider: use_cuda=True must raise, never run on the CPU silently
    monkeypatch.setattr(runtime._ort(), "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="device='cuda'"):
        predict_channel_seizure_probability(x, FS, 'modelA', use_cuda=True, cuda_number=3, **KW)


def test_device_errors_do_not_depend_on_data():
    x = np.full(20 * FS, np.nan)            # nothing to evaluate
    with pytest.raises(ValueError, match="contradicts"):
        predict_channel_seizure_probability(x, FS, 'modelA', use_cuda=False, device='cuda', **KW)


def test_torch_like_model_rejected():
    class FakeTorchModel:
        def __call__(self, x):
            raise AssertionError("must not be called")
    with pytest.raises(TypeError, match="ONNX"):
        infer_seizure_probability(np.zeros((1, 100, 5)), FakeTorchModel())
