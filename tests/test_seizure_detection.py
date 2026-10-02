import pytest
import warnings

from pathlib import Path


import numpy as np
from brainmaze_eeg_models.seizure import (
    infer_seizure_probability,
    predict_channel_seizure_probability,
    preprocess_input, load_trained_model,
)


def test_seizure_detector():
    fs = 250
    length = 60

    model = load_trained_model('modelA')

    x = np.random.randn(1, length * fs)
    xinp = preprocess_input(x, fs)
    y = infer_seizure_probability(xinp, model, use_cuda=False)

    assert y.shape[1] == xinp.shape[2]
    assert y.shape[0] == xinp.shape[0]


def test_seizure_detection_real_data():
    pth = Path(__file__).resolve()
    pth_data = pth.parent / 'data' / 'seizure_segment_1.npz'
    dat = np.load(pth_data)

    idx_ref_s = 380*2 # start where needs to be > 0.8 # < 0.8 at 20 s before this
    idx_ref_e = 450*2 # end where needs to be > 0.8 # < 0.8 at 20 s before this

    x = dat['data'].squeeze().copy()
    fs = float(dat['fs'].squeeze())
    fs = int(fs)

    txx_ref, prob = predict_channel_seizure_probability(x, fs, 'modelA', False, 1)

    # safe clamp helpers to avoid slicing outside array bounds
    idx_ref_s_clamped = max(0, min(idx_ref_s, prob.shape[0]))
    idx_ref_e_clamped = max(0, min(idx_ref_e, prob.shape[0]))

    bl_1 = (np.arange(prob.shape[0]) > idx_ref_s_clamped) & (np.arange(prob.shape[0]) < idx_ref_e_clamped)
    bl_2 = (np.arange(prob.shape[0]) > idx_ref_s_clamped-40) & (np.arange(prob.shape[0]) < idx_ref_e_clamped+40)

    # Assert that within the expected seizure interval there is a high probability (>0.8)
    # Gap samples inside the seizure (if any) are NaN by design -> only check evaluated points.
    assert np.all(prob[bl_1][np.isfinite(prob[bl_1])] > 0.8)

    # Assert that in the 10 seconds immediately before the seizure interval probabilities are low (<0.8)
    assert np.all(prob[~bl_2][np.isfinite(prob[~bl_2])] < 0.8)
    # NaN exactly at t=0 and at the 1 s segments that contain NaN samples of the recording
    hop = fs // 2
    nan_blocks = np.unique(np.where(np.isnan(x))[0] // hop)
    expected = np.unique(np.r_[0, nan_blocks, nan_blocks + 1])
    expected = expected[expected < prob.shape[0]]
    np.testing.assert_array_equal(np.where(np.isnan(prob))[0], expected)
    assert expected.size > 1  # the fixture really contains a short NaN gap


def test_process_data_over_multiple_batches():
    fs = 256
    dur_s = 60
    x = np.random.randn(dur_s*fs)
    txx_ref, prob = predict_channel_seizure_probability(x, fs, 'modelA', False, 1, window_s=10, step_s=1, n_batch=16, discard_edges_s=0.5)

    assert txx_ref.shape[0]
    assert prob.shape[0] == txx_ref.shape[0]


def test_invalid_model_name():
    """Test that an invalid model name raises KeyError."""
    with pytest.raises(KeyError):
        load_trained_model('invalid_model_name')


def test_2d_input_raises_error():
    """Test that 2D input to predict_channel_seizure_probability raises ValueError."""
    fs = 256
    dur_s = 60
    x = np.random.randn(2, dur_s * fs)  # 2D array (multiple channels)

    with pytest.raises(ValueError, match="single channel signal"):
        predict_channel_seizure_probability(x, fs, 'modelA', False, 1)


def test_preprocess_1d_input():
    """Test that preprocess_input handles 1D input correctly by reshaping."""
    fs = 256
    dur_s = 10
    x = np.random.randn(dur_s * fs)  # 1D array

    xinp = preprocess_input(x, fs)
    # Should be (1, 100, time_bins)
    assert xinp.ndim == 3
    assert xinp.shape[0] == 1
    assert xinp.shape[1] == 100


def test_preprocess_return_axes():
    """Test that preprocess_input returns axes when requested."""
    fs = 256
    dur_s = 10
    x = np.random.randn(1, dur_s * fs)

    result = preprocess_input(x, fs, return_axes=True)
    assert len(result) == 3  # (t, f, x)
    t, f, xinp = result
    assert len(t) > 0
    assert len(f) == 100
    assert xinp.ndim == 3


def test_modelB_loads():
    """Test that modelB can be loaded successfully."""
    model = load_trained_model('modelB', device='cpu')
    assert model is not None and model.device == 'cpu'
    assert model.input_names == ['x'] and model.output_names == ['logits', 'probs']