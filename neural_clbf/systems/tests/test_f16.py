"""Test the 2D quadrotor dynamics"""
import pytest

from neural_clbf.systems import F16


def test_f16_init():
    """Test initialization of F16"""
    # Test instantiation with valid parameters
    valid_params = {"lag_error": 0.0}
    f16 = F16(valid_params)
    assert f16 is not None
    assert f16.n_dims == 16
    assert f16.n_controls == 4

    # Test instantiation without all needed parameters
    incomplete_params_list = [
        {},
        {"fake_param": 1.0},
    ]
    for incomplete_params in incomplete_params_list:
        with pytest.raises(ValueError):
            f16 = F16(incomplete_params)
