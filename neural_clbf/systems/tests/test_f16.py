"""Test the F16 dynamics"""
import pytest
from warnings import warn

import torch
import numpy as np

try:
    from neural_clbf.systems import F16
except ImportError:
    warn("Could not import F16 module")
    pytest.skip(
        "Could not import F16 module; is AeroBench installed?", allow_module_level=False
    )


def test_f16_init():
    """Test initialization of F16"""
    # Test instantiation with valid parameters
    valid_params = {"lag_error": 0.0}
    f16 = F16(valid_params)
    assert f16 is not None
    assert f16.n_dims == 16
    assert f16.n_controls == 4

    # Make sure control limits are OK
    upper_lim, lower_lim = f16.control_limits
    # Only Nz and throttle limits are specified, so only check those
    assert torch.allclose(upper_lim[0], torch.tensor(6.0))
    assert torch.allclose(upper_lim[-1], torch.tensor(1.0))
    assert torch.allclose(lower_lim[0], -torch.tensor(1.0))
    assert torch.allclose(lower_lim[-1], torch.tensor(0.0))

    # Test instantiation without all needed parameters
    incomplete_params_list = [
        {},
        {"fake_param": 1.0},
    ]
    for incomplete_params in incomplete_params_list:
        with pytest.raises(ValueError):
            f16 = F16(incomplete_params)


def test_f16_safe_unsafe_mask():
    """Test the safe and unsafe mask for the F16"""
    valid_params = {"lag_error": 0.0}
    f16 = F16(valid_params)
    # This point should be safe
    safe_x = torch.tensor(
        [
            [
                540.0,
                0.035,
                0.0,
                -np.pi / 8,
                -0.15 * np.pi,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1000.0,
                9.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )
    assert torch.all(f16.safe_mask(safe_x))

    # Thise point should be unsafe
    unsafe_x = torch.tensor(
        [
            [
                540.0,
                0.035,
                0.0,
                -np.pi / 8,
                -0.15 * np.pi,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                100.0,
                9.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )
    assert torch.all(f16.unsafe_mask(unsafe_x))
