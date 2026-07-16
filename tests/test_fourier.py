"""Tests for the stateless Fourier-transform contract."""

import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import forward_transform, inverse_transform


def _time(count: int = 4) -> np.ndarray:
    return np.datetime64("2000-01-01") + np.arange(count) * np.timedelta64(6, "h")


def test_forward_uses_rfft_right_padding_and_angular_frequency() -> None:
    data = xr.DataArray([0.0, 1.0, 0.0, -1.0], dims="time", coords={"time": _time()})

    spectrum = forward_transform(data)

    np.testing.assert_allclose(spectrum, np.fft.rfft(data.values, n=8))
    expected_omega = 2 * np.pi * np.fft.rfftfreq(8, d=6 * 60 * 60)
    np.testing.assert_allclose(spectrum.omega, expected_omega)
    assert spectrum.omega.attrs["units"] == "rad s-1"


def test_round_trip_restores_values_coordinates_and_attributes() -> None:
    data = xr.DataArray(
        [[1.0, -1.0, 1.0, -1.0], [2.0, 0.0, -2.0, 0.0]],
        dims=("latitude", "sample"),
        coords={"latitude": [10.0, 20.0], "sample": _time()},
        attrs={"units": "m"},
        name="forcing",
    )
    data["sample"].attrs["long_name"] = "forcing time"

    spectrum = forward_transform(
        data, time_dim="sample", omega_dim="angular_frequency", norm="ortho"
    )
    restored = inverse_transform(
        spectrum, time_dim="sample", omega_dim="angular_frequency", norm="ortho"
    )

    xr.testing.assert_allclose(restored, data)
    assert restored.attrs == data.attrs
    assert restored["sample"].attrs == data["sample"].attrs


def test_forward_rejects_nonzero_mean_by_default() -> None:
    data = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": _time()})

    with pytest.raises(ValueError, match="must be zero mean"):
        forward_transform(data)

    spectrum = forward_transform(data, require_zero_mean=False)
    assert spectrum.isel(omega=0).item() == pytest.approx(10.0)


def test_forward_accepts_float32_residual_means_against_field_scale() -> None:
    data = xr.DataArray(
        np.array(
            [[100.0, -100.0, 100.0, -100.0], [1e-3, -1e-3, 1e-3, -0.999e-3]],
            dtype=np.float32,
        ),
        dims=("point", "time"),
        coords={"time": _time()},
    )

    spectrum = forward_transform(data)

    assert spectrum.sizes["omega"] == 5


def test_inverse_accepts_real_nonzero_dc() -> None:
    data = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": _time()})

    spectrum = forward_transform(data, require_zero_mean=False)
    restored = inverse_transform(spectrum)

    xr.testing.assert_allclose(restored, data)


@pytest.mark.parametrize(
    "time, message",
    [
        (
            np.array(["2000-01-01", "2000-01-02", "2000-01-04"], dtype="datetime64[D]"),
            "uniformly spaced",
        ),
        (
            np.array(["2000-01-01", "2000-01-02", "2000-01-02"], dtype="datetime64[D]"),
            "strictly increasing",
        ),
    ],
)
def test_forward_rejects_invalid_time_grid(time: np.ndarray, message: str) -> None:
    data = xr.DataArray([0.0, 1.0, -1.0], dims="time", coords={"time": time})

    with pytest.raises(ValueError, match=message):
        forward_transform(data)


def test_forward_rejects_nan() -> None:
    data = xr.DataArray([0.0, np.nan, 0.0, 0.0], dims="time", coords={"time": _time()})

    with pytest.raises(ValueError, match="must not contain NaN"):
        forward_transform(data)


def test_inverse_rejects_contract_mismatch() -> None:
    data = xr.DataArray([0.0, 1.0, 0.0, -1.0], dims="time", coords={"time": _time()})
    spectrum = forward_transform(data)

    with pytest.raises(ValueError, match="pad_factor=3.*does not match"):
        inverse_transform(spectrum, pad_factor=3)


def test_inverse_rejects_imaginary_dc_component() -> None:
    data = xr.DataArray([0.0, 1.0, 0.0, -1.0], dims="time", coords={"time": _time()})
    spectrum = forward_transform(data)
    spectrum.data[0] = 1j

    with pytest.raises(ValueError, match="inconsistent with a real time series"):
        inverse_transform(spectrum)
