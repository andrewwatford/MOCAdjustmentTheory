"""Tests for the stateless Fourier-transform contract."""

import numpy as np
import pytest
import xarray as xr
from dask.array import Array

from moc_adjustment_theory import (
    butterworth_filter,
    forward_transform,
    inverse_transform,
)


def _time(count: int = 4) -> np.ndarray:
    return np.datetime64("2000-01-01") + np.arange(count) * np.timedelta64(6, "h")


def test_forward_uses_requested_padding_and_angular_frequency() -> None:
    data = xr.DataArray([0.0, 1.0, 0.0, -1.0], dims="time", coords={"time": _time()})

    spectrum = forward_transform(data, pad_length=4)

    np.testing.assert_allclose(spectrum, np.fft.rfft(data.values, n=9))
    expected_omega = 2 * np.pi * np.fft.rfftfreq(9, d=6 * 60 * 60)
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

    spectrum = forward_transform(data, time_dim="sample")
    restored = inverse_transform(spectrum)

    xr.testing.assert_allclose(restored, data)
    assert restored.attrs == data.attrs
    assert restored["sample"].attrs == data["sample"].attrs


def test_lazy_round_trip_remains_dask_backed() -> None:
    data = xr.DataArray(
        [[1.0, -1.0, 1.0, -1.0], [2.0, 0.0, -2.0, 0.0]],
        dims=("point", "time"),
        coords={"time": _time()},
    ).chunk({"point": 1, "time": 2})

    spectrum = forward_transform(data)
    restored = inverse_transform(spectrum)

    assert isinstance(spectrum.data, Array)
    assert isinstance(restored.data, Array)
    assert spectrum.chunks[spectrum.get_axis_num("omega")] == (3,)
    xr.testing.assert_allclose(restored.compute(), data.compute())


def test_forward_retains_nonzero_mean() -> None:
    data = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": _time()})

    spectrum = forward_transform(data)
    assert spectrum.isel(omega=0).item() == pytest.approx(10.0)


def test_inverse_accepts_real_nonzero_dc() -> None:
    data = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": _time()})

    spectrum = forward_transform(data)
    restored = inverse_transform(spectrum)

    xr.testing.assert_allclose(restored, data)


def test_explicit_sample_spacing_preserves_calendar_labels() -> None:
    """A physical interval can override irregular monthly label spacing."""
    time = np.array(
        ["2000-01-01", "2000-02-01", "2000-03-01", "2000-04-01"],
        dtype="datetime64[D]",
    )
    data = xr.DataArray([0.0, 1.0, 0.0, -1.0], dims="time", coords={"time": time})
    spacing = 365.25 / 12 * 24 * 60 * 60

    spectrum = forward_transform(data, sample_spacing_seconds=spacing)
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
    spectrum.attrs["_moc_adjustment_fourier"]["pad_length"] = 3

    with pytest.raises(ValueError, match="inconsistent transform lengths"):
        inverse_transform(spectrum)


def test_inverse_rejects_imaginary_dc_component() -> None:
    data = xr.DataArray([0.0, 1.0, 0.0, -1.0], dims="time", coords={"time": _time()})
    spectrum = forward_transform(data)
    spectrum.data[0] = 1j

    with pytest.raises(ValueError, match="inconsistent with a real time series"):
        inverse_transform(spectrum)


def test_butterworth_filter_has_expected_forward_backward_response() -> None:
    count = 2001
    time = np.datetime64("2000-01-01") + np.arange(count) * np.timedelta64(1, "s")
    cutoff_hz = 0.05
    signal = np.sin(2.0 * np.pi * cutoff_hz * np.arange(count))
    data = xr.DataArray(
        signal,
        dims="time",
        coords={"time": time},
        attrs={"units": "m"},
        name="signal",
    )
    cutoff_omega = 2.0 * np.pi * cutoff_hz

    filtered = butterworth_filter(data, cutoff_omega, order=4)

    np.testing.assert_allclose(
        filtered[200:-200], signal[200:-200] / 2.0, atol=1e-10
    )
    assert filtered.name == data.name
    assert filtered.attrs == data.attrs


def test_butterworth_filter_handles_datasets_and_complete_masks() -> None:
    count = 65
    time = np.datetime64("2000-01-01") + np.arange(count) * np.timedelta64(1, "D")
    signal = np.sin(2.0 * np.pi * np.arange(count) / 10.0)
    dataset = xr.Dataset(
        {
            "series": (("point", "time"), [signal, [np.nan] * count]),
            "label": ("point", ["active", "masked"]),
        },
        coords={"point": [0, 1], "time": time},
        attrs={"title": "example"},
    )

    filtered = butterworth_filter(dataset, 2.0 * np.pi / (3.0 * 24 * 60 * 60))

    assert isinstance(filtered, xr.Dataset)
    assert filtered.attrs == dataset.attrs
    xr.testing.assert_identical(filtered.label, dataset.label)
    assert np.all(np.isfinite(filtered.series.sel(point=0)))
    assert np.all(np.isnan(filtered.series.sel(point=1)))
