from __future__ import annotations

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import GlobalForcing


def manual_roundtrip(values: np.ndarray, *, pad: int = 7, n_fft: int = 32) -> np.ndarray:
    pad_width = [(0, 0)] * values.ndim
    pad_width[0] = (pad, pad)
    spectrum = np.fft.rfft(np.pad(values, pad_width, mode="reflect"), n=n_fft, axis=0)
    spectrum[0] = 0.0
    return np.fft.irfft(spectrum, n=n_fft, axis=0)[pad : pad + values.shape[0]]


def forcing_inputs(*, chunks: bool = False) -> dict[str, xr.DataArray]:
    time = np.arange("2001-01-01", "2001-01-09", dtype="datetime64[D]")
    latitude = np.array([-30.0, 30.0])
    longitude = np.array([0.0, 10.0, 20.0])
    phase = 2.0 * np.pi * np.arange(time.size) / time.size
    M_ek_x_values = np.sin(phase)[:, None, None] * np.ones(
        (1, latitude.size, longitude.size)
    )
    M_ek_y_values = np.cos(phase)[:, None, None] * np.ones(
        (1, latitude.size, longitude.size)
    )
    if chunks:
        M_ek_x_values = dsa.from_array(M_ek_x_values, chunks=(4, 1, 3))
        M_ek_y_values = dsa.from_array(M_ek_y_values, chunks=(4, 1, 3))
    coordinates = {
        "time": time,
        "latitude": latitude,
        "longitude": longitude,
    }
    return {
        "M_ek_x": xr.DataArray(
            M_ek_x_values,
            dims=("time", "latitude", "longitude"),
            coords=coordinates,
            attrs={"units": "m2 s-1"},
        ),
        "M_ek_y": xr.DataArray(
            M_ek_y_values,
            dims=("time", "latitude", "longitude"),
            coords=coordinates,
            attrs={"units": "m^2/s"},
        ),
        "northern_transport": xr.DataArray(
            10.0 + np.cos(phase),
            dims="time",
            coords={"time": time},
            attrs={"units": "Sv"},
        ),
        "southern_transport": xr.DataArray(
            np.zeros(time.size),
            dims="time",
            coords={"time": time},
            attrs={"units": "m3 s-1"},
        ),
    }


def test_one_reflected_fft_convention_round_trips_every_input() -> None:
    forcing = GlobalForcing.from_time_series(
        **forcing_inputs(), padding_samples=7, n_fft=32
    )

    assert forcing.n_fft == 32
    assert forcing.padding_samples == 7
    assert forcing.sample_interval_seconds == 86400.0
    assert forcing.spectral.omega.attrs["units"] == "rad s-1"
    for name in forcing.time_domain.data_vars:
        reconstructed = forcing.inverse_transform(forcing.spectral[name])
        expected = manual_roundtrip(np.asarray(forcing.time_domain[name]))
        np.testing.assert_allclose(reconstructed, expected)
        assert complex(forcing.spectral[name].isel(omega=0).max()) == 0j
    assert forcing.time_domain.northern_transport.attrs["units"] == "m3 s-1"
    assert float(forcing.time_domain.northern_transport.max()) == pytest.approx(1e6)


def test_transform_and_inverse_inherit_the_forcing_coordinates() -> None:
    forcing = GlobalForcing.from_time_series(
        **forcing_inputs(), padding_samples=7, n_fft=32
    )
    signal = xr.DataArray(
        np.arange(16, dtype=float).reshape(8, 2),
        dims=("time", "region"),
        coords={"time": forcing.time_domain.time, "region": ["one", "two"]},
    )
    spectrum = forcing.transform(signal)
    result = forcing.inverse_transform(spectrum)

    assert spectrum.dims == ("omega", "region")
    assert result.dims == ("time", "region")
    np.testing.assert_allclose(result, manual_roundtrip(np.asarray(signal)))


def test_dask_ekman_transport_spectra_remain_lazy() -> None:
    forcing = GlobalForcing.from_time_series(
        **forcing_inputs(chunks=True), padding_samples=7, n_fft=32
    )

    assert isinstance(forcing.spectral.M_ek_x.data, dsa.Array)
    np.testing.assert_allclose(
        forcing.inverse_transform(forcing.spectral.M_ek_x).compute(),
        manual_roundtrip(np.asarray(forcing.time_domain.M_ek_x.compute())),
    )


def test_calendar_month_sampling_can_be_declared_explicitly() -> None:
    inputs = forcing_inputs()
    monthly = np.arange("2001-01", "2001-09", dtype="datetime64[M]")
    inputs = {
        name: array.assign_coords(time=monthly) for name, array in inputs.items()
    }
    with pytest.raises(ValueError, match="uniformly sampled"):
        GlobalForcing.from_time_series(**inputs)

    forcing = GlobalForcing.from_time_series(
        **inputs,
        sample_interval_seconds=365.25 * 86400.0 / 12.0,
        n_fft=32,
    )
    assert forcing.sample_interval_seconds == pytest.approx(365.25 * 86400.0 / 12.0)


def test_missing_or_inconsistent_forcing_fails_early() -> None:
    inputs = forcing_inputs()
    inputs["M_ek_x"] = inputs["M_ek_x"].where(
        inputs["M_ek_x"].time.dt.day != 3
    )
    with pytest.raises(ValueError, match="cannot contain missing"):
        GlobalForcing.from_time_series(**inputs)

    inputs = forcing_inputs()
    inputs["northern_transport"] = inputs["northern_transport"].isel(time=slice(1, None))
    with pytest.raises(ValueError):
        GlobalForcing.from_time_series(**inputs)
