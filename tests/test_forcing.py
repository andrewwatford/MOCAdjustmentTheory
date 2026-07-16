from __future__ import annotations

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import FFTConvention
from moc_adjustment_theory.forcing import _normalise_forcing, _resolve_fft


def manual_roundtrip(values: np.ndarray, *, pad: int = 7, n_fft: int = 32) -> np.ndarray:
    pad_width = [(0, 0)] * values.ndim
    pad_width[0] = (pad, pad)
    spectrum = np.fft.rfft(np.pad(values, pad_width, mode="reflect"), n=n_fft, axis=0)
    spectrum[0] = 0.0
    return np.fft.irfft(spectrum, n=n_fft, axis=0)[pad : pad + values.shape[0]]


def forcing_dataset(*, chunks: bool = False) -> xr.Dataset:
    time = np.arange("2001-01-01", "2001-01-09", dtype="datetime64[D]")
    latitude = np.array([-30.0, 30.0])
    longitude = np.array([0.0, 10.0, 20.0])
    phase = 2.0 * np.pi * np.arange(time.size) / time.size
    M_Ek_x = np.sin(phase)[:, None, None] * np.ones(
        (1, latitude.size, longitude.size)
    )
    M_Ek_y = np.cos(phase)[:, None, None] * np.ones(
        (1, latitude.size, longitude.size)
    )
    if chunks:
        M_Ek_x = dsa.from_array(M_Ek_x, chunks=(4, 1, 3))
        M_Ek_y = dsa.from_array(M_Ek_y, chunks=(4, 1, 3))
    coordinates = {"time": time, "latitude": latitude, "longitude": longitude}
    return xr.Dataset(
        {
            "M_Ek_x": xr.DataArray(
                M_Ek_x,
                dims=("time", "latitude", "longitude"),
                coords=coordinates,
                attrs={"units": "m2 s-1"},
            ),
            "M_Ek_y": xr.DataArray(
                M_Ek_y,
                dims=("time", "latitude", "longitude"),
                coords=coordinates,
                attrs={"units": "m^2/s"},
            ),
            "T_N": xr.DataArray(
                10.0 + np.cos(phase),
                dims="time",
                coords={"time": time},
                attrs={"units": "Sv"},
            ),
        }
    )


def test_one_reflected_fft_convention_round_trips_every_input() -> None:
    forcing = _normalise_forcing(forcing_dataset())
    transform = _resolve_fft(
        FFTConvention(padding_samples=7, n_fft=32), forcing.time
    )
    spectral = transform.transform_dataset(forcing)

    assert spectral.attrs["n_fft"] == 32
    assert spectral.attrs["padding_samples"] == 7
    assert spectral.attrs["sample_interval_seconds"] == 86400.0
    assert spectral.omega.attrs["units"] == "rad s-1"
    for name in forcing.data_vars:
        reconstructed = transform.inverse_transform(spectral[name])
        np.testing.assert_allclose(
            reconstructed, manual_roundtrip(np.asarray(forcing[name]))
        )
        assert complex(spectral[name].isel(omega=0).max()) == 0j
    assert forcing.T_N.attrs["units"] == "m3 s-1"
    assert float(forcing.T_N.max()) == pytest.approx(1e6)


def test_dask_ekman_transport_spectra_remain_lazy() -> None:
    forcing = _normalise_forcing(forcing_dataset(chunks=True))
    transform = _resolve_fft(
        FFTConvention(padding_samples=7, n_fft=32), forcing.time
    )
    spectral = transform.transform_dataset(forcing)

    assert isinstance(spectral.M_Ek_x.data, dsa.Array)
    np.testing.assert_allclose(
        transform.inverse_transform(spectral.M_Ek_x).compute(),
        manual_roundtrip(np.asarray(forcing.M_Ek_x.compute())),
    )


def test_calendar_month_sampling_can_be_declared_explicitly() -> None:
    forcing = forcing_dataset().assign_coords(
        time=np.arange("2001-01", "2001-09", dtype="datetime64[M]")
    )
    with pytest.raises(ValueError, match="uniformly sampled"):
        _resolve_fft(FFTConvention(), forcing.time)

    transform = _resolve_fft(
        FFTConvention(
            sample_interval_seconds=365.25 * 86400.0 / 12.0,
            n_fft=32,
        ),
        forcing.time,
    )
    assert transform.sample_interval_seconds == pytest.approx(
        365.25 * 86400.0 / 12.0
    )


def test_explicit_interval_still_requires_increasing_unique_time() -> None:
    forcing = forcing_dataset()
    shuffled = np.asarray(forcing.time).copy()
    shuffled[[2, 3]] = shuffled[[3, 2]]
    forcing = forcing.assign_coords(time=shuffled)
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        _resolve_fft(
            FFTConvention(sample_interval_seconds=86_400.0), forcing.time
        )


def test_forcing_schema_and_missing_values_fail_early() -> None:
    with pytest.raises(ValueError, match="exactly M_Ek_x"):
        _normalise_forcing(forcing_dataset().drop_vars("T_N"))

    forcing = forcing_dataset()
    forcing["M_Ek_x"] = forcing.M_Ek_x.where(forcing.time.dt.day != 3)
    with pytest.raises(ValueError, match="cannot contain missing"):
        _normalise_forcing(forcing)
