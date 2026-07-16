"""Forcing validation and shared real-Fourier conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import dask.array as dsa
from dask import compute
import numpy as np
import xarray as xr
from scipy.fft import next_fast_len


_EKMAN_TRANSPORT_UNITS = {
    "m2/s",
    "m^2/s",
    "m2 s-1",
    "m^2 s^-1",
    "m2 s**-1",
}
_TRANSPORT_M3_UNITS = {
    "m3/s",
    "m^3/s",
    "m3 s-1",
    "m^3 s^-1",
    "m3 s**-1",
}
_TRANSPORT_SV_UNITS = {"sv", "sverdrup", "sverdrups"}
_FORCING_VARIABLES = ("M_Ek_x", "M_Ek_y", "T_N")


def _unit_text(array: xr.DataArray) -> str:
    """Return a case- and whitespace-normalised unit string."""

    return " ".join(str(array.attrs.get("units", "")).lower().split())


def _normalise_ekman_transport(array: xr.DataArray, name: str) -> xr.DataArray:
    """Validate one vector-transport component and standardise its units."""

    if set(array.dims) != {"time", "latitude", "longitude"}:
        raise ValueError(f"{name} must have time, latitude, and longitude dimensions")
    if _unit_text(array) not in _EKMAN_TRANSPORT_UNITS:
        raise ValueError(f"{name} units must be m2 s-1")
    result = array.transpose("time", "latitude", "longitude").astype(float)
    result.attrs = {**array.attrs, "units": "m2 s-1"}
    return result.rename(name)


def _normalise_transport(array: xr.DataArray, name: str) -> xr.DataArray:
    """Convert a one-dimensional volume transport to cubic metres per second."""

    if array.dims != ("time",):
        raise ValueError(f"{name} must have only a time dimension")
    units = _unit_text(array)
    if units in _TRANSPORT_SV_UNITS:
        result = array.astype(float) * 1e6
    elif units in _TRANSPORT_M3_UNITS:
        result = array.astype(float)
    else:
        raise ValueError(f"{name} units must be m3 s-1 or Sv")
    result.attrs = {**array.attrs, "units": "m3 s-1"}
    return result.rename(name)


def _normalise_forcing(dataset: xr.Dataset) -> xr.Dataset:
    """Validate the three-variable forcing dataset and return SI anomalies."""

    if not isinstance(dataset, xr.Dataset):
        raise TypeError("forcing must be an xarray.Dataset")
    missing = sorted(set(_FORCING_VARIABLES) - set(dataset.data_vars))
    extra = sorted(set(dataset.data_vars) - set(_FORCING_VARIABLES))
    if missing or extra:
        raise ValueError(
            "forcing must contain exactly M_Ek_x, M_Ek_y, and T_N; "
            f"missing={missing}, extra={extra}"
        )
    result = xr.Dataset(
        {
            "M_Ek_x": _normalise_ekman_transport(dataset.M_Ek_x, "M_Ek_x"),
            "M_Ek_y": _normalise_ekman_transport(dataset.M_Ek_y, "M_Ek_y"),
            "T_N": _normalise_transport(dataset.T_N, "T_N"),
        },
        attrs=dict(dataset.attrs),
    )
    for coordinate in ("latitude", "longitude"):
        values = np.asarray(result[coordinate], dtype=float)
        if values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Ekman-transport {coordinate} must contain at least two finite values"
            )
        if not np.all(np.diff(values) > 0):
            raise ValueError(f"Ekman-transport {coordinate} must be strictly increasing")
    contains_missing = compute(
        *(array.isnull().any() for array in result.data_vars.values())
    )
    for name, missing_values in zip(result.data_vars, contains_missing, strict=True):
        if bool(missing_values):
            raise ValueError(f"forcing input {name!r} cannot contain missing values")
    result = result - result.mean("time")
    result.M_Ek_x.attrs["units"] = "m2 s-1"
    result.M_Ek_y.attrs["units"] = "m2 s-1"
    result.T_N.attrs["units"] = "m3 s-1"
    result.attrs["time_mean_removed"] = True
    return result


def _sample_interval_seconds(time: xr.DataArray, supplied: float | None) -> float:
    """Validate time ordering and determine the transform sample interval."""

    values = np.asarray(time)
    if np.issubdtype(values.dtype, np.datetime64):
        difference = np.diff(values.astype("datetime64[ns]")).astype("timedelta64[ns]")
        seconds = difference.astype(np.float64) / 1e9
    else:
        units = " ".join(str(time.attrs.get("units", "")).lower().split())
        if units not in {"s", "second", "seconds"}:
            raise ValueError(
                "numeric time requires second units or sample_interval_seconds"
            )
        seconds = np.diff(values.astype(float))
    if seconds.size == 0 or not np.all(np.isfinite(seconds)):
        raise ValueError("forcing requires at least two finite time samples")
    if np.any(seconds <= 0):
        raise ValueError("time must be unique and strictly increasing")
    if supplied is not None:
        interval = float(supplied)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("sample_interval_seconds must be positive and finite")
        return interval
    interval = float(np.median(seconds))
    if not np.allclose(seconds, interval, rtol=1e-8, atol=1e-6):
        raise ValueError(
            "time must be uniformly sampled; pass sample_interval_seconds for "
            "a calendar-month record"
        )
    return interval


def _fft_data(data: object, *, n_fft: int, pad: int) -> object:
    """Reflect-pad one time-leading array and apply an rFFT along time."""

    pad_width = [(0, 0)] * np.ndim(data)
    pad_width[0] = (pad, pad)
    if isinstance(data, dsa.Array):
        padded = dsa.pad(data, pad_width, mode="reflect").rechunk({0: -1})
        return dsa.fft.rfft(padded, n=n_fft, axis=0)
    padded = np.pad(np.asarray(data), pad_width, mode="reflect")
    return np.fft.rfft(padded, n=n_fft, axis=0)


@dataclass(frozen=True, slots=True)
class FFTConvention:
    """User-selected sampling, padding, and real-FFT convention."""

    sample_interval_seconds: float | None = None
    padding_samples: int | None = None
    n_fft: int | None = None
    padding_mode: Literal["reflect"] = "reflect"


def _resolve_fft(convention: FFTConvention, time: xr.DataArray) -> _ResolvedFFT:
    """Resolve a declarative convention against one forcing time coordinate."""

    if convention.padding_mode != "reflect":
        raise ValueError("only reflect padding is currently supported")
    interval = _sample_interval_seconds(time, convention.sample_interval_seconds)
    sample_count = time.size
    padding = (
        sample_count - 1
        if convention.padding_samples is None
        else convention.padding_samples
    )
    if isinstance(padding, bool) or not isinstance(padding, (int, np.integer)):
        raise ValueError("padding_samples must be an integer")
    padding = int(padding)
    if padding < 0 or padding > sample_count - 1:
        raise ValueError(
            "reflect padding_samples must lie between zero and n_time - 1"
        )
    padded_count = sample_count + 2 * padding
    n_fft = (
        next_fast_len(padded_count, real=True)
        if convention.n_fft is None
        else convention.n_fft
    )
    if isinstance(n_fft, bool) or not isinstance(n_fft, (int, np.integer)):
        raise ValueError("n_fft must be an integer")
    n_fft = int(n_fft)
    if n_fft < padded_count:
        raise ValueError("n_fft cannot be shorter than the padded record")
    return _ResolvedFFT(
        time=time,
        sample_interval_seconds=interval,
        padding_samples=padding,
        n_fft=n_fft,
        padding_mode=convention.padding_mode,
    )


@dataclass(frozen=True, slots=True)
class _ResolvedFFT:
    """Concrete transform associated with one forcing time coordinate."""

    time: xr.DataArray
    sample_interval_seconds: float
    padding_samples: int
    n_fft: int
    padding_mode: str

    def transform(self, array: xr.DataArray) -> xr.DataArray:
        """Transform one array that uses the resolved time coordinate."""

        if "time" not in array.dims or not array.time.equals(self.time):
            raise ValueError("array must use the forcing time coordinate")
        ordered = array.transpose("time", ...)
        omega = 2.0 * np.pi * np.fft.rfftfreq(
            self.n_fft, d=self.sample_interval_seconds
        )
        spectrum = xr.DataArray(
            _fft_data(
                ordered.data,
                n_fft=self.n_fft,
                pad=self.padding_samples,
            ),
            dims=("omega", *ordered.dims[1:]),
            coords={
                "omega": omega,
                **{
                    dimension: ordered.coords[dimension]
                    for dimension in ordered.dims
                    if dimension != "time"
                },
            },
            name=array.name,
            attrs={**array.attrs, "transform": "numpy rfft"},
        )
        spectrum.omega.attrs.update(
            units="rad s-1", long_name="non-negative angular frequency"
        )
        return spectrum.where(spectrum.omega != 0.0, 0.0)

    def transform_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Transform every time-dependent variable with one shared convention."""

        spectral = xr.Dataset(
            {name: self.transform(array) for name, array in dataset.data_vars.items()}
        )
        spectral.attrs.update(
            n_fft=self.n_fft,
            sample_interval_seconds=self.sample_interval_seconds,
            padding_samples=self.padding_samples,
            padding_mode=self.padding_mode,
            zero_frequency="set to zero for anomaly forcing",
        )
        return spectral

    def inverse_transform(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Apply the matching inverse transform and crop to the source record."""

        expected_omega = 2.0 * np.pi * np.fft.rfftfreq(
            self.n_fft, d=self.sample_interval_seconds
        )
        if "omega" not in spectrum.dims or not np.array_equal(
            np.asarray(spectrum.omega), expected_omega
        ):
            raise ValueError("spectrum must use the resolved omega coordinate")
        ordered = spectrum.transpose("omega", ...)
        if isinstance(ordered.data, dsa.Array):
            transformed = dsa.fft.irfft(
                ordered.data.rechunk({0: -1}), n=self.n_fft, axis=0
            )
        else:
            transformed = np.fft.irfft(
                np.asarray(ordered.data), n=self.n_fft, axis=0
            )
        start = self.padding_samples
        cropped = transformed[start : start + self.time.size]
        return xr.DataArray(
            cropped,
            dims=("time", *ordered.dims[1:]),
            coords={
                "time": self.time,
                **{
                    dimension: ordered.coords[dimension]
                    for dimension in ordered.dims
                    if dimension != "omega"
                },
            },
            name=spectrum.name,
            attrs={**spectrum.attrs, "transform": "numpy irfft, cropped"},
        )
