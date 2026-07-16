"""Time alignment and Fourier conventions for global model forcing."""

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


def _unit_text(array: xr.DataArray) -> str:
    return " ".join(str(array.attrs.get("units", "")).lower().split())


def _normalise_ekman_transport(array: xr.DataArray, name: str) -> xr.DataArray:
    if set(array.dims) != {"time", "latitude", "longitude"}:
        raise ValueError(f"{name} must have time, latitude, and longitude dimensions")
    if _unit_text(array) not in _EKMAN_TRANSPORT_UNITS:
        raise ValueError(f"{name} units must be m2 s-1")
    result = array.transpose("time", "latitude", "longitude").astype(float)
    result.name = name
    result.attrs = dict(array.attrs)
    result.attrs["units"] = "m2 s-1"
    return result


def _normalise_transport(array: xr.DataArray, name: str) -> xr.DataArray:
    if array.dims != ("time",):
        raise ValueError(f"{name} must have only a time dimension")
    units = _unit_text(array)
    if units in _TRANSPORT_SV_UNITS:
        result = array.astype(float) * 1e6
    elif units in _TRANSPORT_M3_UNITS:
        result = array.astype(float)
    else:
        raise ValueError(f"{name} units must be m3 s-1 or Sv")
    result.name = name
    result.attrs = dict(array.attrs)
    result.attrs["units"] = "m3 s-1"
    return result


def _validate_grid(M_ek_x: xr.DataArray, M_ek_y: xr.DataArray) -> None:
    if not M_ek_x.latitude.equals(M_ek_y.latitude) or not M_ek_x.longitude.equals(
        M_ek_y.longitude
    ):
        raise ValueError("M_ek_x and M_ek_y must use the same spatial grid")
    for coordinate in ("latitude", "longitude"):
        values = np.asarray(M_ek_x[coordinate], dtype=float)
        if values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Ekman-transport {coordinate} must contain at least two finite values"
            )
        if not np.all(np.diff(values) > 0):
            raise ValueError(
                f"Ekman-transport {coordinate} must be strictly increasing"
            )


def _sample_interval_seconds(
    time: xr.DataArray, supplied: float | None
) -> float:
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
    pad_width = [(0, 0)] * np.ndim(data)
    pad_width[0] = (pad, pad)
    if isinstance(data, dsa.Array):
        padded = dsa.pad(data, pad_width, mode="reflect").rechunk({0: -1})
        return dsa.fft.rfft(padded, n=n_fft, axis=0)
    return np.fft.rfft(np.pad(np.asarray(data), pad_width, mode="reflect"), n=n_fft, axis=0)


@dataclass(frozen=True, slots=True)
class GlobalForcing:
    """Aligned forcing anomalies with one immutable real-FFT convention."""

    _time_domain: xr.Dataset
    _spectral: xr.Dataset
    _sample_interval_seconds: float
    _n_fft: int
    _padding_samples: int
    _padding_mode: str

    @classmethod
    def from_time_series(
        cls,
        *,
        M_ek_x: xr.DataArray,
        M_ek_y: xr.DataArray,
        northern_transport: xr.DataArray,
        southern_transport: xr.DataArray,
        sample_interval_seconds: float | None = None,
        padding_samples: int | None = None,
        n_fft: int | None = None,
        padding_mode: Literal["reflect"] = "reflect",
    ) -> GlobalForcing:
        """Build anomaly forcing from Ekman and external total transports.

        Wind-stress conversion, equatorial regularization, and coastal tapering
        are intentionally upstream choices. The model derives pumping and
        section transports from these same vector-transport anomalies. Every
        input time mean is removed before applying the shared transform.
        """

        if padding_mode != "reflect":
            raise ValueError("only reflect padding is currently supported")
        M_ek_x = _normalise_ekman_transport(M_ek_x, "M_ek_x")
        M_ek_y = _normalise_ekman_transport(M_ek_y, "M_ek_y")
        northern_transport = _normalise_transport(
            northern_transport, "northern_transport"
        )
        southern_transport = _normalise_transport(
            southern_transport, "southern_transport"
        )
        M_ek_x, M_ek_y, northern_transport, southern_transport = xr.align(
            M_ek_x,
            M_ek_y,
            northern_transport,
            southern_transport,
            join="exact",
            copy=False,
        )
        _validate_grid(M_ek_x, M_ek_y)
        interval = _sample_interval_seconds(M_ek_x.time, sample_interval_seconds)
        time_domain = xr.Dataset(
            {
                "M_ek_x": M_ek_x,
                "M_ek_y": M_ek_y,
                "northern_transport": northern_transport,
                "southern_transport": southern_transport,
            }
        )
        missing = compute(
            *(array.isnull().any() for array in time_domain.data_vars.values())
        )
        for name, contains_missing in zip(time_domain.data_vars, missing, strict=True):
            if bool(contains_missing):
                raise ValueError(f"forcing input {name!r} cannot contain missing values")
        time_domain = time_domain - time_domain.mean("time")
        time_domain.M_ek_x.attrs["units"] = "m2 s-1"
        time_domain.M_ek_y.attrs["units"] = "m2 s-1"
        time_domain.northern_transport.attrs["units"] = "m3 s-1"
        time_domain.southern_transport.attrs["units"] = "m3 s-1"
        time_domain.attrs["time_mean_removed"] = True

        sample_count = time_domain.sizes["time"]
        if sample_count < 2:
            raise ValueError("forcing requires at least two time samples")
        if padding_samples is None:
            padding_samples = sample_count - 1
        if isinstance(padding_samples, bool) or not isinstance(
            padding_samples, (int, np.integer)
        ):
            raise ValueError("padding_samples must be an integer")
        padding_samples = int(padding_samples)
        if padding_samples < 0 or padding_samples > sample_count - 1:
            raise ValueError(
                "reflect padding_samples must lie between zero and n_time - 1"
            )
        padded_count = sample_count + 2 * padding_samples
        if n_fft is None:
            n_fft = next_fast_len(padded_count, real=True)
        if isinstance(n_fft, bool) or not isinstance(n_fft, (int, np.integer)):
            raise ValueError("n_fft must be an integer")
        n_fft = int(n_fft)
        if n_fft < padded_count:
            raise ValueError("n_fft cannot be shorter than the padded record")

        omega = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=interval)
        spectra: dict[str, xr.DataArray] = {}
        for name, array in time_domain.data_vars.items():
            ordered = array.transpose("time", ...)
            data = _fft_data(
                ordered.data,
                n_fft=n_fft,
                pad=padding_samples,
            )
            coordinates = {
                "omega": omega,
                **{
                    dimension: ordered.coords[dimension]
                    for dimension in ordered.dims
                    if dimension != "time"
                },
            }
            spectrum = xr.DataArray(
                data,
                dims=("omega", *ordered.dims[1:]),
                coords=coordinates,
                name=name,
                attrs={**array.attrs, "transform": "numpy rfft"},
            )
            spectra[name] = spectrum.where(spectrum.omega != 0.0, 0.0)
        spectral = xr.Dataset(spectra)
        spectral.omega.attrs.update(
            units="rad s-1", long_name="non-negative angular frequency"
        )
        spectral.attrs.update(
            n_fft=n_fft,
            sample_interval_seconds=interval,
            padding_samples=padding_samples,
            padding_mode=padding_mode,
            zero_frequency="set to zero for anomaly forcing",
        )
        return cls(
            time_domain,
            spectral,
            interval,
            n_fft,
            padding_samples,
            padding_mode,
        )

    @property
    def time_domain(self) -> xr.Dataset:
        """Aligned time-domain anomalies in SI units."""

        return self._time_domain

    @property
    def spectral(self) -> xr.Dataset:
        """The matching non-negative-frequency spectra."""

        return self._spectral

    @property
    def omega(self) -> xr.DataArray:
        return self._spectral.omega

    @property
    def sample_interval_seconds(self) -> float:
        return self._sample_interval_seconds

    @property
    def n_fft(self) -> int:
        return self._n_fft

    @property
    def padding_samples(self) -> int:
        return self._padding_samples

    def transform(self, array: xr.DataArray) -> xr.DataArray:
        """Apply this forcing object's FFT convention to another time series."""

        if "time" not in array.dims or not array.time.equals(self._time_domain.time):
            raise ValueError("array must use the forcing object's time coordinate")
        ordered = array.transpose("time", ...)
        data = _fft_data(
            ordered.data,
            n_fft=self._n_fft,
            pad=self._padding_samples,
        )
        result = xr.DataArray(
            data,
            dims=("omega", *ordered.dims[1:]),
            coords={
                "omega": self.omega,
                **{
                    dimension: ordered.coords[dimension]
                    for dimension in ordered.dims
                    if dimension != "time"
                },
            },
            name=array.name,
            attrs={**array.attrs, "transform": "numpy rfft"},
        )
        return result.where(result.omega != 0.0, 0.0)

    def inverse_transform(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Apply the matching inverse transform and crop to the source record."""

        if "omega" not in spectrum.dims or not spectrum.omega.equals(self.omega):
            raise ValueError("spectrum must use the forcing object's omega coordinate")
        ordered = spectrum.transpose("omega", ...)
        data = ordered.data
        if isinstance(data, dsa.Array):
            transformed = dsa.fft.irfft(
                data.rechunk({0: -1}), n=self._n_fft, axis=0
            )
        else:
            transformed = np.fft.irfft(np.asarray(data), n=self._n_fft, axis=0)
        start = self._padding_samples
        stop = start + self._time_domain.sizes["time"]
        cropped = transformed[start:stop]
        return xr.DataArray(
            cropped,
            dims=("time", *ordered.dims[1:]),
            coords={
                "time": self._time_domain.time,
                **{
                    dimension: ordered.coords[dimension]
                    for dimension in ordered.dims
                    if dimension != "omega"
                },
            },
            name=spectrum.name,
            attrs={**spectrum.attrs, "transform": "numpy irfft, cropped"},
        )
