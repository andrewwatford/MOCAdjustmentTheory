"""Stateless Fourier transforms for model forcing and output arrays."""

from collections.abc import Mapping
from typing import Literal

import numpy as np
import xarray as xr

FFTNorm = Literal["backward", "ortho", "forward"]

_METADATA_KEY = "_moc_adjustment_fourier"
_VALID_NORMS = {"backward", "ortho", "forward"}


def forward_transform(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
    omega_dim: str = "omega",
    pad_factor: int = 2,
    norm: FFTNorm = "backward",
    require_zero_mean: bool = True,
    zero_mean_rtol: float = 1e-7,
    sample_spacing_seconds: float | None = None,
) -> xr.DataArray:
    """Transform real, uniformly sampled data to angular-frequency space.

    The transform uses `numpy.fft.rfft`, causal right-zero-padding to the
    smallest odd length at least ``pad_factor`` times the input length, and
    angular frequency in rad/s. The default padded length is therefore
    ``2N + 1``; the odd length avoids a self-conjugate Nyquist bin when the model
    applies complex travel-time phases. No detrending or window is applied. By
    default, each time series must have a mean no larger than
    ``zero_mean_rtol`` times the field-wide maximum absolute value. Accepted
    roundoff-level means are set exactly to zero in frequency space.
    ``sample_spacing_seconds`` may explicitly supply the physical sample
    interval for monotonically increasing but nonuniform calendar labels. For
    monthly data, this project uses ``365.25 / 12`` days per sample. Metadata
    needed for an exact, stateless inverse is attached to the result.
    """
    _validate_contract(time_dim, omega_dim, pad_factor, norm)
    if not isinstance(require_zero_mean, bool):
        raise TypeError("require_zero_mean must be a bool")
    _validate_tolerance(zero_mean_rtol, "zero_mean_rtol")
    _validate_sample_spacing(sample_spacing_seconds)
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    if time_dim not in data.dims:
        raise ValueError(f"data must have a {time_dim!r} dimension")
    if omega_dim in data.dims:
        raise ValueError(f"data already has an {omega_dim!r} dimension")

    time_ns, dt_seconds = _validated_time(
        data, time_dim, sample_spacing_seconds
    )
    values = np.asarray(data.data)
    _validate_real_finite(values, "data")
    axis = data.get_axis_num(time_dim)
    if require_zero_mean:
        means = np.abs(np.mean(values, axis=axis))
        field_scale = np.max(np.abs(values))
        if np.any(means > zero_mean_rtol * field_scale):
            raise ValueError(
                "each time series must be zero mean; pass "
                "require_zero_mean=False to retain a nonzero mean"
            )

    original_length = data.sizes[time_dim]
    padded_length = _padded_length(original_length, pad_factor)
    transformed = np.fft.rfft(values, n=padded_length, axis=axis, norm=norm)
    if require_zero_mean:
        # The anomaly contract is exact in spectral space even when float32
        # storage leaves a small residual after demeaning.
        index = [slice(None)] * transformed.ndim
        index[axis] = 0
        transformed[tuple(index)] = 0.0
    omega = 2.0 * np.pi * np.fft.rfftfreq(padded_length, d=dt_seconds)
    dims = tuple(omega_dim if dim == time_dim else dim for dim in data.dims)
    coords = {
        name: coord
        for name, coord in data.coords.items()
        if time_dim not in coord.dims
    }
    coords[omega_dim] = xr.DataArray(
        omega,
        dims=(omega_dim,),
        attrs={"long_name": "angular frequency", "units": "rad s-1"},
    )
    attrs = dict(data.attrs)
    attrs[_METADATA_KEY] = {
        "time_dim": time_dim,
        "omega_dim": omega_dim,
        "original_time_ns": tuple(int(value) for value in time_ns),
        "time_attrs": dict(data[time_dim].attrs),
        "original_length": original_length,
        "padded_length": padded_length,
        "pad_factor": pad_factor,
        "dt_seconds": dt_seconds,
        "norm": norm,
        "sample_spacing_seconds": sample_spacing_seconds,
    }
    return xr.DataArray(
        transformed,
        dims=dims,
        coords=coords,
        attrs=attrs,
        name=data.name,
    )


def inverse_transform(
    spectrum: xr.DataArray,
    *,
    time_dim: str = "time",
    omega_dim: str = "omega",
    pad_factor: int = 2,
    norm: FFTNorm = "backward",
    imaginary_rtol: float = 1e-12,
    sample_spacing_seconds: float | None = None,
) -> xr.DataArray:
    """Invert a spectrum made by `forward_transform` onto its time grid.

    The inverse validates the explicit transform contract against spectral
    metadata, applies `numpy.fft.irfft`, and removes the causal right
    padding. An imaginary DC component, which cannot represent a real time
    series, is rejected above ``imaginary_rtol`` relative to the largest
    spectral magnitude in the corresponding series. The odd padded length has
    no self-conjugate Nyquist component. When an explicit physical sample
    interval was used, the original calendar labels are still restored.
    """
    _validate_contract(time_dim, omega_dim, pad_factor, norm)
    _validate_tolerance(imaginary_rtol, "imaginary_rtol")
    _validate_sample_spacing(sample_spacing_seconds)
    if not isinstance(spectrum, xr.DataArray):
        raise TypeError("spectrum must be an xarray.DataArray")
    if omega_dim not in spectrum.dims:
        raise ValueError(f"spectrum must have an {omega_dim!r} dimension")
    if time_dim in spectrum.dims:
        raise ValueError(f"spectrum already has a {time_dim!r} dimension")

    metadata = spectrum.attrs.get(_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        raise ValueError("spectrum is missing forward-transform metadata")
    expected_contract = {
        "time_dim": time_dim,
        "omega_dim": omega_dim,
        "pad_factor": pad_factor,
        "norm": norm,
        "sample_spacing_seconds": sample_spacing_seconds,
    }
    for key, expected in expected_contract.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{key}={expected!r} does not match the forward transform "
                f"value {metadata.get(key)!r}"
            )

    original_length = _metadata_int(metadata, "original_length")
    padded_length = _metadata_int(metadata, "padded_length")
    if padded_length != _padded_length(original_length, pad_factor):
        raise ValueError("inconsistent transform lengths in spectral metadata")
    expected_size = padded_length // 2 + 1
    if spectrum.sizes[omega_dim] != expected_size:
        raise ValueError("spectrum length does not match forward-transform metadata")
    dt_seconds = metadata.get("dt_seconds")
    if (
        not isinstance(dt_seconds, (int, float))
        or not np.isfinite(dt_seconds)
        or dt_seconds <= 0
    ):
        raise ValueError("invalid time step in spectral metadata")
    expected_omega = 2.0 * np.pi * np.fft.rfftfreq(
        padded_length, d=float(dt_seconds)
    )
    if omega_dim not in spectrum.coords or not np.allclose(
        spectrum[omega_dim].values, expected_omega, rtol=1e-13, atol=0.0
    ):
        raise ValueError("omega coordinate does not match forward-transform metadata")

    values = np.asarray(spectrum.data)
    _validate_numeric_finite(values, "spectrum")
    axis = spectrum.get_axis_num(omega_dim)
    scale = np.max(np.abs(values), axis=axis)
    dc_imaginary = np.abs(np.take(values, 0, axis=axis).imag)
    inconsistent = dc_imaginary > imaginary_rtol * scale
    if np.any(inconsistent):
        raise ValueError("spectrum is inconsistent with a real time series")

    restored = np.fft.irfft(values, n=padded_length, axis=axis, norm=norm)
    restored = np.take(restored, np.arange(original_length), axis=axis)
    dims = tuple(time_dim if dim == omega_dim else dim for dim in spectrum.dims)
    coords = {
        name: coord
        for name, coord in spectrum.coords.items()
        if omega_dim not in coord.dims
    }
    time_values = metadata.get("original_time_ns")
    if (
        not isinstance(time_values, (tuple, list))
        or len(time_values) != original_length
    ):
        raise ValueError("invalid original time coordinate in spectral metadata")
    time_attrs = metadata.get("time_attrs", {})
    if not isinstance(time_attrs, Mapping):
        raise ValueError("invalid time-coordinate attributes in spectral metadata")
    coords[time_dim] = xr.DataArray(
        np.asarray(time_values, dtype="datetime64[ns]"),
        dims=(time_dim,),
        attrs=dict(time_attrs),
    )
    attrs = {
        key: value for key, value in spectrum.attrs.items() if key != _METADATA_KEY
    }
    return xr.DataArray(
        restored,
        dims=dims,
        coords=coords,
        attrs=attrs,
        name=spectrum.name,
    )


def _validate_contract(
    time_dim: str, omega_dim: str, pad_factor: int, norm: FFTNorm
) -> None:
    """Validate dimension names, padding, and NumPy normalization."""
    if not isinstance(time_dim, str) or not time_dim:
        raise TypeError("time_dim must be a nonempty string")
    if not isinstance(omega_dim, str) or not omega_dim:
        raise TypeError("omega_dim must be a nonempty string")
    if time_dim == omega_dim:
        raise ValueError("time_dim and omega_dim must differ")
    if isinstance(pad_factor, bool) or not isinstance(pad_factor, int):
        raise TypeError("pad_factor must be an integer")
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least 1")
    if norm not in _VALID_NORMS:
        raise ValueError(f"norm must be one of {sorted(_VALID_NORMS)}")


def _validate_tolerance(value: float, name: str) -> None:
    """Require a finite, nonnegative numerical tolerance."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _padded_length(original_length: int, pad_factor: int) -> int:
    """Return the smallest odd FFT length meeting the requested padding."""
    minimum = pad_factor * original_length
    return minimum if minimum % 2 else minimum + 1


def _validate_sample_spacing(value: float | None) -> None:
    """Validate an optional physical sample interval in seconds."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("sample_spacing_seconds must be a real number or None")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("sample_spacing_seconds must be finite and positive")


def _validated_time(
    data: xr.DataArray,
    time_dim: str,
    sample_spacing_seconds: float | None,
) -> tuple[np.ndarray, float]:
    """Return timestamps and the validated or explicitly supplied interval."""
    if time_dim not in data.coords or data[time_dim].dims != (time_dim,):
        raise ValueError(f"{time_dim!r} must have a one-dimensional coordinate")
    time = np.asarray(data[time_dim].values)
    if not np.issubdtype(time.dtype, np.datetime64):
        raise TypeError(f"{time_dim!r} coordinate must contain datetime64 values")
    if len(time) < 2:
        raise ValueError("at least two time samples are required")
    if np.any(np.isnat(time)):
        raise ValueError(f"{time_dim!r} coordinate must not contain NaT")
    time_ns = time.astype("datetime64[ns]").astype(np.int64)
    steps = np.diff(time_ns)
    if np.any(steps <= 0):
        raise ValueError(f"{time_dim!r} coordinate must be strictly increasing")
    if sample_spacing_seconds is None and np.any(steps != steps[0]):
        raise ValueError(f"{time_dim!r} coordinate must be uniformly spaced")
    dt_seconds = (
        float(steps[0]) / 1e9
        if sample_spacing_seconds is None
        else float(sample_spacing_seconds)
    )
    return time_ns, dt_seconds


def _validate_numeric_finite(values: np.ndarray, name: str) -> None:
    """Require numeric values without NaNs or infinities."""
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must not contain NaN or infinite values")


def _validate_real_finite(values: np.ndarray, name: str) -> None:
    """Require finite real-valued input for a real Fourier transform."""
    _validate_numeric_finite(values, name)
    if np.iscomplexobj(values):
        raise TypeError(f"{name} must contain real values")


def _metadata_int(metadata: Mapping[object, object], key: str) -> int:
    """Read a positive integer from stored transform metadata."""
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"invalid {key} in spectral metadata")
    return value
