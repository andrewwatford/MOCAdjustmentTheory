"""Stateless Fourier transforms for model forcing and output arrays."""

from collections.abc import Mapping

import dask.array as da
import numpy as np
import xarray as xr
from scipy import fft
from scipy.signal import butter, sosfiltfilt

_METADATA_KEY = "_moc_adjustment_fourier"
_OMEGA_DIM = "omega"


def forward_transform(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
    pad_length: int = 0,
    require_zero_mean: bool = True,
    zero_mean_rtol: float = 1e-7,
    sample_spacing_seconds: float | None = None,
) -> xr.DataArray:
    """Transform real, uniformly sampled data to angular-frequency space.

    The transform uses a real FFT, appending at least ``pad_length`` zero
    samples on the right. One additional zero is appended when needed to make
    the full transform length odd, avoiding a self-conjugate Nyquist bin when
    the model applies complex travel-time phases. Angular frequency is returned
    in rad/s. No detrending or window is applied. By default, each time series
    must have a mean no larger than
    ``zero_mean_rtol`` times the field-wide maximum absolute value. Accepted
    roundoff-level means are set exactly to zero in frequency space.
    ``sample_spacing_seconds`` may explicitly supply the physical sample
    interval for monotonically increasing but nonuniform calendar labels. For
    monthly data, this project uses ``365.25 / 12`` days per sample. Metadata
    needed for an exact, stateless inverse is attached to the result.
    """
    _validate_forward_contract(time_dim, pad_length)
    if not isinstance(require_zero_mean, bool):
        raise TypeError("require_zero_mean must be a bool")
    _validate_tolerance(zero_mean_rtol, "zero_mean_rtol")
    _validate_sample_spacing(sample_spacing_seconds)
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    if time_dim not in data.dims:
        raise ValueError(f"data must have a {time_dim!r} dimension")
    if _OMEGA_DIM in data.dims:
        raise ValueError(f"data already has an {_OMEGA_DIM!r} dimension")

    time_ns, dt_seconds = _validated_time(
        data, time_dim, sample_spacing_seconds
    )
    values = data.data
    axis = data.get_axis_num(time_dim)
    if isinstance(values, da.Array):
        _validate_real_dtype(values, "data")
        values = values.rechunk({axis: -1})
        values = values.map_blocks(
            _validated_real_finite_block,
            name="validate-real-finite",
            dtype=values.dtype,
        )
        if require_zero_mean:
            field_scale = da.max(da.absolute(values))
            indices = tuple(range(values.ndim))
            values = da.blockwise(
                _validated_zero_mean_block,
                indices,
                values,
                indices,
                field_scale,
                (),
                axis=axis,
                zero_mean_rtol=zero_mean_rtol,
                dtype=values.dtype,
                name="validate-zero-mean",
            )
    else:
        values = np.asarray(values)
        _validate_real_finite(values, "data")
        if require_zero_mean:
            _validate_zero_mean(values, axis, zero_mean_rtol)

    original_length = data.sizes[time_dim]
    padded_length = _padded_length(original_length, pad_length)
    if isinstance(values, da.Array):
        transformed = da.fft.rfft(values, n=padded_length, axis=axis)
    else:
        transformed = fft.rfft(
            values, n=padded_length, axis=axis, workers=-1
        )
    if require_zero_mean:
        # The anomaly contract is exact in spectral space even when float32
        # storage leaves a small residual after demeaning.
        index = [slice(None)] * transformed.ndim
        index[axis] = 0
        if isinstance(transformed, da.Array):
            transformed = transformed.map_blocks(
                _zero_dc_block,
                axis=axis,
                dtype=transformed.dtype,
            )
        else:
            transformed[tuple(index)] = 0.0
    omega = 2.0 * np.pi * fft.rfftfreq(padded_length, d=dt_seconds)
    dims = tuple(_OMEGA_DIM if dim == time_dim else dim for dim in data.dims)
    coords = {
        name: coord
        for name, coord in data.coords.items()
        if time_dim not in coord.dims
    }
    coords[_OMEGA_DIM] = xr.DataArray(
        omega,
        dims=(_OMEGA_DIM,),
        attrs={"long_name": "angular frequency", "units": "rad s-1"},
    )
    attrs = dict(data.attrs)
    attrs[_METADATA_KEY] = {
        "time_dim": time_dim,
        "original_time_ns": tuple(int(value) for value in time_ns),
        "time_attrs": dict(data[time_dim].attrs),
        "original_length": original_length,
        "padded_length": padded_length,
        "pad_length": padded_length - original_length,
        "dt_seconds": dt_seconds,
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
    imaginary_rtol: float = 1e-12,
) -> xr.DataArray:
    """Invert a spectrum made by `forward_transform` onto its time grid.

    The inverse reads the dimension, sampling, and padding contract stored by
    `forward_transform`, applies a real inverse FFT, and removes the causal
    right padding. An imaginary DC component, which cannot represent a real time
    series, is rejected above ``imaginary_rtol`` relative to the largest
    spectral magnitude in the corresponding series. The odd padded length has
    no self-conjugate Nyquist component. When an explicit physical sample
    interval was used, the original calendar labels are still restored.
    """
    _validate_tolerance(imaginary_rtol, "imaginary_rtol")
    if not isinstance(spectrum, xr.DataArray):
        raise TypeError("spectrum must be an xarray.DataArray")
    if _OMEGA_DIM not in spectrum.dims:
        raise ValueError(f"spectrum must have an {_OMEGA_DIM!r} dimension")

    metadata = spectrum.attrs.get(_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        raise ValueError("spectrum is missing forward-transform metadata")
    time_dim = metadata.get("time_dim")
    if not isinstance(time_dim, str) or not time_dim:
        raise ValueError("invalid time dimension in spectral metadata")
    if time_dim in spectrum.dims:
        raise ValueError(f"spectrum already has a {time_dim!r} dimension")

    original_length = _metadata_int(metadata, "original_length")
    padded_length = _metadata_int(metadata, "padded_length")
    pad_length = _metadata_int(metadata, "pad_length", minimum=0)
    if padded_length != original_length + pad_length or padded_length % 2 == 0:
        raise ValueError("inconsistent transform lengths in spectral metadata")
    expected_size = padded_length // 2 + 1
    if spectrum.sizes[_OMEGA_DIM] != expected_size:
        raise ValueError("spectrum length does not match forward-transform metadata")
    dt_seconds = metadata.get("dt_seconds")
    if (
        not isinstance(dt_seconds, (int, float))
        or not np.isfinite(dt_seconds)
        or dt_seconds <= 0
    ):
        raise ValueError("invalid time step in spectral metadata")
    expected_omega = 2.0 * np.pi * fft.rfftfreq(
        padded_length, d=float(dt_seconds)
    )
    if _OMEGA_DIM not in spectrum.coords or not np.allclose(
        spectrum[_OMEGA_DIM].values, expected_omega, rtol=1e-13, atol=0.0
    ):
        raise ValueError("omega coordinate does not match forward-transform metadata")

    values = spectrum.data
    axis = spectrum.get_axis_num(_OMEGA_DIM)
    if isinstance(values, da.Array):
        _validate_numeric_dtype(values, "spectrum")
        values = values.rechunk({axis: -1}).map_blocks(
            _validated_spectrum_block,
            axis=axis,
            imaginary_rtol=imaginary_rtol,
            dtype=values.dtype,
        )
        restored = da.fft.irfft(values, n=padded_length, axis=axis)
        index = [slice(None)] * restored.ndim
        index[axis] = slice(0, original_length)
        restored = restored[tuple(index)]
    else:
        values = np.asarray(values)
        _validate_numeric_finite(values, "spectrum")
        scale = np.max(np.abs(values), axis=axis)
        dc_imaginary = np.abs(np.take(values, 0, axis=axis).imag)
        inconsistent = dc_imaginary > imaginary_rtol * scale
        if np.any(inconsistent):
            raise ValueError(
                "spectrum is inconsistent with a real time series"
            )
        restored = fft.irfft(
            values, n=padded_length, axis=axis, workers=-1
        )
        restored = np.take(
            restored, np.arange(original_length), axis=axis
        )
    dims = tuple(time_dim if dim == _OMEGA_DIM else dim for dim in spectrum.dims)
    coords = {
        name: coord
        for name, coord in spectrum.coords.items()
        if _OMEGA_DIM not in coord.dims
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


def butterworth_filter(
    data: xr.DataArray | xr.Dataset,
    cutoff_omega: float,
    *,
    order: int = 4,
    time_dim: str = "time",
    sample_spacing_seconds: float | None = None,
) -> xr.DataArray | xr.Dataset:
    """Apply a zero-phase low-pass Butterworth filter along time.

    ``cutoff_omega`` is the single-pass critical angular frequency in rad/s and
    ``order`` is the single-pass filter order. The filter is designed as
    second-order sections and applied forward and backward, so the returned
    signal has zero phase, twice the stated order, and half the passband
    amplitude at ``cutoff_omega``.

    For a `Dataset`, every numeric data variable containing ``time_dim`` is
    filtered; other variables are preserved. Spatial locations that are NaN for
    the complete time series remain masked, while partial gaps are rejected.
    Odd reflection and SciPy's standard `sosfiltfilt` padding length are used at
    both endpoints. Coordinates, names, and attributes are preserved.

    ``sample_spacing_seconds`` has the same calendar-label semantics as
    `forward_transform`. In particular, monthly samples may be represented by
    their calendar labels while using ``365.25 / 12`` days as the physical
    interval.
    """
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise TypeError("data must be an xarray.DataArray or xarray.Dataset")
    if isinstance(cutoff_omega, bool) or not isinstance(
        cutoff_omega, (int, float)
    ):
        raise TypeError("cutoff_omega must be a real number")
    if not np.isfinite(cutoff_omega) or cutoff_omega <= 0:
        raise ValueError("cutoff_omega must be finite and positive")
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 1:
        raise ValueError("order must be at least 1")
    if not isinstance(time_dim, str) or not time_dim:
        raise TypeError("time_dim must be a nonempty string")
    _validate_sample_spacing(sample_spacing_seconds)
    if time_dim not in data.dims:
        raise ValueError(f"data must have a {time_dim!r} dimension")

    time = data if isinstance(data, xr.DataArray) else data[time_dim]
    _, dt_seconds = _validated_time(time, time_dim, sample_spacing_seconds)
    cutoff_hz = float(cutoff_omega) / (2.0 * np.pi)
    sampling_hz = 1.0 / dt_seconds
    if cutoff_hz >= 0.5 * sampling_hz:
        raise ValueError("cutoff_omega must be below the Nyquist frequency")
    sections = butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=sampling_hz,
        output="sos",
    )

    if isinstance(data, xr.DataArray):
        return _butterworth_dataarray(
            data,
            sections,
            time_dim,
        )

    variables = {}
    for name, variable in data.data_vars.items():
        if time_dim in variable.dims and np.issubdtype(variable.dtype, np.number):
            variables[name] = _butterworth_dataarray(
                variable,
                sections,
                time_dim,
            )
        else:
            variables[name] = variable
    return xr.Dataset(variables, coords=data.coords, attrs=data.attrs)


def _butterworth_dataarray(
    data: xr.DataArray,
    sections: np.ndarray,
    time_dim: str,
) -> xr.DataArray:
    """Filter one numeric array while retaining complete-series masks."""
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("data must contain numeric values")
    values = np.asarray(data.data)
    axis = data.get_axis_num(time_dim)
    if np.any(np.isinf(values)):
        raise ValueError("data must not contain infinite values")
    missing = np.isnan(values)
    fully_missing = np.all(missing, axis=axis)
    partially_missing = np.any(missing, axis=axis) & ~fully_missing
    if np.any(partially_missing):
        raise ValueError("data must not contain partial gaps along time")

    expanded_mask = np.expand_dims(fully_missing, axis=axis)
    working = np.where(expanded_mask, 0.0, values)
    filtered = sosfiltfilt(
        sections,
        working,
        axis=axis,
        padtype="odd",
    )
    filtered = np.where(expanded_mask, np.nan, filtered)
    return data.copy(data=filtered)


def _validate_forward_contract(time_dim: str, pad_length: int) -> None:
    """Validate the temporal dimension name and requested zero padding."""
    if not isinstance(time_dim, str) or not time_dim:
        raise TypeError("time_dim must be a nonempty string")
    if time_dim == _OMEGA_DIM:
        raise ValueError(f"time_dim must differ from {_OMEGA_DIM!r}")
    if isinstance(pad_length, bool) or not isinstance(pad_length, int):
        raise TypeError("pad_length must be an integer")
    if pad_length < 0:
        raise ValueError("pad_length must be nonnegative")


def _validate_tolerance(value: float, name: str) -> None:
    """Require a finite, nonnegative numerical tolerance."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _padded_length(original_length: int, pad_length: int) -> int:
    """Return an odd FFT length with at least the requested zero padding."""
    minimum = original_length + pad_length
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


def _validate_numeric_dtype(values, name: str) -> None:
    """Require a numeric dtype without evaluating a lazy array."""
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")


def _validate_real_dtype(values, name: str) -> None:
    """Require a real numeric dtype without evaluating a lazy array."""
    _validate_numeric_dtype(values, name)
    if np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real values")


def _validate_real_finite(values: np.ndarray, name: str) -> None:
    """Require finite real-valued input for a real Fourier transform."""
    _validate_numeric_finite(values, name)
    if np.iscomplexobj(values):
        raise TypeError(f"{name} must contain real values")


def _validated_real_finite_block(values: np.ndarray) -> np.ndarray:
    """Validate one lazy input block when it is evaluated."""
    _validate_real_finite(values, "data")
    return values


def _validated_zero_mean_block(
    values: np.ndarray,
    field_scale: np.ndarray,
    *,
    axis: int,
    zero_mean_rtol: float,
) -> np.ndarray:
    """Validate complete time series against the lazy global field scale."""
    means = np.abs(np.mean(values, axis=axis))
    if np.any(means > zero_mean_rtol * float(field_scale)):
        raise ValueError(
            "each time series must be zero mean; pass "
            "require_zero_mean=False to retain a nonzero mean"
        )
    return values


def _zero_dc_block(values: np.ndarray, *, axis: int) -> np.ndarray:
    """Project the zero-frequency coefficient of one block to zero."""
    values = values.copy()
    index = [slice(None)] * values.ndim
    index[axis] = 0
    values[tuple(index)] = 0.0
    return values


def _validated_spectrum_block(
    values: np.ndarray,
    *,
    axis: int,
    imaginary_rtol: float,
) -> np.ndarray:
    """Validate one block containing complete frequency series."""
    _validate_numeric_finite(values, "spectrum")
    scale = np.max(np.abs(values), axis=axis)
    dc_imaginary = np.abs(np.take(values, 0, axis=axis).imag)
    if np.any(dc_imaginary > imaginary_rtol * scale):
        raise ValueError("spectrum is inconsistent with a real time series")
    return values


def _validate_zero_mean(
    values: np.ndarray, axis: int, zero_mean_rtol: float
) -> None:
    """Require every series along ``axis`` to have negligible mean."""
    means = np.abs(np.mean(values, axis=axis))
    field_scale = np.max(np.abs(values))
    if np.any(means > zero_mean_rtol * field_scale):
        raise ValueError(
            "each time series must be zero mean; pass "
            "require_zero_mean=False to retain a nonzero mean"
        )


def _metadata_int(
    metadata: Mapping[object, object], key: str, *, minimum: int = 1
) -> int:
    """Read an integer no smaller than ``minimum`` from transform metadata."""
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {key} in spectral metadata")
    return value
