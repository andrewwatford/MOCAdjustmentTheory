"""Global Rossby-wave adjustment model."""

from __future__ import annotations

from numbers import Real
from typing import NamedTuple

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed

from .fourier import (
    _METADATA_KEY,
    _validate_real_dtype,
    _validate_real_finite,
    _validate_zero_mean,
    _validated_real_finite_block,
    _validated_zero_mean_block,
    _validated_time,
    forward_transform,
    inverse_transform,
)

EARTH_RADIUS_M = 6_371_000.0
EARTH_ROTATION_S = 7.292115e-5
_T_N_LATITUDE_ATTR = "latitude_degrees_north"
_INPUT_LATITUDE_CHUNK = 64
_INPUT_LONGITUDE_CHUNK = 128
_HEIGHT_LATITUDE_CHUNK = 8
_OUTPUT_TIME_CHUNK = 12

_REGIONS = (
    ("north_atlantic", "x_wA", "x_eA"),
    ("north_indian", "x_wI", "x_eI"),
    ("north_pacific", "x_wP", "x_eP"),
    ("atlantic_indian", "x_wA", "x_eI"),
    ("atlantic_pacific", "x_wA", "x_eP"),
)


class _ZonalRule(NamedTuple):
    """Precomputed interpolation and integration data for one latitude."""

    latitude_index: int
    interior: slice
    west: tuple[int, int, float]
    east: tuple[int, int, float]
    grid_indices: np.ndarray
    grid_nodes: np.ndarray
    x: np.ndarray
    weights: np.ndarray


class GlobalRossbyModel:
    """Solve the global reduced-gravity adjustment theory.

    Parameters
    ----------
    isobath_ds
        Active-layer geometry described by the six latitude-dependent boundary
        variables in the package specification.
    g_prime
        Reduced gravity in m s-2.

    Attributes
    ----------
    longest_crossing_time_seconds
        Longest zonal Rossby-wave crossing time across the model geometry.

    Notes
    -----
    `solve` is the complete temporal interface. Frequency-space operations
    are an internal implementation detail.
    """

    def __init__(self, isobath_ds: xr.Dataset, g_prime: float):
        """Validate and retain the active-layer geometry and reduced gravity."""
        required = {name for _, west, east in _REGIONS for name in (west, east)}
        missing = required.difference(isobath_ds.data_vars)
        if missing or "latitude" not in isobath_ds.coords:
            absent = sorted(missing | ({"latitude"} if "latitude" not in isobath_ds.coords else set()))
            raise ValueError(f"isobath dataset is missing: {', '.join(absent)}")
        if "isobath_depth_m" not in isobath_ds.attrs:
            raise ValueError("isobath dataset is missing attribute 'isobath_depth_m'")
        if not np.isfinite(g_prime) or g_prime <= 0:
            raise ValueError("g_prime must be positive")
        self.isobath_ds = isobath_ds
        self.g_prime = float(g_prime)
        self.longest_crossing_time_seconds = self._longest_crossing_time()

    def solve(
        self,
        forcing_ds: xr.Dataset,
        *,
        time_dim: str = "time",
        pad_length: int | None = None,
        require_zero_mean: bool = True,
        zero_mean_rtol: float = 1e-7,
        imaginary_rtol: float = 1e-12,
        sample_spacing_seconds: float | None = None,
    ) -> xr.Dataset:
        """Solve temporal forcing and return temporal model diagnostics.

        The three forcing variables ``M_Ek_x``, ``M_Ek_y`` and ``T_N`` are
        combined into Ekman diagnostics, transformed with one explicit
        contract, and inverse transformed onto their original time coordinate.
        All forcing-dependent calculations are represented as a Dask graph;
        this method returns without loading the full forcing or computing any
        output values. Callers may compute a subset, persist selected results,
        or stream the dataset to a chunked store.
        ``sample_spacing_seconds`` supplies a physical sample
        interval when increasing calendar labels are not uniformly separated;
        monthly forcing in this project uses ``365.25 / 12`` days. All other
        transform arguments are forwarded unchanged to the paired Fourier
        functions. With ``pad_length=None``, the model appends enough zero
        samples to cover its longest Rossby-wave crossing time and makes the
        complete FFT length odd. An explicit ``pad_length`` is the minimum
        number of zero samples appended and overrides that physical default.

        Parameters
        ----------
        forcing_ds
            Temporal Ekman transports and northern transport on a shared time
            coordinate. ``T_N`` must have a numeric
            ``latitude_degrees_north`` attribute specifying its boundary.
        time_dim
            Temporal dimension name. Angular frequency is always the internal
            ``omega`` dimension in rad s-1.
        pad_length
            Minimum number of right-padding samples. With ``None``, use at
            least the longest model crossing time in units of the forcing time
            step. One additional sample is used when needed to make the full
            FFT length odd.
        require_zero_mean, zero_mean_rtol
            Whether to enforce forcing anomalies and the field-scale relative
            tolerance used before projecting accepted DC residuals to zero.
        imaginary_rtol
            Relative tolerance for spectral components that must be real.
        sample_spacing_seconds
            Optional physical interval for nonuniform calendar labels. With
            ``None``, a uniform interval is inferred from the time coordinate.

        Returns
        -------
        xarray.Dataset
            ``h_e``, ``h_b``, ``h_w``, ``h``, ``T``, ``T_g`` and ``T_Ek`` on
            the original time coordinate and the five-region model geometry.
            Every numerical variable is Dask-backed. ``h`` has dimensions
            ``(time, region, latitude, longitude)`` and is NaN outside each
            region. Other fields use their documented lower-dimensional
            coordinates.
        """
        required = ("M_Ek_x", "M_Ek_y", "T_N")
        missing = set(required).difference(forcing_ds.data_vars)
        if missing:
            raise ValueError(
                f"forcing dataset is missing: {', '.join(sorted(missing))}"
            )
        self._northern_boundary_latitude(forcing_ds["T_N"])

        forcing_lazy = forcing_ds[list(required)].chunk(
            {
                time_dim: -1,
                "latitude": _INPUT_LATITUDE_CHUNK,
                "longitude": _INPUT_LONGITUDE_CHUNK,
            }
        )

        if pad_length is None:
            _, dt_seconds = _validated_time(
                forcing_ds["T_N"], time_dim, sample_spacing_seconds
            )
            pad_length = int(
                np.ceil(self.longest_crossing_time_seconds / dt_seconds)
            )

        transform_options = {
            "time_dim": time_dim,
            "pad_length": pad_length,
            "sample_spacing_seconds": sample_spacing_seconds,
        }
        t_n_hat = forward_transform(
            forcing_lazy["T_N"],
            require_zero_mean=require_zero_mean,
            zero_mean_rtol=zero_mean_rtol,
            **transform_options,
        )

        latitude, longitude, mx, my = self._temporal_wind_arrays(
            forcing_lazy,
            time_dim,
            require_zero_mean,
            zero_mean_rtol,
        )
        geometry = self._geometry(
            latitude, self._northern_boundary_latitude(t_n_hat)
        )
        quadrature = self._zonal_quadrature(
            longitude, latitude, geometry
        )
        # These spatial linear operations commute with the temporal FFT.
        w_ek = self._ekman_curl(mx, my, latitude, longitude)
        del mx
        t_ek = self._regional_ekman_transport(
            my, quadrature, latitude.size
        )
        del my

        common_coords = {
            time_dim: forcing_ds[time_dim],
            "latitude": forcing_ds["latitude"],
        }
        w_ek_array = xr.DataArray(
            w_ek,
            dims=(time_dim, "latitude", "longitude"),
            coords={**common_coords, "longitude": forcing_ds["longitude"]},
            name="w_Ek",
        )
        t_ek_array = xr.DataArray(
            t_ek,
            dims=(time_dim, "region", "latitude"),
            coords={
                **common_coords,
                "region": [name for name, _, _ in _REGIONS],
            },
            name="T_Ek",
        )
        w_ek_hat = forward_transform(
            w_ek_array,
            require_zero_mean=require_zero_mean,
            zero_mean_rtol=zero_mean_rtol,
            **transform_options,
        )
        del w_ek, w_ek_array
        t_ek_hat = forward_transform(
            t_ek_array,
            require_zero_mean=require_zero_mean,
            zero_mean_rtol=zero_mean_rtol,
            **transform_options,
        )

        solution_hat = self._solve_from_ekman_lazy(
            np.asarray(w_ek_hat.omega.values, dtype=float),
            latitude,
            w_ek_hat.transpose("omega", "latitude", "longitude").data,
            t_ek_hat.transpose("omega", "region", "latitude").data,
            t_n_hat.transpose("omega").data,
            geometry,
            quadrature,
            w_ek_hat.omega,
            w_ek_hat.latitude,
            w_ek_hat.longitude,
        )

        metadata = t_n_hat.attrs[_METADATA_KEY]
        del w_ek_hat, t_ek_hat, t_n_hat
        solution = {}
        for name, spectrum in solution_hat.data_vars.items():
            spectrum = spectrum.copy(deep=False)
            spectrum.attrs = dict(spectrum.attrs)
            spectrum.attrs[_METADATA_KEY] = metadata
            masked = spectrum.isnull().all("omega")
            transform_input = spectrum.fillna(0.0)
            restored = inverse_transform(
                transform_input, imaginary_rtol=imaginary_rtol
            )
            restored = restored.where(~masked)
            if isinstance(restored.data, da.Array):
                chunk_map = {time_dim: _OUTPUT_TIME_CHUNK}
                if "region" in restored.dims:
                    chunk_map["region"] = 1
                if "latitude" in restored.dims:
                    chunk_map["latitude"] = _INPUT_LATITUDE_CHUNK
                if "longitude" in restored.dims:
                    chunk_map["longitude"] = _INPUT_LONGITUDE_CHUNK
                restored = restored.chunk(chunk_map)
            solution[name] = restored

        result = xr.Dataset(solution)
        result.attrs.update(
            {
                "g_prime_m_s-2": self.g_prime,
                "isobath_depth_m": float(
                    self.isobath_ds.attrs["isobath_depth_m"]
                ),
            }
        )
        return result

    def _solve_frequency(self, forcing_hat: xr.Dataset) -> xr.Dataset:
        """Run the internal frequency-space compatibility path."""
        omega, latitude, longitude, mx, my, t_n, y_n = self._forcing_arrays(
            forcing_hat
        )
        geometry = self._geometry(latitude, y_n)
        quadrature = self._zonal_quadrature(
            longitude, latitude, geometry
        )
        w_ek = self._ekman_curl(mx, my, latitude, longitude)
        t_ek = self._regional_ekman_transport(
            my, quadrature, latitude.size
        )
        return self._solve_from_ekman(
            omega,
            latitude,
            w_ek,
            t_ek,
            t_n,
            geometry,
            quadrature,
            forcing_hat.omega,
            forcing_hat.latitude,
            forcing_hat.longitude,
        )

    def _solve_from_ekman_lazy(
        self,
        omega: np.ndarray,
        latitude: np.ndarray,
        w_ek: da.Array,
        t_ek_forcing: da.Array,
        t_n: da.Array,
        geometry,
        quadrature,
        omega_coordinate: xr.DataArray,
        latitude_coordinate: xr.DataArray,
        longitude_coordinate: xr.DataArray,
    ) -> xr.Dataset:
        """Build the complete frequency-space solution as a Dask graph."""
        lat_rad = np.deg2rad(latitude)
        f = 2.0 * EARTH_ROTATION_S * np.sin(lat_rad)
        depth = float(self.isobath_ds.attrs["isobath_depth_m"])
        gh = self.g_prime * depth
        c = rossby_speed(latitude, self.g_prime, depth)
        latitude_size = latitude.size

        # The two zonal source terms share the same forcing rows and phases.
        # Build them together in latitude blocks so the Dask graph references
        # each full-longitude forcing block only once.
        q_full, source_full = self._zonal_terms_lazy(
            w_ek, omega, c, quadrature, latitude_size
        )

        active_mask = np.zeros((len(_REGIONS), latitude_size), dtype=bool)
        f_regions = []
        r_regions = []
        source_regions = []
        propagation_regions = []
        for region in range(len(_REGIONS)):
            indices, x_w, x_e = geometry[region]
            active_mask[region, indices] = True
            q_f = q_full[:, region, indices]
            source = source_full[:, region, indices]
            dy = EARTH_RADIUS_M * np.deg2rad(latitude[indices])
            f_region = _integral_to_north(q_f, dy)
            phase_e = np.exp(
                1j
                * omega[:, None]
                * EARTH_RADIUS_M
                * np.cos(np.deg2rad(latitude[indices]))[None, :]
                * np.deg2rad(x_w - x_e)[None, :]
                / c[indices][None, :]
            )
            r_region = _integral_to_north(
                c[indices][None, :] * (phase_e - 1.0), dy
            )
            f_regions.append(f_region)
            r_regions.append(r_region)
            source_regions.append(source)
            propagation_regions.append(phase_e)

        south = np.array([item[0][0] for item in geometry])
        full_f = da.stack(
            [values[:, 0] for values in f_regions], axis=1
        )
        full_r = np.stack(
            [values[:, 0] for values in r_regions], axis=1
        )
        t_i_ek = t_ek_forcing[:, 1, south[1]]
        t_p_ek = t_ek_forcing[:, 2, south[2]]
        t_s = t_ek_forcing[:, 4, south[4]]

        y_i = latitude[south[0]]
        y_p = latitude[south[2]]
        kappa_i = gh / (
            2.0 * EARTH_ROTATION_S * np.sin(np.deg2rad(y_i))
        )
        kappa_p = gh / (
            2.0 * EARTH_ROTATION_S * np.sin(np.deg2rad(y_p))
        )
        rhs = da.stack(
            (
                full_f[:, 0]
                + full_f[:, 3]
                + full_f[:, 4]
                + t_n
                + t_i_ek
                + t_p_ek
                - t_s,
                full_f[:, 1] - t_i_ek,
                full_f[:, 2] - t_p_ek,
            ),
            axis=1,
        )
        inverse = self._boundary_inverse(
            omega, full_r, kappa_i, kappa_p
        )
        h_basin = da.einsum("nij,nj->ni", inverse, rhs)
        h_e = h_basin[:, [0, 1, 2, 1, 2]]

        source = da.stack(
            [
                _pad_active_latitudes(
                    values, geometry[region][0], latitude_size
                )
                for region, values in enumerate(source_regions)
            ],
            axis=1,
        )
        propagation = np.full(
            (omega.size, len(_REGIONS), latitude_size), np.nan + 0j
        )
        for region, values in enumerate(propagation_regions):
            propagation[:, region, geometry[region][0]] = values
        h_b = h_e[:, :, None] * propagation - source

        active_transport = []
        top = [t_n, da.zeros_like(t_n), da.zeros_like(t_n)]
        for region in range(3):
            active_transport.append(
                top[region][:, None]
                + f_regions[region]
                - r_regions[region] * h_e[:, region, None]
            )
        top_4 = active_transport[0][:, 0] + active_transport[1][:, 0]
        active_transport.append(
            top_4[:, None]
            + f_regions[3]
            - r_regions[3] * h_e[:, 3, None]
        )
        top_5 = active_transport[2][:, 0] + active_transport[3][:, 0]
        active_transport.append(
            top_5[:, None]
            + f_regions[4]
            - r_regions[4] * h_e[:, 4, None]
        )
        transport = da.stack(
            [
                _pad_active_latitudes(
                    values, geometry[region][0], latitude_size
                )
                for region, values in enumerate(active_transport)
            ],
            axis=1,
        )

        t_ek = da.where(
            active_mask[None, :, :], t_ek_forcing, np.nan
        )
        t_g = transport - t_ek
        h_w = h_e[:, :, None] - f[None, None, :] * t_g / gh
        h = self._height_field_lazy(
            w_ek,
            omega,
            c,
            h_e,
            quadrature,
            latitude_size,
            longitude_coordinate.size,
        )

        coords = {
            "omega": omega_coordinate,
            "region": [name for name, _, _ in _REGIONS],
            "latitude": latitude_coordinate,
            "longitude": longitude_coordinate,
        }
        result = xr.Dataset(
            {
                "h_e": (("omega", "region"), h_e),
                "h_b": (("omega", "region", "latitude"), h_b),
                "h_w": (("omega", "region", "latitude"), h_w),
                "h": (
                    ("omega", "region", "latitude", "longitude"),
                    h,
                ),
                "T": (("omega", "region", "latitude"), transport),
                "T_g": (("omega", "region", "latitude"), t_g),
                "T_Ek": (("omega", "region", "latitude"), t_ek),
            },
            coords=coords,
        )
        for name in ("h_e", "h_b", "h_w", "h"):
            result[name].attrs["units"] = "m"
        for name in ("T", "T_g", "T_Ek"):
            result[name].attrs["units"] = "m3 s-1"
        result.omega.attrs.setdefault("units", "rad s-1")
        return result

    @staticmethod
    def _boundary_inverse(
        omega: np.ndarray,
        r: np.ndarray,
        kappa_i: float,
        kappa_p: float,
    ) -> np.ndarray:
        """Return the inverse boundary matrix at every nonzero frequency."""
        inverse = np.zeros((omega.size, 3, 3), dtype=complex)
        zero = np.isclose(
            omega, 0.0, rtol=0.0, atol=np.finfo(float).eps
        )
        for n in np.flatnonzero(~zero):
            matrix = np.array(
                [
                    [
                        r[n, 0] + kappa_i,
                        r[n, 3] + kappa_p - kappa_i,
                        r[n, 4] - kappa_p,
                    ],
                    [-kappa_i, r[n, 1] + kappa_i, 0.0],
                    [0.0, -kappa_p, r[n, 2] + kappa_p],
                ],
                dtype=complex,
            )
            inverse[n] = np.linalg.inv(matrix)
        return inverse

    @staticmethod
    def _height_field_lazy(
        w: da.Array,
        omega: np.ndarray,
        c: np.ndarray,
        h_e: da.Array,
        quadrature,
        latitude_size: int,
        longitude_size: int,
    ) -> da.Array:
        """Build lazy dense height spectra in full-longitude latitude blocks.

        Each output block contains eight latitudes for one region. A block
        needs complete longitude rows because the height at a point depends on
        the forcing integral from that point to the eastern boundary. The
        returned graph is later split into normal output chunks without ever
        constructing a dense in-memory global spectrum.
        """
        dtype = np.result_type(w.dtype, np.complex64)
        # A small latitude chunk bounds each full-longitude task to roughly
        # tens of megabytes at the supplied native resolution.
        zonal = w.rechunk({0: -1, 1: _HEIGHT_LATITUDE_CHUNK, 2: -1})
        latitude_chunks = zonal.chunks[1]
        source_blocks = zonal.to_delayed().reshape(
            1, len(latitude_chunks), 1
        )[0, :, 0]
        h_e_block = h_e.rechunk({0: -1, 1: -1}).to_delayed().reshape(
            1, 1
        )[0, 0]
        regions = []
        for region in range(len(_REGIONS)):
            output_blocks = []
            latitude_start = 0
            for source, latitude_count in zip(
                source_blocks, latitude_chunks
            ):
                block = delayed(_height_spectral_block)(
                    source,
                    h_e_block,
                    region,
                    latitude_start,
                    quadrature[region],
                    omega,
                    c,
                    longitude_size,
                    dtype,
                )
                output_blocks.append(
                    da.from_delayed(
                        block,
                        shape=(omega.size, latitude_count, longitude_size),
                        dtype=dtype,
                    )
                )
                latitude_start += latitude_count
            regions.append(da.concatenate(output_blocks, axis=1))
        if sum(latitude_chunks) != latitude_size:
            raise RuntimeError("lazy height-field latitude shape is inconsistent")
        return da.stack(regions, axis=1).rechunk(
            {
                0: -1,
                1: 1,
                2: _INPUT_LATITUDE_CHUNK,
                3: _INPUT_LONGITUDE_CHUNK,
            }
        )

    def _solve_from_ekman(
        self,
        omega: np.ndarray,
        latitude: np.ndarray,
        w_ek: np.ndarray,
        t_ek_forcing: np.ndarray,
        t_n: np.ndarray,
        geometry,
        quadrature,
        omega_coordinate: xr.DataArray,
        latitude_coordinate: xr.DataArray,
        longitude_coordinate: xr.DataArray,
    ) -> xr.Dataset:
        """Apply the frequency-space Rossby-wave kernel."""
        lat_rad = np.deg2rad(latitude)
        f = 2.0 * EARTH_ROTATION_S * np.sin(lat_rad)
        depth = float(self.isobath_ds.attrs["isobath_depth_m"])
        gh = self.g_prime * depth
        c = rossby_speed(latitude, self.g_prime, depth)

        shape = (omega.size, len(_REGIONS), latitude.size)
        f_partial = np.full(shape, np.nan + 0j)
        r_partial = np.full(shape, np.nan + 0j)
        h_source = np.full(shape, np.nan + 0j)
        t_ek = np.full(shape, np.nan + 0j)
        propagation = np.full(shape, np.nan + 0j)

        for region, (_, _, _) in enumerate(_REGIONS):
            indices, x_w, x_e = geometry[region]
            q_f, source = self._zonal_terms(
                w_ek,
                omega,
                c,
                quadrature[region],
            )
            dy = EARTH_RADIUS_M * np.deg2rad(latitude[indices])
            f_partial[:, region, indices] = _integral_to_north(q_f, dy)
            phase_e = np.exp(
                1j
                * omega[:, None]
                * EARTH_RADIUS_M
                * np.cos(np.deg2rad(latitude[indices]))[None, :]
                * np.deg2rad(x_w - x_e)[None, :]
                / c[indices][None, :]
            )
            r_partial[:, region, indices] = _integral_to_north(
                c[indices][None, :] * (phase_e - 1.0), dy
            )
            h_source[:, region, indices] = source
            t_ek[:, region, indices] = t_ek_forcing[:, region, indices]
            propagation[:, region, indices] = phase_e

        south = np.array([item[0][0] for item in geometry])
        full_f = f_partial[:, np.arange(5), south]
        full_r = r_partial[:, np.arange(5), south]
        t_i_ek = t_ek[:, 1, south[1]]
        t_p_ek = t_ek[:, 2, south[2]]
        t_s = t_ek[:, 4, south[4]]

        y_i = latitude[south[0]]
        y_p = latitude[south[2]]
        kappa_i = gh / (2.0 * EARTH_ROTATION_S * np.sin(np.deg2rad(y_i)))
        kappa_p = gh / (2.0 * EARTH_ROTATION_S * np.sin(np.deg2rad(y_p)))
        h_basin = self._solve_boundaries(
            omega, full_f, full_r, t_n, t_i_ek, t_p_ek, t_s, kappa_i, kappa_p
        )
        h_e = h_basin[:, [0, 1, 2, 1, 2]]
        h_b = h_e[:, :, None] * propagation - h_source

        transport = np.full(shape, np.nan + 0j)
        top = [t_n, np.zeros_like(t_n), np.zeros_like(t_n)]
        for region in range(3):
            indices = geometry[region][0]
            transport[:, region, indices] = (
                top[region][:, None]
                + f_partial[:, region, indices]
                - r_partial[:, region, indices] * h_e[:, region, None]
            )
        top_4 = transport[:, 0, south[0]] + transport[:, 1, south[1]]
        indices = geometry[3][0]
        transport[:, 3, indices] = (
            top_4[:, None]
            + f_partial[:, 3, indices]
            - r_partial[:, 3, indices] * h_e[:, 3, None]
        )
        top_5 = transport[:, 2, south[2]] + transport[:, 3, south[3]]
        indices = geometry[4][0]
        transport[:, 4, indices] = (
            top_5[:, None]
            + f_partial[:, 4, indices]
            - r_partial[:, 4, indices] * h_e[:, 4, None]
        )

        t_g = transport - t_ek
        h_w = h_e[:, :, None] - f[None, None, :] * t_g / gh
        h = self._height_field(
            w_ek,
            omega,
            c,
            h_e,
            quadrature,
            latitude,
            np.asarray(longitude_coordinate.values, dtype=float),
        )
        coords = {
            "omega": omega_coordinate,
            "region": [name for name, _, _ in _REGIONS],
            "latitude": latitude_coordinate,
            "longitude": longitude_coordinate,
        }
        result = xr.Dataset(
            {
                "h_e": (("omega", "region"), h_e),
                "h_b": (("omega", "region", "latitude"), h_b),
                "h_w": (("omega", "region", "latitude"), h_w),
                "h": (("omega", "region", "latitude", "longitude"), h),
                "T": (("omega", "region", "latitude"), transport),
                "T_g": (("omega", "region", "latitude"), t_g),
                "T_Ek": (("omega", "region", "latitude"), t_ek),
            },
            coords=coords,
        )
        for name in ("h_e", "h_b", "h_w", "h"):
            result[name].attrs["units"] = "m"
        for name in ("T", "T_g", "T_Ek"):
            result[name].attrs["units"] = "m3 s-1"
        result.omega.attrs.setdefault("units", "rad s-1")
        return result

    @staticmethod
    def _forcing_arrays(forcing_hat: xr.Dataset):
        """Validate spectral forcing and return consistently ordered arrays."""
        missing = {"M_Ek_x", "M_Ek_y", "T_N"}.difference(forcing_hat.data_vars)
        missing_dims = {"omega", "latitude", "longitude"}.difference(forcing_hat.coords)
        if missing or missing_dims:
            raise ValueError(f"spectral forcing is missing: {', '.join(sorted(missing | missing_dims))}")
        omega = np.asarray(forcing_hat.omega.values, dtype=float)
        latitude = np.asarray(forcing_hat.latitude.values, dtype=float)
        longitude = np.asarray(forcing_hat.longitude.values, dtype=float)
        if omega.ndim != 1 or latitude.ndim != 1 or longitude.ndim != 1:
            raise ValueError("omega, latitude and longitude must be one-dimensional")
        if latitude.size < 3 or longitude.size < 3:
            raise ValueError("spatial coordinates require at least three points")
        if np.any(np.diff(latitude) <= 0) or np.any(np.diff(longitude) <= 0):
            raise ValueError("latitude and longitude must increase")
        mx = np.asarray(
            forcing_hat.M_Ek_x.transpose(
                "omega", "latitude", "longitude"
            ).values
        )
        my = np.asarray(
            forcing_hat.M_Ek_y.transpose(
                "omega", "latitude", "longitude"
            ).values
        )
        wind_dtype = np.result_type(mx.dtype, my.dtype, np.complex64)
        mx = np.asarray(mx, dtype=wind_dtype)
        my = np.asarray(my, dtype=wind_dtype)
        t_n = np.asarray(forcing_hat.T_N.transpose("omega").values, dtype=complex)
        y_n = GlobalRossbyModel._northern_boundary_latitude(forcing_hat.T_N)
        return omega, latitude, longitude, mx, my, t_n, y_n

    @staticmethod
    def _temporal_wind_arrays(
        forcing_ds: xr.Dataset,
        time_dim: str,
        require_zero_mean: bool,
        zero_mean_rtol: float,
    ):
        """Validate temporal wind forcing and return ordered arrays."""
        missing_coords = {"latitude", "longitude"}.difference(
            forcing_ds.coords
        )
        if missing_coords:
            raise ValueError(
                "forcing dataset is missing: "
                + ", ".join(sorted(missing_coords))
            )
        latitude = np.asarray(forcing_ds.latitude.values, dtype=float)
        longitude = np.asarray(forcing_ds.longitude.values, dtype=float)
        if latitude.ndim != 1 or longitude.ndim != 1:
            raise ValueError("latitude and longitude must be one-dimensional")
        if latitude.size < 3 or longitude.size < 3:
            raise ValueError("spatial coordinates require at least three points")
        if np.any(np.diff(latitude) <= 0) or np.any(np.diff(longitude) <= 0):
            raise ValueError("latitude and longitude must increase")

        expected_dims = {time_dim, "latitude", "longitude"}
        arrays = []
        for name in ("M_Ek_x", "M_Ek_y"):
            variable = forcing_ds[name]
            if set(variable.dims) != expected_dims:
                raise ValueError(
                    f"{name} must have dimensions {sorted(expected_dims)}"
                )
            raw_values = variable.transpose(
                time_dim, "latitude", "longitude"
            ).data
            if isinstance(raw_values, da.Array):
                _validate_real_dtype(raw_values, name)
                values = raw_values.map_blocks(
                    _validated_real_finite_block,
                    dtype=raw_values.dtype,
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
                        axis=0,
                        zero_mean_rtol=zero_mean_rtol,
                        dtype=values.dtype,
                    )
            else:
                values = np.asarray(raw_values)
                _validate_real_finite(values, name)
                if require_zero_mean:
                    _validate_zero_mean(values, 0, zero_mean_rtol)
            arrays.append(values)

        wind_dtype = np.result_type(
            arrays[0].dtype, arrays[1].dtype, np.float32
        )
        if isinstance(arrays[0], da.Array):
            mx = arrays[0].astype(wind_dtype)
            my = arrays[1].astype(wind_dtype)
        else:
            mx = np.asarray(arrays[0], dtype=wind_dtype)
            my = np.asarray(arrays[1], dtype=wind_dtype)
        return latitude, longitude, mx, my

    @staticmethod
    def _ekman_curl(
        mx: np.ndarray,
        my: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> np.ndarray:
        """Return the global divergence of the Ekman transport."""
        lat_rad = np.deg2rad(latitude)
        lon_rad = np.deg2rad(longitude)
        real_dtype = np.empty((), dtype=mx.dtype).real.dtype
        cos_lat = np.cos(lat_rad).astype(real_dtype, copy=False)

        if isinstance(mx, da.Array):
            zonal = da.gradient(
                mx, lon_rad, axis=2, edge_order=2
            )
            meridional = da.gradient(
                my * cos_lat[None, :, None],
                lat_rad,
                axis=1,
                edge_order=2,
            )
            denominator = (
                EARTH_RADIUS_M * cos_lat
            ).astype(real_dtype, copy=False)
            return (
                (zonal + meridional) / denominator[None, :, None]
            ).astype(real_dtype)

        w_ek = np.gradient(mx, lon_rad, axis=2, edge_order=2)
        my_cos = my * cos_lat[None, :, None]
        meridional = np.gradient(
            my_cos, lat_rad, axis=1, edge_order=2
        )
        np.add(w_ek, meridional, out=w_ek)
        del meridional, my_cos
        denominator = (EARTH_RADIUS_M * cos_lat).astype(
            real_dtype, copy=False
        )
        np.divide(w_ek, denominator[None, :, None], out=w_ek)
        return w_ek

    @staticmethod
    def _northern_boundary_latitude(t_n: xr.DataArray) -> float:
        """Return the validated latitude at which ``T_N`` is prescribed."""
        value = t_n.attrs.get(_T_N_LATITUDE_ATTR)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"T_N must have a finite numeric '{_T_N_LATITUDE_ATTR}' attribute"
            )
        value = float(value)
        if value < 0.0:
            raise ValueError("T_N latitude must not fall south of the equator")
        return value

    def _longest_crossing_time(self) -> float:
        """Return the longest zonal Rossby-wave transit in the geometry."""
        latitude = np.asarray(self.isobath_ds.latitude.values, dtype=float)
        if (
            latitude.ndim != 1
            or latitude.size < 2
            or not np.all(np.isfinite(latitude))
            or np.any(np.diff(latitude) <= 0)
            or np.any(np.abs(latitude) > 90.0)
        ):
            raise ValueError("isobath latitude must be finite and increasing")
        depth = float(self.isobath_ds.attrs["isobath_depth_m"])
        if not np.isfinite(depth) or depth <= 0:
            raise ValueError("isobath_depth_m must be positive")
        speed = rossby_speed(latitude, self.g_prime, depth)
        zonal_scale = EARTH_RADIUS_M * np.cos(np.deg2rad(latitude))
        boundaries = {}
        support = {}
        for name in {item for _, west, east in _REGIONS for item in (west, east)}:
            if self.isobath_ds[name].dims != ("latitude",):
                raise ValueError("isobath boundaries must depend only on latitude")
            values = np.asarray(self.isobath_ds[name].values, dtype=float)
            valid = np.flatnonzero(np.isfinite(values))
            if valid.size == 0 or np.any(np.diff(valid) != 1):
                raise ValueError(f"{name} must have one contiguous finite segment")
            boundaries[name] = values
            support[name] = (latitude[valid[0]], latitude[valid[-1]])
        y_s = max(support["x_wA"][0], support["x_eP"][0])
        y_p = max(support["x_wP"][0], support["x_eI"][0])
        y_i = max(support["x_eA"][0], support["x_wI"][0])
        limits = (
            (y_i, min(support["x_wA"][1], support["x_eA"][1])),
            (y_i, min(support["x_wI"][1], support["x_eI"][1])),
            (y_p, min(support["x_wP"][1], support["x_eP"][1])),
            (y_p, y_i),
            (y_s, y_p),
        )
        longest = 0.0
        for (region, west, east), (south, north) in zip(_REGIONS, limits):
            x_w = boundaries[west]
            x_e = boundaries[east]
            valid = (
                np.isfinite(x_w)
                & np.isfinite(x_e)
                & (latitude >= south)
                & (latitude <= north)
                & (speed > 0.0)
            )
            if not np.any(valid):
                raise ValueError(f"isobath geometry does not resolve {region}")
            width = x_e[valid] - x_w[valid]
            if np.any(width <= 0.0):
                raise ValueError(
                    "each eastern boundary must lie east of its western boundary"
                )
            crossing = (
                zonal_scale[valid] * np.deg2rad(width) / speed[valid]
            )
            longest = max(longest, float(np.max(crossing)))
        return longest

    def _geometry(self, latitude: np.ndarray, atlantic_north: float):
        """Interpolate boundaries and infer the five contiguous region supports."""
        source_lat = np.asarray(self.isobath_ds.latitude.values, dtype=float)
        support = {}
        boundaries = {}
        for name in {item for _, west, east in _REGIONS for item in (west, east)}:
            values = np.asarray(self.isobath_ds[name].values, dtype=float)
            valid = np.flatnonzero(np.isfinite(values))
            if valid.size == 0 or np.any(np.diff(valid) != 1):
                raise ValueError(f"{name} must have one contiguous finite segment")
            support[name] = (source_lat[valid[0]], source_lat[valid[-1]])
            boundaries[name] = np.interp(latitude, source_lat[valid], values[valid], left=np.nan, right=np.nan)

        y_s = max(support["x_wA"][0], support["x_eP"][0])
        y_p = max(support["x_wP"][0], support["x_eI"][0])
        y_i = max(support["x_eA"][0], support["x_wI"][0])
        atlantic_support_north = min(support["x_wA"][1], support["x_eA"][1])
        if atlantic_north > latitude[-1]:
            raise ValueError("forcing latitude does not reach the T_N latitude")
        if atlantic_north > atlantic_support_north:
            raise ValueError("Atlantic geometry does not reach the T_N latitude")
        limits = (
            (y_i, atlantic_north),
            (y_i, min(support["x_wI"][1], support["x_eI"][1])),
            (y_p, min(support["x_wP"][1], support["x_eP"][1])),
            (y_p, y_i),
            (y_s, y_p),
        )
        def ceiling(value):
            """Return the first forcing latitude at or north of a boundary."""
            candidates = latitude[latitude >= value]
            if candidates.size == 0:
                raise ValueError("forcing latitude does not cover the active domain")
            return candidates[0]

        def floor(value):
            """Return the last forcing latitude at or south of a boundary."""
            candidates = latitude[latitude <= value]
            if candidates.size == 0:
                raise ValueError("forcing latitude does not cover the active domain")
            return candidates[-1]

        # Transition latitudes are snapped northward once and reused on both
        # sides, so adjacent regions contain the same grid point.
        grid_y_s, grid_y_p, grid_y_i = ceiling(y_s), ceiling(y_p), ceiling(y_i)
        snapped = (
            (grid_y_i, floor(limits[0][1])),
            (grid_y_i, floor(limits[1][1])),
            (grid_y_p, floor(limits[2][1])),
            (grid_y_p, grid_y_i),
            (grid_y_s, grid_y_p),
        )
        output = []
        for region, (_, west, east) in enumerate(_REGIONS):
            lower, upper = snapped[region]
            mask = (
                (latitude >= lower)
                & (latitude <= upper)
                & np.isfinite(boundaries[west])
                & np.isfinite(boundaries[east])
            )
            indices = np.flatnonzero(mask)
            if indices.size < 2:
                raise ValueError(f"forcing grid does not resolve region {_REGIONS[region][0]}")
            output.append((indices, boundaries[west][indices], boundaries[east][indices]))
        return output

    @staticmethod
    def _zonal_quadrature(longitude, latitude, geometry):
        """Precompute regional interpolation and trapezoidal weights."""
        output = []
        for indices, x_w, x_e in geometry:
            rules = []
            for latitude_index, west, east in zip(indices, x_w, x_e):
                inside = np.flatnonzero(
                    (longitude > west) & (longitude < east)
                )
                interior = (
                    slice(int(inside[0]), int(inside[-1]) + 1)
                    if inside.size
                    else slice(0, 0)
                )
                x_deg = np.concatenate(
                    ([west], longitude[interior], [east])
                )
                if np.any(np.diff(x_deg) <= 0):
                    raise ValueError(
                        "each eastern boundary must lie east of its "
                        "western boundary"
                    )
                x = (
                    EARTH_RADIUS_M
                    * np.cos(np.deg2rad(latitude[latitude_index]))
                    * np.deg2rad(x_deg)
                )
                dx = np.diff(x)
                weights = np.empty_like(x)
                weights[0] = 0.5 * dx[0]
                weights[-1] = 0.5 * dx[-1]
                if weights.size > 2:
                    weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
                rules.append(
                    _ZonalRule(
                        int(latitude_index),
                        interior,
                        GlobalRossbyModel._interpolation_rule(
                            longitude, west
                        ),
                        GlobalRossbyModel._interpolation_rule(
                            longitude, east
                        ),
                        np.flatnonzero(
                            (longitude >= west) & (longitude <= east)
                        ),
                        np.searchsorted(
                            x_deg,
                            longitude[
                                (longitude >= west) & (longitude <= east)
                            ],
                        ),
                        x,
                        weights,
                    )
                )
            output.append(rules)
        return output

    @staticmethod
    def _height_field(
        w: np.ndarray,
        omega: np.ndarray,
        c: np.ndarray,
        h_e: np.ndarray,
        quadrature,
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> np.ndarray:
        """Evaluate thickness at every forcing-grid point in each region."""
        output = np.full(
            (
                omega.size,
                len(_REGIONS),
                latitude.size,
                longitude.size,
            ),
            np.nan + 0j,
            dtype=np.result_type(w.dtype, np.complex64),
        )
        for region, rules in enumerate(quadrature):
            for rule in rules:
                field = GlobalRossbyModel._field_on_zonal_grid(
                    w[:, rule.latitude_index, :], rule
                )
                speed = c[rule.latitude_index]
                phase_basis = np.exp(
                    -1j * omega[:, None] * rule.x[None, :] / speed
                )
                integrand = field * phase_basis
                increments = (
                    0.5
                    * (integrand[:, :-1] + integrand[:, 1:])
                    * np.diff(rule.x)[None, :]
                )
                integral = np.zeros_like(integrand, dtype=complex)
                integral[:, :-1] = np.cumsum(
                    increments[:, ::-1], axis=1
                )[:, ::-1]
                phase = np.exp(
                    1j
                    * omega[:, None]
                    * (rule.x[None, :] - rule.x[-1])
                    / speed
                )
                height = (
                    h_e[:, region, None] * phase
                    - np.exp(
                        1j * omega[:, None] * rule.x[None, :] / speed
                    )
                    * integral
                    / speed
                )
                output[
                    :, region, rule.latitude_index, rule.grid_indices
                ] = height[:, rule.grid_nodes]
        return output

    @staticmethod
    def _interpolation_rule(
        coordinate: np.ndarray, value: float
    ) -> tuple[int, int, float]:
        """Return the two indices and weight for linear interpolation."""
        if value <= coordinate[0]:
            return 0, 0, 0.0
        if value >= coordinate[-1]:
            last = coordinate.size - 1
            return last, last, 0.0
        high = int(np.searchsorted(coordinate, value, side="right"))
        low = high - 1
        weight = (value - coordinate[low]) / (
            coordinate[high] - coordinate[low]
        )
        return low, high, float(weight)

    @staticmethod
    def _field_on_zonal_grid(
        field: np.ndarray, rule: _ZonalRule
    ) -> np.ndarray:
        """Interpolate both boundaries and retain interior grid values."""
        interior = field[:, rule.interior]
        if isinstance(field, da.Array):
            boundaries = []
            for low, high, weight in (rule.west, rule.east):
                boundaries.append(
                    field[:, low]
                    + weight * (field[:, high] - field[:, low])
                )
            return da.concatenate(
                (boundaries[0][:, None], interior, boundaries[1][:, None]),
                axis=1,
            )
        output = np.empty(
            (field.shape[0], interior.shape[1] + 2), dtype=field.dtype
        )
        output[:, 1:-1] = interior
        for column, (low, high, weight) in (
            (0, rule.west),
            (-1, rule.east),
        ):
            output[:, column] = (
                field[:, low]
                + weight * (field[:, high] - field[:, low])
            )
        return output

    @staticmethod
    def _regional_ekman_transport(
        my: np.ndarray, quadrature, latitude_size: int
    ) -> np.ndarray:
        """Integrate northward Ekman transport across each region."""
        if isinstance(my, da.Array):
            # Every zonal quadrature spans a complete basin width. Rechunking
            # longitude once lets each delayed latitude block perform all five
            # regional reductions without constructing one graph layer per row.
            zonal = my.rechunk({0: -1, 2: -1})
            latitude_chunks = zonal.chunks[1]
            source_blocks = zonal.to_delayed().reshape(
                1, len(latitude_chunks), 1
            )[0, :, 0]
            output_blocks = []
            latitude_start = 0
            for source, latitude_count in zip(
                source_blocks, latitude_chunks
            ):
                block = delayed(_regional_ekman_block)(
                    source,
                    latitude_start,
                    quadrature,
                )
                output_blocks.append(
                    da.from_delayed(
                        block,
                        shape=(my.shape[0], len(_REGIONS), latitude_count),
                        dtype=np.result_type(my.dtype, np.float64),
                    )
                )
                latitude_start += latitude_count
            return da.concatenate(output_blocks, axis=2)
        output = np.zeros(
            (my.shape[0], len(_REGIONS), latitude_size),
            dtype=np.result_type(my.dtype, np.float64),
        )
        for region, rules in enumerate(quadrature):
            for rule in rules:
                field = GlobalRossbyModel._field_on_zonal_grid(
                    my[:, rule.latitude_index, :], rule
                )
                output[:, region, rule.latitude_index] = np.sum(
                    field * rule.weights[None, :], axis=1
                )
        return output

    @staticmethod
    def _zonal_terms(w, omega, c, rules):
        """Integrate frequency-dependent regional wind forcing."""
        if isinstance(w, da.Array):
            q_f = []
            source = []
            dtype = np.result_type(w.dtype, np.complex64)
            for rule in rules:
                field = GlobalRossbyModel._field_on_zonal_grid(
                    w[:, rule.latitude_index, :], rule
                )
                weighted = field * rule.weights[None, :]
                phase = np.exp(
                    1j
                    * omega[:, None]
                    * (rule.x[0] - rule.x[None, :])
                    / c[rule.latitude_index]
                ).astype(dtype)
                unphased = da.sum(weighted, axis=1)
                phased = da.sum(phase * weighted, axis=1)
                q_f.append(phased - unphased)
                source.append(phased / c[rule.latitude_index])
            return da.stack(q_f, axis=1), da.stack(source, axis=1)
        q_f = np.empty((omega.size, len(rules)), dtype=complex)
        source = np.empty_like(q_f)
        for j, rule in enumerate(rules):
            field = GlobalRossbyModel._field_on_zonal_grid(
                w[:, rule.latitude_index, :], rule
            )
            weighted = field * rule.weights[None, :]
            phase = np.exp(
                1j
                * omega[:, None]
                * (rule.x[0] - rule.x[None, :])
                / c[rule.latitude_index]
            )
            unphased = np.sum(weighted, axis=1)
            phased = np.sum(phase * weighted, axis=1)
            q_f[:, j] = phased - unphased
            source[:, j] = phased / c[rule.latitude_index]
        return q_f, source

    @staticmethod
    def _zonal_terms_lazy(
        w: da.Array,
        omega: np.ndarray,
        c: np.ndarray,
        quadrature,
        latitude_size: int,
    ) -> tuple[da.Array, da.Array]:
        """Return lazy zonal budget terms on ``(omega, region, latitude)``.

        Longitude is rechunked to one block because every regional integral
        spans from its interpolated western boundary to its eastern boundary.
        Each delayed task handles all regions for a latitude block, avoiding
        thousands of nearly identical row-level graph layers.
        """
        zonal = w.rechunk({0: -1, 2: -1})
        latitude_chunks = zonal.chunks[1]
        source_blocks = zonal.to_delayed().reshape(
            1, len(latitude_chunks), 1
        )[0, :, 0]
        output_blocks = []
        latitude_start = 0
        dtype = np.result_type(w.dtype, np.float64, np.complex64)
        for source, latitude_count in zip(
            source_blocks, latitude_chunks
        ):
            block = delayed(_zonal_terms_spectral_block)(
                source,
                latitude_start,
                quadrature,
                omega,
                c,
            )
            output_blocks.append(
                da.from_delayed(
                    block,
                    shape=(
                        2,
                        omega.size,
                        len(_REGIONS),
                        latitude_count,
                    ),
                    dtype=dtype,
                )
            )
            latitude_start += latitude_count
        terms = da.concatenate(output_blocks, axis=3)
        if terms.shape[3] != latitude_size:
            raise RuntimeError("lazy zonal-term latitude shape is inconsistent")
        return terms[0], terms[1]

    @staticmethod
    def _solve_boundaries(omega, f_term, r, t_n, t_i, t_p, t_s, k_i, k_p):
        """Solve the three-basin boundary-thickness system at each frequency."""
        h = np.zeros((omega.size, 3), dtype=complex)
        rhs = np.column_stack(
            (
                f_term[:, 0] + f_term[:, 3] + f_term[:, 4] + t_n + t_i + t_p - t_s,
                f_term[:, 1] - t_i,
                f_term[:, 2] - t_p,
            )
        )
        zero = np.isclose(omega, 0.0, rtol=0.0, atol=np.finfo(float).eps)
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(rhs))))
        if np.any(np.abs(rhs[zero]) > tolerance):
            raise ValueError("zero-frequency forcing must vanish")
        for n in np.flatnonzero(~zero):
            matrix = np.array(
                [
                    [r[n, 0] + k_i, r[n, 3] + k_p - k_i, r[n, 4] - k_p],
                    [-k_i, r[n, 1] + k_i, 0.0],
                    [0.0, -k_p, r[n, 2] + k_p],
                ],
                dtype=complex,
            )
            h[n] = np.linalg.solve(matrix, rhs[n])
        return h


def _regional_ekman_block(
    my_values: np.ndarray,
    latitude_start: int,
    quadrature,
) -> np.ndarray:
    """Integrate Ekman transport for one complete-longitude latitude block.

    Parameters
    ----------
    my_values
        Northward Ekman transport on ``(time, block_latitude, longitude)``.
        Longitude is complete; latitude is one Dask chunk.
    latitude_start
        Global index of the block's first latitude.
    quadrature
        Precomputed rules for every model region.

    Returns
    -------
    numpy.ndarray
        Regional integrals on ``(time, region, block_latitude)``. Rows outside
        a region are zero here and are masked after the spectral solve.
    """
    latitude_count = my_values.shape[1]
    output = np.zeros(
        (my_values.shape[0], len(_REGIONS), latitude_count),
        dtype=np.result_type(my_values.dtype, np.float64),
    )
    latitude_stop = latitude_start + latitude_count
    for region, rules in enumerate(quadrature):
        for rule in rules:
            if not latitude_start <= rule.latitude_index < latitude_stop:
                continue
            local_latitude = rule.latitude_index - latitude_start
            field = GlobalRossbyModel._field_on_zonal_grid(
                my_values[:, local_latitude, :], rule
            )
            output[:, region, local_latitude] = np.sum(
                field * rule.weights[None, :], axis=1
            )
    return output


def _zonal_terms_spectral_block(
    w_values: np.ndarray,
    latitude_start: int,
    quadrature,
    omega: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Evaluate both zonal forcing terms for one spectral latitude block.

    The task receives complete-longitude ``w_Ek`` spectra and evaluates all
    regional quadratures that intersect its latitude chunk. Returning both
    the transport-budget term and height-source term together prevents Dask
    from reading and interpolating the same forcing block twice.

    Returns
    -------
    numpy.ndarray
        Array on ``(term, omega, region, block_latitude)`` where ``term=0`` is
        ``q_f`` and ``term=1`` is the height source. Inactive entries are NaN.
    """
    latitude_count = w_values.shape[1]
    dtype = np.result_type(w_values.dtype, np.float64, np.complex64)
    output = np.full(
        (2, omega.size, len(_REGIONS), latitude_count),
        np.nan + 0j,
        dtype=dtype,
    )
    latitude_stop = latitude_start + latitude_count
    for region, rules in enumerate(quadrature):
        for rule in rules:
            if not latitude_start <= rule.latitude_index < latitude_stop:
                continue
            local_latitude = rule.latitude_index - latitude_start
            field = GlobalRossbyModel._field_on_zonal_grid(
                w_values[:, local_latitude, :], rule
            )
            weighted = field * rule.weights[None, :]
            phase = np.exp(
                1j
                * omega[:, None]
                * (rule.x[0] - rule.x[None, :])
                / c[rule.latitude_index]
            )
            unphased = np.sum(weighted, axis=1)
            phased = np.sum(phase * weighted, axis=1)
            output[0, :, region, local_latitude] = phased - unphased
            output[1, :, region, local_latitude] = (
                phased / c[rule.latitude_index]
            )
    return output


def _height_spectral_block(
    w_values: np.ndarray,
    h_e_values: np.ndarray,
    region: int,
    latitude_start: int,
    rules: list[_ZonalRule],
    omega: np.ndarray,
    c: np.ndarray,
    longitude_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Evaluate dense height spectra for one region and latitude block.

    ``w_values`` contains complete longitude rows, which are required by the
    cumulative integral from each evaluation point to the eastern boundary.
    Only active regional cells are filled; all other longitudes and inactive
    latitudes remain NaN. The spatial forcing precision controls ``dtype`` so
    native float32 forcing produces complex64 spectra and float32 time output.
    """
    dtype = np.dtype(dtype)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    latitude_count = w_values.shape[1]
    output = np.full(
        (omega.size, latitude_count, longitude_size),
        np.nan + 0j,
        dtype=dtype,
    )
    latitude_stop = latitude_start + latitude_count
    for rule in rules:
        if not latitude_start <= rule.latitude_index < latitude_stop:
            continue
        local_latitude = rule.latitude_index - latitude_start
        field = GlobalRossbyModel._field_on_zonal_grid(
            w_values[:, local_latitude, :], rule
        ).astype(dtype, copy=False)
        speed = real_dtype.type(c[rule.latitude_index])

        # Factoring exp(i*omega*x/c) makes all point-to-east integrals one
        # reverse cumulative sum rather than a separate integral per point.
        phase_basis = np.exp(
            -1j * omega[:, None] * rule.x[None, :] / speed
        ).astype(dtype)
        integrand = field * phase_basis
        increments = (
            0.5
            * (integrand[:, :-1] + integrand[:, 1:])
            * np.diff(rule.x).astype(real_dtype)[None, :]
        )
        integral = np.zeros_like(integrand, dtype=dtype)
        integral[:, :-1] = np.cumsum(
            increments[:, ::-1], axis=1
        )[:, ::-1]
        phase = np.exp(
            1j
            * omega[:, None]
            * (rule.x[None, :] - rule.x[-1])
            / speed
        ).astype(dtype)
        height = (
            h_e_values[:, region, None].astype(dtype) * phase
            - np.exp(
                1j * omega[:, None] * rule.x[None, :] / speed
            ).astype(dtype)
            * integral
            / speed
        )
        output[:, local_latitude, rule.grid_indices] = height[
            :, rule.grid_nodes
        ]
    return output


def _integral_to_north(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Trapezoidal integral from each coordinate to the northern endpoint."""
    if isinstance(values, da.Array):
        values = values.rechunk({1: -1})
        increments = (
            0.5
            * (values[:, :-1] + values[:, 1:])
            * np.diff(coordinate)[None, :]
        )
        return da.concatenate(
            (
                da.cumsum(increments[:, ::-1], axis=1)[:, ::-1],
                da.zeros(
                    (values.shape[0], 1),
                    chunks=(values.chunks[0], 1),
                    dtype=np.result_type(values.dtype, np.complex64),
                ),
            ),
            axis=1,
        )
    result = np.zeros_like(values, dtype=complex)
    increments = 0.5 * (values[:, :-1] + values[:, 1:]) * np.diff(coordinate)[None, :]
    result[:, :-1] = np.cumsum(increments[:, ::-1], axis=1)[:, ::-1]
    return result


def _pad_active_latitudes(
    values: da.Array, indices: np.ndarray, latitude_size: int
) -> da.Array:
    """Pad one contiguous active-latitude array with lazy NaNs."""
    south = int(indices[0])
    north = latitude_size - int(indices[-1]) - 1
    return da.pad(
        values,
        ((0, 0), (south, north)),
        mode="constant",
        constant_values=np.nan,
    )


def rossby_speed(
    latitude: np.ndarray, g_prime: float, depth: float
) -> np.ndarray:
    """Return the capped first-mode long Rossby-wave speed in m s-1."""
    latitude_rad = np.deg2rad(latitude)
    f = 2.0 * EARTH_ROTATION_S * np.sin(latitude_rad)
    beta = (
        2.0 * EARTH_ROTATION_S * np.cos(latitude_rad) / EARTH_RADIUS_M
    )
    gh = g_prime * depth
    long_wave = np.divide(
        beta * gh,
        f * f,
        out=np.full_like(f, np.inf),
        where=f != 0,
    )
    return np.minimum(long_wave, np.sqrt(gh) / 3.0)
