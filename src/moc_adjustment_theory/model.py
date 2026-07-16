"""Frequency-domain global adjustment model."""

from __future__ import annotations

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0
EARTH_ROTATION_S = 7.292115e-5

_REGIONS = (
    ("north_atlantic", "x_wA", "x_eA"),
    ("north_indian", "x_wI", "x_eI"),
    ("north_pacific", "x_wP", "x_eP"),
    ("atlantic_indian", "x_wA", "x_eI"),
    ("atlantic_pacific", "x_wA", "x_eP"),
)


class GlobalAdjustmentModel:
    """Solve the global reduced-gravity adjustment theory in frequency space.

    Parameters
    ----------
    isobath_ds
        Active-layer geometry described by the six latitude-dependent boundary
        variables in the package specification.
    g_prime
        Reduced gravity in m s-2.

    Notes
    -----
    ``solve_frequency`` is the frequency-domain kernel. It deliberately does
    not transform time series; the module-level Fourier transform functions
    own that convention.
    """

    def __init__(self, isobath_ds: xr.Dataset, g_prime: float):
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

    def solve_frequency(self, forcing_hat: xr.Dataset) -> xr.Dataset:
        """Solve for spectral thickness and transport anomalies.

        ``forcing_hat`` must contain complex Fourier coefficients ``M_Ek_x``
        and ``M_Ek_y`` on ``(omega, latitude, longitude)`` and ``T_N`` on
        ``omega``. Coordinate values are angular frequencies in rad s-1.
        A zero-frequency coefficient may be present, but its forcing must be
        zero; its thickness gauge is set to zero explicitly.

        Returns
        -------
        xarray.Dataset
            ``h_e`` on ``(omega, region)`` and ``h_b``, ``h_w``, ``T``,
            ``T_g`` and ``T_Ek`` on ``(omega, region, latitude)``. Values
            outside a region's latitude support are NaN.
        """
        omega, latitude, longitude, mx, my, t_n = self._forcing_arrays(forcing_hat)
        geometry = self._geometry(latitude)
        lat_rad = np.deg2rad(latitude)
        lon_rad = np.deg2rad(longitude)
        cos_lat = np.cos(lat_rad)
        f = 2.0 * EARTH_ROTATION_S * np.sin(lat_rad)
        beta = 2.0 * EARTH_ROTATION_S * cos_lat / EARTH_RADIUS_M
        depth = float(self.isobath_ds.attrs["isobath_depth_m"])
        gh = self.g_prime * depth
        long_wave = np.divide(
            beta * gh,
            f * f,
            out=np.full_like(f, np.inf),
            where=f != 0,
        )
        c = np.minimum(long_wave, np.sqrt(gh) / 3.0)

        dmx_dlon = np.gradient(mx, lon_rad, axis=2, edge_order=2)
        dmycos_dlat = np.gradient(my * cos_lat[None, :, None], lat_rad, axis=1, edge_order=2)
        w_ek = (dmx_dlon + dmycos_dlat) / (
            EARTH_RADIUS_M * cos_lat[None, :, None]
        )

        shape = (omega.size, len(_REGIONS), latitude.size)
        f_partial = np.full(shape, np.nan + 0j)
        r_partial = np.full(shape, np.nan + 0j)
        h_source = np.full(shape, np.nan + 0j)
        t_ek = np.full(shape, np.nan + 0j)
        propagation = np.full(shape, np.nan + 0j)

        for region, (_, _, _) in enumerate(_REGIONS):
            indices, x_w, x_e = geometry[region]
            q_f, source, ek = self._zonal_terms(
                w_ek[:, indices, :],
                my[:, indices, :],
                longitude,
                latitude[indices],
                x_w,
                x_e,
                omega,
                c[indices],
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
            t_ek[:, region, indices] = ek
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
        coords = {
            "omega": forcing_hat.omega,
            "region": [name for name, _, _ in _REGIONS],
            "latitude": forcing_hat.latitude,
        }
        result = xr.Dataset(
            {
                "h_e": (("omega", "region"), h_e),
                "h_b": (("omega", "region", "latitude"), h_b),
                "h_w": (("omega", "region", "latitude"), h_w),
                "T": (("omega", "region", "latitude"), transport),
                "T_g": (("omega", "region", "latitude"), t_g),
                "T_Ek": (("omega", "region", "latitude"), t_ek),
            },
            coords=coords,
        )
        for name in ("h_e", "h_b", "h_w"):
            result[name].attrs["units"] = "m"
        for name in ("T", "T_g", "T_Ek"):
            result[name].attrs["units"] = "m3 s-1"
        result.omega.attrs.setdefault("units", "rad s-1")
        return result

    @staticmethod
    def _forcing_arrays(forcing_hat: xr.Dataset):
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
            forcing_hat.M_Ek_x.transpose("omega", "latitude", "longitude").values,
            dtype=complex,
        )
        my = np.asarray(
            forcing_hat.M_Ek_y.transpose("omega", "latitude", "longitude").values,
            dtype=complex,
        )
        t_n = np.asarray(forcing_hat.T_N.transpose("omega").values, dtype=complex)
        return omega, latitude, longitude, mx, my, t_n

    def _geometry(self, latitude: np.ndarray):
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
        limits = (
            (y_i, min(support["x_wA"][1], support["x_eA"][1])),
            (y_i, min(support["x_wI"][1], support["x_eI"][1])),
            (y_p, min(support["x_wP"][1], support["x_eP"][1])),
            (y_p, y_i),
            (y_s, y_p),
        )
        def ceiling(value):
            candidates = latitude[latitude >= value]
            if candidates.size == 0:
                raise ValueError("forcing latitude does not cover the active domain")
            return candidates[0]

        def floor(value):
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
    def _zonal_terms(w, my, longitude, latitude, x_w, x_e, omega, c):
        n_omega, n_latitude, _ = w.shape
        q_f = np.empty((n_omega, n_latitude), dtype=complex)
        source = np.empty_like(q_f)
        ek = np.empty_like(q_f)
        for j in range(n_latitude):
            inside = longitude[(longitude > x_w[j]) & (longitude < x_e[j])]
            x_deg = np.concatenate(([x_w[j]], inside, [x_e[j]]))
            if np.any(np.diff(x_deg) <= 0):
                raise ValueError("each eastern boundary must lie east of its western boundary")
            field_w = np.vstack([np.interp(x_deg, longitude, row) for row in w[:, j]])
            field_my = np.vstack([np.interp(x_deg, longitude, row) for row in my[:, j]])
            x = EARTH_RADIUS_M * np.cos(np.deg2rad(latitude[j])) * np.deg2rad(x_deg)
            phase = np.exp(1j * omega[:, None] * (x[0] - x[None, :]) / c[j])
            dx = np.diff(x)[None, :]
            f_integrand = (phase - 1.0) * field_w
            h_integrand = phase * field_w / c[j]
            q_f[:, j] = np.sum(0.5 * (f_integrand[:, :-1] + f_integrand[:, 1:]) * dx, axis=1)
            source[:, j] = np.sum(0.5 * (h_integrand[:, :-1] + h_integrand[:, 1:]) * dx, axis=1)
            ek[:, j] = np.sum(0.5 * (field_my[:, :-1] + field_my[:, 1:]) * dx, axis=1)
        return q_f, source, ek

    @staticmethod
    def _solve_boundaries(omega, f_term, r, t_n, t_i, t_p, t_s, k_i, k_p):
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


def _integral_to_north(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Trapezoidal integral from each coordinate to the northern endpoint."""
    result = np.zeros_like(values, dtype=complex)
    increments = 0.5 * (values[:, :-1] + values[:, 1:]) * np.diff(coordinate)[None, :]
    result[:, :-1] = np.cumsum(increments[:, ::-1], axis=1)[:, ::-1]
    return result
