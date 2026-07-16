"""Fixed five-region global adjustment solve."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import dask.array as dsa
import numpy as np
import xarray as xr
from scipy import integrate

from .forcing import GlobalForcing
from .geometry import MultiBasinGeometry, REGION_KEYS
from .output import GlobalAdjustmentOutput


EARTH_RADIUS = 6.371e6
EARTH_ROTATION_RATE = 7.292115e-5


def _coriolis(latitude: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
    return 2.0 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(latitude))


def _beta(latitude: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
    return (
        2.0
        * EARTH_ROTATION_RATE
        * np.cos(np.deg2rad(latitude))
        / EARTH_RADIUS
    )


def _rossby_speed(
    latitude: np.ndarray, *, g_prime: float, H: float
) -> np.ndarray:
    f = np.asarray(_coriolis(latitude), dtype=float)
    uncapped = np.full_like(f, np.inf)
    np.divide(
        np.asarray(_beta(latitude)) * g_prime * H,
        f**2,
        out=uncapped,
        where=f != 0.0,
    )
    return np.minimum(uncapped, np.sqrt(g_prime * H) / 3.0)


def _continuous_longitude(
    array: xr.DataArray, *, center: float
) -> xr.DataArray:
    longitude = np.asarray(array.longitude, dtype=float)
    continuous = (longitude - center + 180.0) % 360.0 + center - 180.0
    order = np.argsort(continuous)
    continuous = continuous[order]
    if np.any(np.diff(continuous) <= 0):
        raise ValueError(
            "Ekman-transport longitude becomes duplicated in basin coordinates"
        )
    return array.isel(longitude=order).assign_coords(longitude=continuous)


def _cumulative_to_north(
    values: xr.DataArray, *, scale: float = EARTH_RADIUS * np.pi / 180.0
) -> xr.DataArray:
    ordered = values.transpose("omega", "latitude")
    latitude = np.asarray(ordered.latitude, dtype=float)
    data = np.asarray(ordered)
    cumulative = -integrate.cumulative_trapezoid(
        data[:, ::-1],
        x=latitude[::-1] * scale,
        axis=1,
        initial=0.0,
    )[:, ::-1]
    return xr.DataArray(
        cumulative,
        dims=("omega", "latitude"),
        coords={"omega": ordered.omega, "latitude": ordered.latitude},
    )


@dataclass(frozen=True, slots=True)
class GlobalAdjustmentModel:
    """Derive Ekman terms and solve the fixed 3 x 3 system at every frequency."""

    geometry: MultiBasinGeometry
    forcing: GlobalForcing
    g_prime: float = 0.02
    condition_warning: float = 1e10

    def __post_init__(self) -> None:
        if not np.isfinite(self.g_prime) or self.g_prime <= 0:
            raise ValueError("g_prime must be positive and finite")
        if not np.isfinite(self.condition_warning) or self.condition_warning <= 1:
            raise ValueError("condition_warning must be finite and greater than one")

    def _region_terms(self, region: str) -> xr.Dataset:
        bounds = self.geometry.dataset.sel(region=region)
        south = float(bounds.region_south)
        north = float(bounds.region_north)
        native_latitude = np.asarray(self.forcing.spectral.latitude, dtype=float)
        if south < native_latitude[0] or north > native_latitude[-1]:
            raise ValueError(
                f"Ekman-transport grid does not bracket {region!r} latitude support"
            )
        latitude = np.unique(
            np.r_[
                native_latitude[
                    (native_latitude >= south) & (native_latitude <= north)
                ],
                south,
                north,
            ]
        )
        if latitude.size < 3:
            raise ValueError(
                f"Ekman-transport grid has insufficient rows for {region!r}"
            )
        boundary = self.geometry.boundaries_on(latitude).sel(region=region)
        x_b = np.asarray(boundary.x_b, dtype=float)
        x_e = np.asarray(boundary.x_e, dtype=float)
        if not np.all(np.isfinite(x_b)) or not np.all(np.isfinite(x_e)):
            raise ValueError(
                f"geometry is undefined on the {region!r} forcing rows"
            )
        center = float(np.median(0.5 * (x_b + x_e)))

        transport_x = _continuous_longitude(
            self.forcing.spectral.M_ek_x, center=center
        )
        transport_y = _continuous_longitude(
            self.forcing.spectral.M_ek_y, center=center
        )
        longitude = np.asarray(transport_x.longitude, dtype=float)
        spacing = float(np.median(np.diff(longitude)))
        margin = spacing
        west = float(np.min(x_b) - margin)
        east = float(np.max(x_e) + margin)
        if west < longitude[0] or east > longitude[-1]:
            raise ValueError(
                f"Ekman-transport longitude does not bracket the {region!r} boundaries"
            )
        transport_x = transport_x.sel(longitude=slice(west, east)).interp(
            latitude=latitude
        )
        transport_y = transport_y.sel(longitude=slice(west, east)).interp(
            latitude=latitude
        )
        longitude = np.asarray(transport_x.longitude, dtype=float)

        inside = xr.DataArray(
            (longitude[None, :] >= x_b[:, None])
            & (longitude[None, :] <= x_e[:, None]),
            dims=("latitude", "longitude"),
            coords={"latitude": latitude, "longitude": longitude},
        )
        latitude_da = transport_x.latitude
        cosine = np.cos(np.deg2rad(latitude_da))
        w_ek = (180.0 / np.pi) * (
            transport_x.differentiate("longitude")
            + (transport_y * cosine).differentiate("latitude")
        ) / (EARTH_RADIUS * cosine)
        w_ek = w_ek.where(inside, 0.0)
        transport_y_inside = transport_y.where(inside, 0.0)

        c_values = _rossby_speed(
            latitude, g_prime=self.g_prime, H=self.geometry.H
        )
        c = xr.DataArray(c_values, dims="latitude", coords={"latitude": latitude})
        distance = (
            EARTH_RADIUS
            * np.cos(np.deg2rad(latitude))[:, None]
            * np.deg2rad(longitude[None, :] - x_b[:, None])
        )
        travel_time = distance / c_values[:, None]
        omega_values = np.asarray(self.forcing.omega, dtype=float)
        phase_data = dsa.exp(
            -1j
            * dsa.from_array(
                omega_values, chunks=(min(64, omega_values.size),)
            )[:, None, None]
            * dsa.from_array(
                travel_time,
                chunks=(min(64, latitude.size), min(256, longitude.size)),
            )[None, :, :]
        )
        phase = xr.DataArray(
            phase_data,
            dims=("omega", "latitude", "longitude"),
            coords={
                "omega": self.forcing.omega,
                "latitude": latitude,
                "longitude": longitude,
            },
        )
        dx_per_degree = EARTH_RADIUS * np.cos(np.deg2rad(latitude_da)) * np.pi / 180.0
        q = (w_ek * phase).integrate("longitude") * dx_per_degree
        zonal_pumping = w_ek.integrate("longitude") * dx_per_degree
        forcing_density = q - zonal_pumping

        width = (
            EARTH_RADIUS
            * np.cos(np.deg2rad(latitude))
            * np.deg2rad(x_e - x_b)
        )
        eastern_phase = np.exp(
            -1j
            * omega_values[:, None]
            * width[None, :]
            / c_values[None, :]
        )
        eastern_phase = xr.DataArray(
            eastern_phase,
            dims=("omega", "latitude"),
            coords={"omega": self.forcing.omega, "latitude": latitude},
        )
        storage_density = c * (eastern_phase - 1.0)
        meridional_scale = EARTH_RADIUS * np.pi / 180.0
        F = forcing_density.integrate("latitude") * meridional_scale
        r = storage_density.integrate("latitude") * meridional_scale
        transport_ekman = (
            transport_y_inside.integrate("longitude") * dx_per_degree
        )
        pumping_integral = zonal_pumping.integrate("latitude") * meridional_scale
        compatibility = pumping_integral - (
            transport_ekman.sel(latitude=north)
            - transport_ekman.sel(latitude=south)
        )
        return xr.Dataset(
            {
                "F": F,
                "r": r,
                "q": q,
                "forcing_density": forcing_density,
                "storage_density": storage_density,
                "eastern_phase": eastern_phase,
                "transport_ekman": transport_ekman,
                "compatibility_residual": compatibility,
            }
        )

    def solve(self) -> GlobalAdjustmentOutput:
        """Compute all thickness and transport diagnostics in one model run."""

        region_index = xr.IndexVariable("region", list(REGION_KEYS))
        combined_terms = xr.concat(
            [self._region_terms(region) for region in REGION_KEYS],
            dim=region_index,
            join="outer",
            coords="minimal",
            compat="override",
        ).compute()
        terms = {
            region: combined_terms.sel(region=region, drop=True).dropna(
                "latitude", how="all"
            )
            for region in REGION_KEYS
        }
        omega = np.asarray(self.forcing.omega, dtype=float)
        n_omega = omega.size
        r1, r2, r3, r4, r5 = (
            np.asarray(terms[region].r)
            for region in REGION_KEYS
        )
        F1, F2, F3, F4, F5 = (
            np.asarray(terms[region].F)
            for region in REGION_KEYS
        )
        y_I = float(
            self.geometry.dataset.region_south.sel(region="atlantic_north")
        )
        y_P = float(
            self.geometry.dataset.region_south.sel(region="pacific_north")
        )
        kappa_I = self.g_prime * self.geometry.H / float(_coriolis(y_I))
        kappa_P = self.g_prime * self.geometry.H / float(_coriolis(y_P))
        transport_I_ekman = np.asarray(
            terms["indian_north"].transport_ekman.sel(latitude=y_I)
        )
        transport_P_ekman = np.asarray(
            terms["pacific_north"].transport_ekman.sel(latitude=y_P)
        )
        northern = np.asarray(self.forcing.spectral.northern_transport.compute())
        southern = np.asarray(self.forcing.spectral.southern_transport.compute())

        matrix = np.zeros((n_omega, 3, 3), dtype=np.complex128)
        matrix[:, 0, 0] = r1 + kappa_I
        matrix[:, 0, 1] = r4 + kappa_P - kappa_I
        matrix[:, 0, 2] = r5 - kappa_P
        matrix[:, 1, 0] = -kappa_I
        matrix[:, 1, 1] = r2 + kappa_I
        matrix[:, 2, 1] = -kappa_P
        matrix[:, 2, 2] = r3 + kappa_P
        rhs = np.column_stack(
            [
                F1
                + F4
                + F5
                + northern
                + transport_I_ekman
                + transport_P_ekman
                - southern,
                F2 - transport_I_ekman,
                F3 - transport_P_ekman,
            ]
        )
        matrix[0] = 0.0
        rhs[0] = 0.0
        solution = np.zeros((n_omega, 3), dtype=np.complex128)
        condition = np.full(n_omega, np.inf)
        if n_omega > 1:
            condition[1:] = np.linalg.cond(matrix[1:])
            if not np.all(np.isfinite(condition[1:])):
                raise np.linalg.LinAlgError("global model matrix is singular")
            solution[1:] = np.linalg.solve(matrix[1:], rhs[1:, :, None])[:, :, 0]
        maximum_condition = float(np.max(condition[1:], initial=0.0))
        if maximum_condition > self.condition_warning:
            warnings.warn(
                f"global solve reaches condition number {maximum_condition:.3g}",
                RuntimeWarning,
                stacklevel=2,
            )

        h_A, h_I, h_P = solution.T
        h_by_region = {
            "atlantic_north": h_A,
            "indian_north": h_I,
            "pacific_north": h_P,
            "atlantic_indian_transition": h_I,
            "atlantic_pacific_transition": h_P,
        }
        h_e = xr.DataArray(
            np.stack([h_by_region[region] for region in REGION_KEYS], axis=1),
            dims=("omega", "region"),
            coords={"omega": self.forcing.omega, "region": list(REGION_KEYS)},
            attrs={"units": "m"},
        )

        total_profiles: dict[str, xr.DataArray] = {}
        north_transport: dict[str, xr.DataArray] = {
            "atlantic_north": self.forcing.spectral.northern_transport.compute(),
            "indian_north": xr.zeros_like(self.forcing.omega, dtype=complex),
            "pacific_north": xr.zeros_like(self.forcing.omega, dtype=complex),
        }
        southern_transport: dict[str, xr.DataArray] = {}

        def build_transport(region: str) -> None:
            term = terms[region]
            cumulative_F = _cumulative_to_north(term.forcing_density)
            cumulative_r = _cumulative_to_north(term.storage_density)
            he = xr.DataArray(
                h_by_region[region],
                dims="omega",
                coords={"omega": self.forcing.omega},
            )
            profile = north_transport[region] + cumulative_F - cumulative_r * he
            total_profiles[region] = profile
            southern_transport[region] = profile.isel(latitude=0)

        build_transport("atlantic_north")
        build_transport("indian_north")
        build_transport("pacific_north")
        north_transport["atlantic_indian_transition"] = (
            southern_transport["atlantic_north"]
            + southern_transport["indian_north"]
        )
        build_transport("atlantic_indian_transition")
        north_transport["atlantic_pacific_transition"] = (
            southern_transport["atlantic_indian_transition"]
            + southern_transport["pacific_north"]
        )
        build_transport("atlantic_pacific_transition")

        profile_arrays: dict[str, list[xr.DataArray]] = {
            "h_b": [],
            "h_w": [],
            "transport": [],
            "transport_ekman": [],
            "transport_geostrophic": [],
        }
        compatibility_arrays: list[xr.DataArray] = []
        F_arrays: list[xr.DataArray] = []
        r_arrays: list[xr.DataArray] = []
        for region in REGION_KEYS:
            term = terms[region]
            latitude = term.latitude
            he = xr.DataArray(
                h_by_region[region],
                dims="omega",
                coords={"omega": self.forcing.omega},
            )
            h_b = he * term.eastern_phase - term.q / _rossby_speed(
                np.asarray(latitude), g_prime=self.g_prime, H=self.geometry.H
            )
            total = total_profiles[region]
            ekman = term.transport_ekman
            geostrophic = total - ekman
            h_w = he - (
                _coriolis(latitude) * geostrophic / (self.g_prime * self.geometry.H)
            )
            for name, array in (
                ("h_b", h_b),
                ("h_w", h_w),
                ("transport", total),
                ("transport_ekman", ekman),
                ("transport_geostrophic", geostrophic),
            ):
                profile_arrays[name].append(array)
            compatibility_arrays.append(term.compatibility_residual)
            F_arrays.append(term.F)
            r_arrays.append(term.r)

        spectral_variables = {
            name: xr.concat(arrays, dim=region_index, join="outer").transpose(
                "omega", "region", "latitude"
            )
            for name, arrays in profile_arrays.items()
        }
        spectral = xr.Dataset(
            {
                **spectral_variables,
                "h_e": h_e,
                "F": xr.concat(F_arrays, dim=region_index).transpose(
                    "omega", "region"
                ),
                "r": xr.concat(r_arrays, dim=region_index).transpose(
                    "omega", "region"
                ),
                "compatibility_residual": xr.concat(
                    compatibility_arrays, dim=region_index
                ).transpose("omega", "region"),
                "condition_number": ("omega", condition),
                "southern_budget_residual": (
                    "omega",
                    np.asarray(
                        southern_transport["atlantic_pacific_transition"]
                    )
                    - southern,
                ),
            }
        )
        spectral.attrs.update(self.forcing.spectral.attrs)
        spectral.attrs["time_mean_removed"] = bool(
            self.forcing.time_domain.attrs["time_mean_removed"]
        )
        for name in ("h_e", "h_b", "h_w"):
            spectral[name].attrs["units"] = "m"
        for name in (
            "transport",
            "transport_ekman",
            "transport_geostrophic",
            "F",
            "compatibility_residual",
            "southern_budget_residual",
        ):
            spectral[name].attrs["units"] = "m3 s-1"
        spectral.r.attrs["units"] = "m2 s-1"
        spectral.attrs.update(
            g_prime=float(self.g_prime),
            H=float(self.geometry.H),
            earth_radius=EARTH_RADIUS,
            earth_rotation_rate=EARTH_ROTATION_RATE,
        )

        time_variables = {
            name: self.forcing.inverse_transform(spectral[name])
            for name in (
                "h_e",
                "h_b",
                "h_w",
                "transport",
                "transport_ekman",
                "transport_geostrophic",
                "compatibility_residual",
                "southern_budget_residual",
            )
        }
        time_domain = xr.Dataset(time_variables)
        time_domain.attrs.update(spectral.attrs)
        return GlobalAdjustmentOutput(time_domain, spectral)
