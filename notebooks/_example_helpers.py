"""Upstream data preparation and plotting used only by worked notebooks.

These helpers are deliberately outside ``moc_adjustment_theory``.  In
particular, reference density, wind-stress regularization, and coastal tapering
are choices made by the notebook author before constructing ``GlobalForcing``.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import xarray as xr

from moc_adjustment_theory import MultiBasinGeometry


R_EARTH = 6.371e6
OMEGA = 7.292115e-5

TRACE_VARIABLES = {
    "atlantic_west": "x_wA",
    "atlantic_east": "x_eA",
    "indian_west": "x_wI",
    "indian_east": "x_eI",
    "pacific_west": "x_wP",
    "pacific_east": "x_eP",
}


def f(latitude: xr.DataArray | np.ndarray | float) -> object:
    """Coriolis parameter in inverse seconds."""

    return 2.0 * OMEGA * np.sin(np.deg2rad(latitude))


def beta(latitude: float) -> float:
    """Meridional Coriolis gradient in inverse metres per second."""

    return 2.0 * OMEGA * np.cos(np.deg2rad(latitude)) / R_EARTH


def regularization_gamma(g_prime: float, H: float) -> float:
    """Match the inverse-f regularization to the capped Rossby speed."""

    gravity_wave_cap = np.sqrt(g_prime * H) / 3.0
    cap_latitude = scipy.optimize.brentq(
        lambda latitude: beta(latitude) * g_prime * H / f(latitude) ** 2
        - gravity_wave_cap,
        1.0,
        30.0,
    )
    return float(f(cap_latitude))


def non_itf_geometry(
    isobaths: xr.Dataset,
    *,
    y_S: float,
    y_P: float,
    y_I: float,
    y_N: float,
    y_NI: float,
    y_NP: float,
) -> MultiBasinGeometry:
    """Construct the explicit five-region, no-throughflow geometry."""

    definitions = {
        "atlantic_north": {
            "west": "atlantic_west",
            "east": "atlantic_east",
            "south": y_I,
            "north": y_N,
        },
        "indian_north": {
            "west": "indian_west",
            "east": "indian_east",
            "south": y_I,
            "north": y_NI,
        },
        "pacific_north": {
            "west": "pacific_west",
            "east": "pacific_east",
            "south": y_P,
            "north": y_NP,
        },
        "atlantic_indian_transition": {
            "west": "atlantic_west",
            "east": "indian_east",
            "south": y_P,
            "north": y_I,
        },
        "atlantic_pacific_transition": {
            "west": "atlantic_west",
            "east": "pacific_east",
            "south": y_S,
            "north": y_P,
        },
    }
    return MultiBasinGeometry.from_isobath_dataset(
        isobaths,
        trace_variables=TRACE_VARIABLES,
        region_definitions=definitions,
    )


def smooth_ramp(distance: np.ndarray, width_degrees: float = 2.0) -> np.ndarray:
    """C-infinity ramp from zero to one over a positive distance."""

    scaled = np.asarray(distance, dtype=float) / width_degrees
    left = np.zeros_like(scaled)
    right = np.zeros_like(scaled)
    positive_left = scaled > 0.0
    positive_right = (1.0 - scaled) > 0.0
    left[positive_left] = np.exp(-1.0 / scaled[positive_left])
    right[positive_right] = np.exp(-1.0 / (1.0 - scaled[positive_right]))
    transition = (scaled > 0.0) & (scaled < 1.0)
    result = np.zeros_like(scaled)
    result[transition] = left[transition] / (
        left[transition] + right[transition]
    )
    result[scaled >= 1.0] = 1.0
    return result


def cyclic_zonal_taper(
    latitude: np.ndarray,
    longitude: np.ndarray,
    west: np.ndarray,
    east: np.ndarray,
    *,
    width_degrees: float,
) -> xr.DataArray:
    """Evaluate one basin taper while retaining a unique 0--360 grid."""

    center = float(np.median(0.5 * (west + east)))
    continuous_longitude = (
        longitude[None, :] - center + 180.0
    ) % 360.0 + center - 180.0
    values = smooth_ramp(
        continuous_longitude - west[:, None], width_degrees
    ) * smooth_ramp(east[:, None] - continuous_longitude, width_degrees)
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )


def non_itf_taper(
    geometry: MultiBasinGeometry,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    *,
    width_degrees: float = 2.0,
) -> xr.DataArray:
    """One continuous, no-ITF ocean taper on a supplied forcing grid."""

    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    bounds = geometry.boundaries_on(lat)
    y_I = float(
        geometry.dataset.region_south.sel(region="atlantic_north")
    )
    y_P = float(
        geometry.dataset.region_south.sel(region="pacific_north")
    )
    taper = xr.DataArray(
        np.zeros((lat.size, lon.size)),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )

    for region, selector in (
        ("atlantic_north", lat >= y_I),
        ("atlantic_indian_transition", (lat >= y_P) & (lat < y_I)),
        ("atlantic_pacific_transition", lat < y_P),
    ):
        region_bounds = bounds.sel(region=region)
        taper.loc[{"latitude": lat[selector]}] += cyclic_zonal_taper(
            lat[selector],
            lon,
            np.asarray(region_bounds.x_b)[selector],
            np.asarray(region_bounds.x_e)[selector],
            width_degrees=width_degrees,
        )

    for region in ("indian_north", "pacific_north"):
        region_bounds = bounds.sel(region=region)
        south = float(geometry.dataset.region_south.sel(region=region))
        north = float(geometry.dataset.region_north.sel(region=region))
        selector = (lat >= south) & (lat <= north)
        closed_basin = cyclic_zonal_taper(
            lat[selector],
            lon,
            np.asarray(region_bounds.x_b)[selector],
            np.asarray(region_bounds.x_e)[selector],
            width_degrees=width_degrees,
        )
        closed_basin = closed_basin * xr.DataArray(
            smooth_ramp(north - lat[selector], width_degrees),
            dims="latitude",
            coords={"latitude": lat[selector]},
        )
        taper.loc[{"latitude": lat[selector]}] += closed_basin
    return taper.clip(max=1.0)


def ekman_transport_from_stress(
    stress: xr.Dataset,
    geometry: MultiBasinGeometry,
    *,
    rho_0: float,
    gamma: float,
    width_degrees: float = 2.0,
    tau_x: str = "avg_iews",
    tau_y: str = "avg_inss",
) -> xr.Dataset:
    """Apply an explicitly chosen upstream stress-to-transport conversion."""

    taper = non_itf_taper(
        geometry,
        stress.latitude,
        stress.longitude,
        width_degrees=width_degrees,
    )
    inverse_f = f(stress.latitude) / (f(stress.latitude) ** 2 + gamma**2)
    result = xr.Dataset(
        {
            "M_ek_x": stress[tau_y] * taper * inverse_f / rho_0,
            "M_ek_y": -stress[tau_x] * taper * inverse_f / rho_0,
        }
    )
    result.M_ek_x.attrs.update(units="m2 s-1", positive="eastward")
    result.M_ek_y.attrs.update(units="m2 s-1", positive="northward")
    result.attrs.update(
        rho_0_used_upstream=float(rho_0),
        gamma_used_upstream=float(gamma),
        taper_width_degrees=float(width_degrees),
    )
    return result


def section_transport(
    M_ek_y: xr.DataArray,
    geometry: MultiBasinGeometry,
    *,
    region: str,
    latitude: float,
) -> xr.DataArray:
    """Integrate northward vector transport across one geometry section."""

    section = M_ek_y.interp(latitude=latitude)
    target = np.array([latitude, latitude + 1e-6])
    bounds = geometry.boundaries_on(target).sel(region=region).isel(latitude=0)
    west = float(bounds.x_b)
    east = float(bounds.x_e)
    center = 0.5 * (west + east)
    continuous = (
        np.asarray(section.longitude) - center + 180.0
    ) % 360.0 + center - 180.0
    order = np.argsort(continuous)
    section = section.isel(longitude=order).assign_coords(
        longitude=continuous[order]
    )
    result = section.where(
        (section.longitude >= west) & (section.longitude <= east),
        0.0,
    ).integrate("longitude")
    result = result * (
        R_EARTH * np.cos(np.deg2rad(latitude)) * np.pi / 180.0
    )
    result.attrs.update(units="m3 s-1", positive="northward")
    return result


def stitched_atlantic(
    array: xr.DataArray,
    geometry: MultiBasinGeometry,
    *,
    target_latitude: xr.DataArray | None = None,
) -> xr.DataArray:
    """Join regions 5, 4, and 1 into the step-shaped Atlantic path."""

    y_I = float(
        geometry.dataset.region_south.sel(region="atlantic_north")
    )
    y_P = float(
        geometry.dataset.region_south.sel(region="pacific_north")
    )
    if target_latitude is None:
        y_S = float(
            geometry.dataset.region_south.sel(
                region="atlantic_pacific_transition"
            )
        )
        y_N = float(
            geometry.dataset.region_north.sel(region="atlantic_north")
        )
        target_latitude = array.latitude.where(
            (array.latitude >= y_S) & (array.latitude <= y_N),
            drop=True,
        )
    pieces = []
    for region, selector in (
        ("atlantic_pacific_transition", target_latitude < y_P),
        (
            "atlantic_indian_transition",
            (target_latitude >= y_P) & (target_latitude < y_I),
        ),
        ("atlantic_north", target_latitude >= y_I),
    ):
        source = array.sel(region=region).dropna("latitude", how="all")
        target = target_latitude.where(selector, drop=True)
        pieces.append(source.interp(latitude=target))
    result = xr.concat(
        pieces,
        dim="latitude",
        coords="minimal",
        compat="override",
    ).sortby("latitude")
    return result.drop_vars("region", errors="ignore")


def plot_geometry(
    geometry: MultiBasinGeometry,
    *,
    ax: plt.Axes | None = None,
    colors: Mapping[str, str] | None = None,
) -> plt.Axes:
    """Small domain schematic based directly on the configured boundaries."""

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    if colors is None:
        colors = {
            "atlantic_north": "#4477AA",
            "indian_north": "#EE6677",
            "pacific_north": "#228833",
            "atlantic_indian_transition": "#CCBB44",
            "atlantic_pacific_transition": "#66CCEE",
        }
    for region in geometry.dataset.region.values:
        west = geometry.x_b.sel(region=region).dropna("latitude")
        east = geometry.x_e.sel(region=region).dropna("latitude")
        ax.fill_betweenx(
            west.latitude,
            west,
            east,
            color=colors[str(region)],
            alpha=0.28,
            label=str(region).replace("_", " "),
        )
        ax.plot(west, west.latitude, color=colors[str(region)], linewidth=1.1)
        ax.plot(east, east.latitude, color=colors[str(region)], linewidth=1.1)
    ax.set(xlabel="continuous longitude (degrees east)", ylabel="latitude")
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper right")
    return ax
