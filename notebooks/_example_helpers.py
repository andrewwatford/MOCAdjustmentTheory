"""Upstream data preparation and plotting used only by worked notebooks.

These helpers are deliberately outside ``moc_adjustment_theory``.  In
particular, reference density and wind-stress regularization are choices made
by the notebook author before constructing the model forcing.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import xarray as xr

from moc_adjustment_theory import (
    EARTH_RADIUS,
    EARTH_ROTATION_RATE,
    MultiBasinGeometry,
)


def f(latitude: xr.DataArray | np.ndarray | float) -> object:
    """Coriolis parameter in inverse seconds."""

    return 2.0 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(latitude))


def beta(latitude: float) -> float:
    """Meridional Coriolis gradient in inverse metres per second."""

    return (
        2.0
        * EARTH_ROTATION_RATE
        * np.cos(np.deg2rad(latitude))
        / EARTH_RADIUS
    )


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


def ekman_transport_from_stress(
    stress: xr.Dataset,
    *,
    rho_0: float,
    gamma: float,
    tau_x: str = "avg_iews",
    tau_y: str = "avg_inss",
) -> xr.Dataset:
    """Convert stress to regularized Ekman transport on the full wind grid."""

    inverse_f = f(stress.latitude) / (f(stress.latitude) ** 2 + gamma**2)
    result = xr.Dataset(
        {
            "M_Ek_x": stress[tau_y] * inverse_f / rho_0,
            "M_Ek_y": -stress[tau_x] * inverse_f / rho_0,
        }
    )
    result.M_Ek_x.attrs.update(units="m2 s-1", positive="eastward")
    result.M_Ek_y.attrs.update(units="m2 s-1", positive="northward")
    result.attrs.update(
        rho_0_used_upstream=float(rho_0),
        gamma_used_upstream=float(gamma),
    )
    return result


def section_transport(
    M_Ek_y: xr.DataArray,
    geometry: MultiBasinGeometry,
    *,
    region: str,
    latitude: float,
) -> xr.DataArray:
    """Integrate northward vector transport across one geometry section."""

    section = M_Ek_y.interp(latitude=latitude)
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
        EARTH_RADIUS * np.cos(np.deg2rad(latitude)) * np.pi / 180.0
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
