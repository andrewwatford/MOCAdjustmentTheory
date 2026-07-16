"""Opt-in regression against the established ERA5 + SCOTIA calculation.

Set ``MOC_REFERENCE_ROOT`` to the legacy reference-data checkout to run this
test. The wind-stress conversion deliberately lives here rather than in
the package: ``GlobalAdjustmentModel`` begins with user-supplied Ekman vector
transport.
"""

from __future__ import annotations

import os
from pathlib import Path

import dask
import numpy as np
import pytest
import scipy.optimize
import xarray as xr

from moc_adjustment_theory import (
    GlobalAdjustmentModel,
    GlobalForcing,
    MultiBasinGeometry,
)


ROOT = Path(os.environ.get("MOC_REFERENCE_ROOT", "__reference_data_missing__"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not ROOT.exists(),
        reason="set MOC_REFERENCE_ROOT to the legacy reference-data checkout",
    ),
]

R_EARTH = 6.371e6
OMEGA = 7.292115e-5
RHO_0 = 1027.0
G_PRIME = 0.02
H = 1000.0

def _f(latitude: xr.DataArray | np.ndarray | float) -> object:
    return 2.0 * OMEGA * np.sin(np.deg2rad(latitude))


def _beta(latitude: float) -> float:
    return 2.0 * OMEGA * np.cos(np.deg2rad(latitude)) / R_EARTH


def _cap_latitude() -> float:
    cap = np.sqrt(G_PRIME * H) / 3.0
    return float(
        scipy.optimize.brentq(
            lambda latitude: _beta(latitude) * G_PRIME * H / _f(latitude) ** 2
            - cap,
            1.0,
            30.0,
        )
    )


def _smooth_ramp(distance: np.ndarray, inverse_width: float = 0.5) -> np.ndarray:
    scaled = inverse_width * distance
    S_left = np.zeros_like(scaled, dtype=float)
    S_right = np.zeros_like(scaled, dtype=float)
    positive_left = scaled > 0.0
    positive_right = (1.0 - scaled) > 0.0
    S_left[positive_left] = np.exp(-1.0 / scaled[positive_left])
    S_right[positive_right] = np.exp(
        -1.0 / (1.0 - scaled[positive_right])
    )
    transition = (scaled > 0.0) & (scaled < 1.0)
    result = np.zeros_like(scaled, dtype=float)
    result[transition] = S_left[transition] / (
        S_left[transition] + S_right[transition]
    )
    result[scaled >= 1.0] = 1.0
    return result


def _zonal_taper(
    latitude: np.ndarray,
    longitude: np.ndarray,
    west: np.ndarray,
    east: np.ndarray,
) -> xr.DataArray:
    center = np.full_like(west, np.median(0.5 * (west + east)))
    continuous_longitude = (
        longitude[None, :] - center[:, None] + 180.0
    ) % 360.0 + center[:, None] - 180.0
    values = _smooth_ramp(continuous_longitude - west[:, None]) * _smooth_ramp(
        east[:, None] - continuous_longitude
    )
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )


def _common_support(dataset: xr.Dataset, *names: str) -> tuple[float, float]:
    traces = [dataset[name].dropna("latitude") for name in names]
    return (
        max(float(trace.latitude[0]) for trace in traces),
        min(float(trace.latitude[-1]) for trace in traces),
    )


def _geometry_and_limits() -> tuple[MultiBasinGeometry, dict[str, float]]:
    isobaths = xr.open_dataset(
        ROOT / "data/isobath/global_isobath_GEBCO_1000m.nc"
    ).dropna("latitude", how="all")
    y_I, y_NI = _common_support(isobaths, "x_wI", "x_eI")
    y_P, _ = _common_support(isobaths, "x_wP", "x_eP")
    y_S, _ = _common_support(isobaths, "x_wA", "x_eP")
    limits = {
        "y_S": float(np.ceil(y_S * 4.0) / 4.0),
        "y_P": y_P,
        "y_I": y_I,
        "y_N": 55.0,
        "y_NI": float(np.floor(y_NI * 4.0) / 4.0),
        "y_NP": 55.0,
    }
    definitions = {
        "atlantic_north": {
            "west": "atlantic_west",
            "east": "atlantic_east",
            "south": limits["y_I"],
            "north": limits["y_N"],
        },
        "indian_north": {
            "west": "indian_west",
            "east": "indian_east",
            "south": limits["y_I"],
            "north": limits["y_NI"],
        },
        "pacific_north": {
            "west": "pacific_west",
            "east": "pacific_east",
            "south": limits["y_P"],
            "north": limits["y_NP"],
        },
        "atlantic_indian_transition": {
            "west": "atlantic_west",
            "east": "indian_east",
            "south": limits["y_P"],
            "north": limits["y_I"],
        },
        "atlantic_pacific_transition": {
            "west": "atlantic_west",
            "east": "pacific_east",
            "south": limits["y_S"],
            "north": limits["y_P"],
        },
    }
    isobaths = isobaths.rename(
        {
            "x_wA": "atlantic_west",
            "x_eA": "atlantic_east",
            "x_wI": "indian_west",
            "x_eI": "indian_east",
            "x_wP": "pacific_west",
            "x_eP": "pacific_east",
        }
    )
    regions = list(definitions)
    isobaths = isobaths.assign_coords(region=regions)
    isobaths["region_west_trace"] = (
        "region",
        [str(definitions[key]["west"]) for key in regions],
    )
    isobaths["region_east_trace"] = (
        "region",
        [str(definitions[key]["east"]) for key in regions],
    )
    isobaths["region_south"] = (
        "region",
        [float(definitions[key]["south"]) for key in regions],
    )
    isobaths["region_north"] = (
        "region",
        [float(definitions[key]["north"]) for key in regions],
    )
    return (
        MultiBasinGeometry.from_isobath_dataset(isobaths),
        limits,
    )


def _forcing(
    geometry: MultiBasinGeometry, limits: dict[str, float]
) -> GlobalForcing:
    scotia = xr.open_dataset(
        ROOT / "data/SCOTIA/SCOTIA_overturning_diagnostics.nc"
    ).moc
    scotia = scotia.assign_coords(time=scotia.time - np.timedelta64(14, "D"))

    winds = xr.open_dataset(
        ROOT / "data/ERA5/global_winds.nc",
        chunks={},
    )[["avg_iews", "avg_inss"]]
    winds = winds.assign_coords(
        valid_time=winds.valid_time - np.timedelta64(6, "h")
    ).rename(valid_time="time")
    winds = winds.sel(latitude=slice(limits["y_N"], limits["y_S"])).sortby(
        "latitude"
    )
    winds = winds.sel(time=scotia.time)
    winds = winds.chunk({"time": -1, "latitude": 24, "longitude": 96})
    winds = winds - winds.mean("time")

    latitude = np.asarray(winds.latitude, dtype=float)
    longitude = np.asarray(winds.longitude, dtype=float)
    bounds = geometry.boundaries_on(latitude)
    y_I = limits["y_I"]
    y_P = limits["y_P"]

    atlantic = xr.zeros_like(
        xr.DataArray(
            np.zeros((latitude.size, longitude.size)),
            dims=("latitude", "longitude"),
            coords={"latitude": latitude, "longitude": longitude},
        )
    )
    for region, selector in (
        ("atlantic_north", latitude >= y_I),
        ("atlantic_indian_transition", (latitude >= y_P) & (latitude < y_I)),
        ("atlantic_pacific_transition", latitude < y_P),
    ):
        region_bounds = bounds.sel(region=region)
        atlantic.loc[{"latitude": latitude[selector]}] = _zonal_taper(
            latitude[selector],
            longitude,
            np.asarray(region_bounds.x_b)[selector],
            np.asarray(region_bounds.x_e)[selector],
        )

    indian_bounds = bounds.sel(region="indian_north")
    indian_selector = (latitude >= y_I) & (latitude <= limits["y_NI"])
    indian = xr.zeros_like(atlantic)
    indian.loc[{"latitude": latitude[indian_selector]}] = _zonal_taper(
        latitude[indian_selector],
        longitude,
        np.asarray(indian_bounds.x_b)[indian_selector],
        np.asarray(indian_bounds.x_e)[indian_selector],
    ) * xr.DataArray(
        _smooth_ramp(limits["y_NI"] - latitude[indian_selector]),
        dims="latitude",
        coords={"latitude": latitude[indian_selector]},
    )

    pacific_bounds = bounds.sel(region="pacific_north")
    pacific_selector = (latitude >= y_P) & (latitude <= limits["y_NP"])
    pacific = xr.zeros_like(atlantic)
    pacific.loc[{"latitude": latitude[pacific_selector]}] = _zonal_taper(
        latitude[pacific_selector],
        longitude,
        np.asarray(pacific_bounds.x_b)[pacific_selector],
        np.asarray(pacific_bounds.x_e)[pacific_selector],
    ) * xr.DataArray(
        _smooth_ramp(limits["y_NP"] - latitude[pacific_selector]),
        dims="latitude",
        coords={"latitude": latitude[pacific_selector]},
    )
    taper = (atlantic + indian + pacific).clip(max=1.0)

    gamma = float(_f(_cap_latitude()))
    inverse_f = _f(winds.latitude) / (_f(winds.latitude) ** 2 + gamma**2)
    M_ek_x = winds.avg_inss * taper * inverse_f / RHO_0
    M_ek_y = -winds.avg_iews * taper * inverse_f / RHO_0
    M_ek_x.attrs["units"] = "m2 s-1"
    M_ek_y.attrs["units"] = "m2 s-1"

    southern_section = M_ek_y.interp(latitude=limits["y_S"])
    southern_bounds = geometry.boundaries_on(
        np.array([limits["y_S"] - 1e-6, limits["y_S"]])
    ).sel(region="atlantic_pacific_transition").isel(latitude=-1)
    west = float(southern_bounds.x_b)
    east = float(southern_bounds.x_e)
    center = 0.5 * (west + east)
    continuous = (
        np.asarray(southern_section.longitude) - center + 180.0
    ) % 360.0 + center - 180.0
    order = np.argsort(continuous)
    southern_section = southern_section.isel(longitude=order).assign_coords(
        longitude=continuous[order]
    )
    southern = southern_section.where(
        (southern_section.longitude >= west)
        & (southern_section.longitude <= east),
        0.0,
    ).integrate("longitude")
    southern = southern * (
        R_EARTH * np.cos(np.deg2rad(limits["y_S"])) * np.pi / 180.0
    )
    southern.attrs["units"] = "m3 s-1"
    scotia.attrs["units"] = "Sv"
    return GlobalForcing.from_time_series(
        M_ek_x=M_ek_x,
        M_ek_y=M_ek_y,
        northern_transport=scotia,
        southern_transport=southern,
        sample_interval_seconds=365.25 * 86_400.0 / 12.0,
        padding_samples=scotia.sizes["time"] - 1,
        n_fft=2048,
    )


def _stitched_atlantic(
    array: xr.DataArray,
    limits: dict[str, float],
    target_latitude: xr.DataArray,
) -> xr.DataArray:
    pieces = []
    for region, selector in (
        ("atlantic_pacific_transition", target_latitude < limits["y_P"]),
        (
            "atlantic_indian_transition",
            (target_latitude >= limits["y_P"])
            & (target_latitude < limits["y_I"]),
        ),
        ("atlantic_north", target_latitude >= limits["y_I"]),
    ):
        piece = array.sel(region=region).dropna("latitude", how="all")
        pieces.append(piece.interp(latitude=target_latitude.where(selector, drop=True)))
    result = xr.concat(
        pieces,
        dim="latitude",
        coords="minimal",
        compat="override",
    ).sortby("latitude")
    if "region" in result.coords:
        result = result.drop_vars("region")
    return result


def _correlation_and_nrmse(
    actual: xr.DataArray, expected: xr.DataArray
) -> tuple[float, float]:
    if "time" in actual.dims:
        actual = actual - actual.mean("time")
    if "time" in expected.dims:
        expected = expected - expected.mean("time")
    actual_values = np.asarray(actual).ravel()
    expected_values = np.asarray(expected).ravel()
    correlation = float(np.corrcoef(actual_values, expected_values)[0, 1])
    nrmse = float(
        np.sqrt(np.mean((actual_values - expected_values) ** 2))
        / np.std(expected_values)
    )
    return correlation, nrmse


@pytest.fixture(scope="module")
def atlantic_reference_case() -> tuple[
    object, xr.Dataset, dict[str, float]
]:
    geometry, limits = _geometry_and_limits()
    with dask.config.set(scheduler="threads", num_workers=2):
        forcing = _forcing(geometry, limits)
        output = GlobalAdjustmentModel(
            geometry=geometry,
            forcing=forcing,
            g_prime=G_PRIME,
        ).solve()
    reference = xr.open_dataset(
        ROOT / "data/model_output/global_atlantic_transport.nc"
    )
    return output, reference, limits


def test_era5_scotia_atlantic_reference(
    atlantic_reference_case: tuple[object, xr.Dataset, dict[str, float]],
) -> None:
    output, reference, limits = atlantic_reference_case

    h_e = output.h_e.sel(region="atlantic_north")
    h_e_correlation, h_e_nrmse = _correlation_and_nrmse(h_e, reference.h_e_A)
    assert h_e_correlation > 0.99
    assert h_e_nrmse < 0.13

    comparisons = {
        "h_w": _stitched_atlantic(output.h_w, limits, reference.latitude),
        "T_total": _stitched_atlantic(
            output.transport / 1e6, limits, reference.latitude
        ),
        "T_Ek": _stitched_atlantic(
            output.transport_ekman / 1e6, limits, reference.latitude
        ),
        "T_geostrophic": _stitched_atlantic(
            output.transport_geostrophic / 1e6, limits, reference.latitude
        ),
    }
    thresholds = {
        "h_w": (0.99, 0.13),
        "T_total": (0.995, 0.10),
        "T_Ek": (0.9999, 0.002),
        "T_geostrophic": (0.99, 0.16),
    }
    for name, actual in comparisons.items():
        expected = reference[name].interp(latitude=actual.latitude)
        correlation, nrmse = _correlation_and_nrmse(actual, expected)
        minimum_correlation, maximum_nrmse = thresholds[name]
        assert correlation > minimum_correlation, (name, correlation, nrmse)
        assert nrmse < maximum_nrmse, (name, correlation, nrmse)

    xr.testing.assert_allclose(
        output.transport,
        output.transport_ekman + output.transport_geostrophic,
    )
    assert float(abs(output.spectral.southern_budget_residual).max()) < 1e-4


def test_era5_scotia_h_b_continuous_transport_regression(
    atlantic_reference_case: tuple[object, xr.Dataset, dict[str, float]],
) -> None:
    output, reference, limits = atlantic_reference_case
    actual = _stitched_atlantic(output.h_b, limits, reference.latitude)
    correlation, nrmse = _correlation_and_nrmse(actual, reference.h_b)
    # The legacy sector-by-sector taper creates gateway curl sheets that are
    # absent from the definitive continuous vector-transport preparation.
    # These reviewed tolerances retain the legacy large-scale h_b benchmark
    # while explicitly allowing that known upstream difference.
    assert correlation > 0.93
    assert nrmse < 0.38
