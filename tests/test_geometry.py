from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import MultiBasinGeometry


TRACE_VARIABLES = {
    "atlantic_west": "x_wA",
    "atlantic_east": "x_eA",
    "indian_west": "x_wI",
    "indian_east": "x_eI",
    "pacific_west": "x_wP",
    "pacific_east": "x_eP",
}

REGIONS = {
    "atlantic_north": {
        "west": "atlantic_west",
        "east": "atlantic_east",
        "south": -34.99,
        "north": 55.0,
    },
    "indian_north": {
        "west": "indian_west",
        "east": "indian_east",
        "south": -34.99,
        "north": 24.5,
    },
    "pacific_north": {
        "west": "pacific_west",
        "east": "pacific_east",
        "south": -43.99,
        "north": 59.0,
    },
    "atlantic_indian_transition": {
        "west": "atlantic_west",
        "east": "indian_east",
        "south": -43.99,
        "north": -34.99,
    },
    "atlantic_pacific_transition": {
        "west": "atlantic_west",
        "east": "pacific_east",
        "south": -55.0,
        "north": -43.99,
    },
}


def tracked_path(depth: int = 1000) -> Path:
    return (
        Path(__file__).parents[1]
        / "data"
        / "tracked"
        / "isobath"
        / f"global_isobath_GEBCO_{depth}m.nc"
    )


def in_memory_isobaths() -> xr.Dataset:
    latitude = np.arange(-60.0, 66.0)

    def trace(value: float, south: float, north: float) -> tuple[str, np.ndarray]:
        data = np.full(latitude.size, np.nan)
        data[(latitude >= south) & (latitude <= north)] = value
        return "latitude", data

    return xr.Dataset(
        {
            "x_wA": trace(-150.0, -56.0, 65.0),
            "x_eA": trace(-80.0, -35.0, 65.0),
            "x_wI": trace(-30.0, -35.0, 30.0),
            "x_eI": trace(30.0, -44.0, 30.0),
            "x_wP": trace(80.0, -44.0, 60.0),
            "x_eP": trace(150.0, -56.0, 60.0),
        },
        coords={"latitude": latitude},
        attrs={"isobath_depth_m": 1000.0},
    )


def test_compact_dataset_builds_labelled_geometry() -> None:
    geometry = MultiBasinGeometry.from_isobath_dataset(
        in_memory_isobaths(),
        trace_variables=TRACE_VARIABLES,
        region_definitions=REGIONS,
    )

    assert geometry.H == 1000.0
    assert geometry.dataset.sizes["trace"] == 6
    assert geometry.dataset.sizes["region"] == 5
    assert float(
        geometry.x_b.sel(region="atlantic_north", latitude=0.0)
    ) == pytest.approx(-150.0)
    assert float(
        geometry.x_e.sel(region="atlantic_north", latitude=0.0)
    ) == pytest.approx(-80.0)
    assert bool((geometry.x_e > geometry.x_b).where(geometry.region_mask).all())


@pytest.mark.parametrize(
    ("H", "expected"),
    [
        (1000.0, (-44.448, 8.581)),
        (500.0, (-44.5073, 8.7459)),
    ],
)
def test_tracked_notebook_products_load(
    H: float, expected: tuple[float, float]
) -> None:
    with xr.open_dataset(tracked_path(int(H))) as dataset:
        geometry = MultiBasinGeometry.from_isobath_dataset(
            dataset,
            trace_variables=TRACE_VARIABLES,
            region_definitions=REGIONS,
        )

    equator = geometry.dataset.latitude.sel(latitude=0.0, method="nearest")
    assert float(
        geometry.x_b.sel(region="atlantic_north", latitude=equator)
    ) == pytest.approx(expected[0], abs=0.02)
    assert float(
        geometry.x_e.sel(region="atlantic_north", latitude=equator)
    ) == pytest.approx(expected[1], abs=0.02)
    assert geometry.H == H


def test_every_region_limit_must_be_explicit() -> None:
    regions = {key: dict(value) for key, value in REGIONS.items()}
    regions["indian_north"].pop("north")
    with pytest.raises(ValueError, match="explicit north"):
        MultiBasinGeometry.from_isobath_dataset(
            in_memory_isobaths(),
            trace_variables=TRACE_VARIABLES,
            region_definitions=regions,
        )


def test_fixed_region_stitching_is_validated() -> None:
    regions = {key: dict(value) for key, value in REGIONS.items()}
    regions["pacific_north"]["south"] = -43.0
    with pytest.raises(ValueError, match="Pacific gateway"):
        MultiBasinGeometry.from_isobath_dataset(
            in_memory_isobaths(),
            trace_variables=TRACE_VARIABLES,
            region_definitions=regions,
        )


def test_isobath_depth_cannot_be_relabelled() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        MultiBasinGeometry.from_isobath_dataset(
            in_memory_isobaths(),
            trace_variables=TRACE_VARIABLES,
            region_definitions=REGIONS,
            H=500.0,
        )


def test_loader_does_not_extrapolate_missing_gateway() -> None:
    regions = {key: dict(value) for key, value in REGIONS.items()}
    regions["atlantic_pacific_transition"]["south"] = -57.0
    with pytest.raises(ValueError, match="western traces do not cover"):
        MultiBasinGeometry.from_isobath_dataset(
            in_memory_isobaths(),
            trace_variables=TRACE_VARIABLES,
            region_definitions=regions,
        )


def test_existing_internal_trace_gap_is_not_silently_filled() -> None:
    dataset = in_memory_isobaths()
    gap = (dataset.latitude >= -1.0) & (dataset.latitude <= 1.0)
    dataset["x_wA"] = dataset.x_wA.where(~gap)
    with pytest.raises(ValueError, match="western traces do not cover"):
        MultiBasinGeometry.from_isobath_dataset(
            dataset,
            trace_variables=TRACE_VARIABLES,
            region_definitions=REGIONS,
        )
