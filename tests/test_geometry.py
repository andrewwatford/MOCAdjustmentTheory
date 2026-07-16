from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import MultiBasinGeometry


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

    dataset = xr.Dataset(
        {
            "atlantic_west": trace(-150.0, -56.0, 65.0),
            "atlantic_east": trace(-80.0, -35.0, 65.0),
            "indian_west": trace(-30.0, -35.0, 30.0),
            "indian_east": trace(30.0, -44.0, 30.0),
            "pacific_west": trace(80.0, -44.0, 60.0),
            "pacific_east": trace(150.0, -56.0, 60.0),
        },
        coords={"latitude": latitude},
        attrs={"isobath_depth_m": 1000.0},
    )
    regions = list(REGIONS)
    dataset = dataset.assign_coords(region=regions)
    dataset["region_west_trace"] = (
        "region",
        [str(REGIONS[key]["west"]) for key in regions],
    )
    dataset["region_east_trace"] = (
        "region",
        [str(REGIONS[key]["east"]) for key in regions],
    )
    dataset["region_south"] = (
        "region",
        [float(REGIONS[key]["south"]) for key in regions],
    )
    dataset["region_north"] = (
        "region",
        [float(REGIONS[key]["north"]) for key in regions],
    )
    return dataset


def test_compact_dataset_builds_labelled_geometry() -> None:
    geometry = MultiBasinGeometry.from_isobath_dataset(in_memory_isobaths())

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
        geometry = MultiBasinGeometry.from_isobath_dataset(dataset)

    equator = geometry.dataset.latitude.sel(latitude=0.0, method="nearest")
    assert float(
        geometry.x_b.sel(region="atlantic_north", latitude=equator)
    ) == pytest.approx(expected[0], abs=0.02)
    assert float(
        geometry.x_e.sel(region="atlantic_north", latitude=equator)
    ) == pytest.approx(expected[1], abs=0.02)
    assert geometry.H == H


def test_region_metadata_must_be_complete() -> None:
    dataset = in_memory_isobaths().drop_vars("region_north")
    with pytest.raises(ValueError, match="incomplete region metadata"):
        MultiBasinGeometry.from_isobath_dataset(dataset)


def test_fixed_region_stitching_is_validated() -> None:
    dataset = in_memory_isobaths()
    dataset["region_south"].loc[{"region": "pacific_north"}] = -43.0
    with pytest.raises(ValueError, match="Pacific gateway"):
        MultiBasinGeometry.from_isobath_dataset(dataset)


def test_six_boundary_roles_must_be_distinct() -> None:
    dataset = in_memory_isobaths()
    for region in ("indian_north", "atlantic_indian_transition"):
        dataset["region_east_trace"].loc[{"region": region}] = "atlantic_east"

    with pytest.raises(ValueError, match="six distinct"):
        MultiBasinGeometry.from_isobath_dataset(dataset)


def test_canonical_trace_names_are_required() -> None:
    dataset = in_memory_isobaths().rename({"indian_east": "x_eI"})
    with pytest.raises(ValueError, match="missing canonical traces"):
        MultiBasinGeometry.from_isobath_dataset(dataset)


def test_isobath_depth_cannot_be_relabelled() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        MultiBasinGeometry.from_isobath_dataset(
            in_memory_isobaths(),
            H=500.0,
        )


def test_loader_does_not_extrapolate_missing_gateway() -> None:
    dataset = in_memory_isobaths()
    dataset["region_south"].loc[
        {"region": "atlantic_pacific_transition"}
    ] = -57.0
    with pytest.raises(ValueError, match="western traces do not cover"):
        MultiBasinGeometry.from_isobath_dataset(dataset)


def test_existing_internal_trace_gap_is_not_silently_filled() -> None:
    dataset = in_memory_isobaths()
    gap = (dataset.latitude >= -1.0) & (dataset.latitude <= 1.0)
    dataset["atlantic_west"] = dataset.atlantic_west.where(~gap)
    with pytest.raises(ValueError, match="western traces do not cover"):
        MultiBasinGeometry.from_isobath_dataset(dataset)
