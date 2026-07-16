from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import MultiBasinGeometry


REGIONS = {
    "atlantic_north": {
        "west": "alpha_west",
        "east": "alpha_east",
        "south": -35.0,
        "north": 55.0,
    },
    "indian_north": {
        "west": "beta_west",
        "east": "beta_east",
        "south": -35.0,
        "north": 30.0,
    },
    "pacific_north": {
        "west": "gamma_west",
        "east": "gamma_east",
        "south": -44.0,
        "north": 58.0,
    },
    "atlantic_indian_transition": {
        "west": "alpha_west",
        "east": "beta_east",
        "south": -44.0,
        "north": -35.0,
    },
    "atlantic_pacific_transition": {
        "west": "alpha_west",
        "east": "gamma_east",
        "south": -56.0,
        "north": -44.0,
    },
}

BASINS = {
    "alpha": {"longitude_bounds": (-170.0, -50.0), "seed": (-110.0, 0.0)},
    "beta": {"longitude_bounds": (-50.0, 60.0), "seed": (0.0, 0.0)},
    "gamma": {"longitude_bounds": (60.0, 170.0), "seed": (110.0, 0.0)},
}


def synthetic_depth() -> xr.DataArray:
    latitude = np.arange(-60.0, 66.0)
    longitude = np.arange(-180.0, 180.0)
    depth = np.zeros((latitude.size, longitude.size), dtype=float)

    def ocean(west: float, east: float, south: float, north: float) -> None:
        y = (latitude >= south) & (latitude <= north)
        x = (longitude >= west) & (longitude <= east)
        depth[np.ix_(y, x)] = 2000.0

    ocean(-150.0, -80.0, -56.0, 60.0)
    ocean(-30.0, 30.0, -44.0, 30.0)
    ocean(80.0, 150.0, -56.0, 58.0)
    return xr.DataArray(
        depth,
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
        name="depth",
        attrs={"units": "m", "positive": "down"},
    )


def test_from_bathymetry_has_no_named_ocean_coordinates() -> None:
    geometry = MultiBasinGeometry.from_bathymetry(
        synthetic_depth(),
        H=1000.0,
        basin_definitions=BASINS,
        region_definitions=REGIONS,
    )

    assert geometry.H == 1000.0
    assert geometry.dataset.sizes["trace"] == 6
    assert geometry.dataset.sizes["region"] == 5
    assert float(geometry.x_b.sel(region="atlantic_north", latitude=0.0)) == pytest.approx(-150.5)
    assert float(geometry.x_e.sel(region="atlantic_north", latitude=0.0)) == pytest.approx(-79.5)
    assert float(geometry.dataset.region_north.sel(region="indian_north")) == 30.0
    assert float(geometry.dataset.region_north.sel(region="pacific_north")) == 58.0
    assert bool((geometry.x_e > geometry.x_b).where(geometry.region_mask).all())


def test_H_changes_the_interpolated_contour() -> None:
    shallow = MultiBasinGeometry.from_bathymetry(
        synthetic_depth(),
        H=500.0,
        basin_definitions=BASINS,
        region_definitions=REGIONS,
    )
    deep = MultiBasinGeometry.from_bathymetry(
        synthetic_depth(),
        H=1000.0,
        basin_definitions=BASINS,
        region_definitions=REGIONS,
    )

    assert float(shallow.x_b.sel(region="atlantic_north", latitude=0.0)) == pytest.approx(-150.75)
    assert float(deep.x_b.sel(region="atlantic_north", latitude=0.0)) == pytest.approx(-150.5)


def test_from_bathymetry_rejects_implicit_or_incomplete_geography() -> None:
    incomplete = dict(REGIONS)
    incomplete.pop("pacific_north")
    with pytest.raises(ValueError, match="fixed five"):
        MultiBasinGeometry.from_bathymetry(
            synthetic_depth(),
            H=1000.0,
            basin_definitions=BASINS,
            region_definitions=incomplete,
        )


TRACE_VARIABLES = {
    "atlantic_west": "x_wA",
    "atlantic_east": "x_eA",
    "indian_west": "x_wI",
    "indian_east": "x_eI",
    "pacific_west": "x_wP",
    "pacific_east": "x_eP",
}

REFERENCE_REGIONS = {
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


@pytest.mark.parametrize(
    ("H", "filename", "expected"),
    [
        (1000.0, "global_isobath_GEBCO_1000m.nc", (-44.448, 8.581)),
        (500.0, "global_isobath_GEBCO_500m.nc", (-44.5073, 8.7459)),
    ],
)
def test_tracked_notebook_isobaths_load_with_explicit_conventions(
    H: float, filename: str, expected: tuple[float, float]
) -> None:
    path = Path(__file__).parents[1] / "data" / "tracked" / "isobath" / filename
    with xr.open_dataset(path) as dataset:
        geometry = MultiBasinGeometry.from_isobath_dataset(
            dataset,
            trace_variables=TRACE_VARIABLES,
            region_definitions=REFERENCE_REGIONS,
        )

    equator = geometry.dataset.latitude.sel(latitude=0.0, method="nearest")
    assert float(geometry.x_b.sel(region="atlantic_north", latitude=equator)) == pytest.approx(expected[0], abs=0.02)
    assert float(geometry.x_e.sel(region="atlantic_north", latitude=equator)) == pytest.approx(expected[1], abs=0.02)
    assert float(geometry.dataset.region_north.sel(region="pacific_north")) > 55.0
    assert geometry.H == H


def test_isobath_loader_does_not_extrapolate_missing_gateway() -> None:
    path = (
        Path(__file__).parents[1]
        / "data/tracked/isobath/global_isobath_GEBCO_1000m.nc"
    )
    regions = {key: dict(value) for key, value in REFERENCE_REGIONS.items()}
    regions["atlantic_pacific_transition"]["south"] = -56.0
    with xr.open_dataset(path) as dataset, pytest.raises(
        ValueError, match="western traces do not cover"
    ):
        MultiBasinGeometry.from_isobath_dataset(
            dataset,
            trace_variables=TRACE_VARIABLES,
            region_definitions=regions,
        )


def test_every_region_limit_must_be_explicit() -> None:
    regions = {key: dict(value) for key, value in REGIONS.items()}
    regions["indian_north"].pop("north")
    with pytest.raises(ValueError, match="explicit north"):
        MultiBasinGeometry.from_bathymetry(
            synthetic_depth(),
            H=1000.0,
            basin_definitions=BASINS,
            region_definitions=regions,
        )


def test_region_stitching_is_validated() -> None:
    regions = {key: dict(value) for key, value in REGIONS.items()}
    regions["pacific_north"]["south"] = -43.0
    with pytest.raises(ValueError, match="Pacific gateway"):
        MultiBasinGeometry.from_bathymetry(
            synthetic_depth(),
            H=1000.0,
            basin_definitions=BASINS,
            region_definitions=regions,
        )


def test_isobath_depth_cannot_be_relabelled() -> None:
    path = (
        Path(__file__).parents[1]
        / "data/tracked/isobath/global_isobath_GEBCO_1000m.nc"
    )
    with xr.open_dataset(path) as dataset, pytest.raises(ValueError, match="conflicts"):
        MultiBasinGeometry.from_isobath_dataset(
            dataset,
            trace_variables=TRACE_VARIABLES,
            region_definitions=REFERENCE_REGIONS,
            H=500.0,
        )


def test_unknown_extraction_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown extraction options"):
        MultiBasinGeometry.from_bathymetry(
            synthetic_depth(),
            H=1000.0,
            basin_definitions=BASINS,
            region_definitions=REGIONS,
            extraction_options={"caribbean_fix": True},
        )


def test_explicit_closure_is_the_only_connectivity_change() -> None:
    unmodified = MultiBasinGeometry.from_bathymetry(
        synthetic_depth(),
        H=1000.0,
        basin_definitions=BASINS,
        region_definitions=REGIONS,
    )
    closed = MultiBasinGeometry.from_bathymetry(
        synthetic_depth(),
        H=1000.0,
        basin_definitions=BASINS,
        region_definitions=REGIONS,
        closures=(
            {
                "name": "synthetic_partition",
                "basins": ("alpha",),
                "points": ((-100.0, -60.0), (-100.0, 65.0)),
                "width_degrees": 0.6,
            },
        ),
    )

    original_east = float(
        unmodified.x_e.sel(region="atlantic_north", latitude=0.0)
    )
    closed_east = float(closed.x_e.sel(region="atlantic_north", latitude=0.0))
    assert original_east == pytest.approx(-79.5)
    assert closed_east < -99.0


def test_dateline_window_remains_continuous() -> None:
    depth = synthetic_depth().copy()
    latitude = np.asarray(depth.latitude)
    longitude = np.asarray(depth.longitude)
    depth.values[:] = 0.0
    alpha_rows = (latitude >= -56.0) & (latitude <= 60.0)
    beta_rows = (latitude >= -44.0) & (latitude <= 30.0)
    gamma_rows = (latitude >= -56.0) & (latitude <= 58.0)
    depth.values[np.ix_(alpha_rows, (longitude >= -60.0) & (longitude <= 20.0))] = 2000.0
    depth.values[np.ix_(beta_rows, (longitude >= 30.0) & (longitude <= 90.0))] = 2000.0
    dateline_ocean = (longitude >= 110.0) | (longitude <= -80.0)
    depth.values[np.ix_(gamma_rows, dateline_ocean)] = 2000.0
    basins = {
        "alpha": {"longitude_bounds": (-70.0, 25.0), "seed": (-20.0, 0.0)},
        "beta": {"longitude_bounds": (25.0, 100.0), "seed": (60.0, 0.0)},
        "gamma": {
            "longitude_bounds": (100.0, 290.0),
            "seed": (200.0, 0.0),
        },
    }

    geometry = MultiBasinGeometry.from_bathymetry(
        depth,
        H=1000.0,
        basin_definitions=basins,
        region_definitions=REGIONS,
    )

    west = float(geometry.dataset.longitude.sel(trace="gamma_west", latitude=0.0))
    east = float(geometry.dataset.longitude.sel(trace="gamma_east", latitude=0.0))
    assert west == pytest.approx(109.5)
    assert east == pytest.approx(280.5)
    assert east > west


def test_existing_internal_trace_gap_is_not_silently_filled() -> None:
    path = (
        Path(__file__).parents[1]
        / "data/tracked/isobath/global_isobath_GEBCO_1000m.nc"
    )
    with xr.open_dataset(path) as source:
        dataset = source.load()
    gap = (dataset.latitude >= -1.0) & (dataset.latitude <= 1.0)
    dataset["x_wA"] = dataset.x_wA.where(~gap)
    with pytest.raises(ValueError, match="western traces do not cover"):
        MultiBasinGeometry.from_isobath_dataset(
            dataset,
            trace_variables=TRACE_VARIABLES,
            region_definitions=REFERENCE_REGIONS,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    "MOC_GEBCO_FILE" not in os.environ,
    reason="set MOC_GEBCO_FILE to run the full notebook extraction regression",
)
@pytest.mark.parametrize(
    ("H", "reference_name"),
    [
        (500.0, "global_isobath_GEBCO_500m.nc"),
        (1000.0, "global_isobath_GEBCO_1000m.nc"),
    ],
)
def test_from_bathymetry_against_notebook_product(
    H: float, reference_name: str
) -> None:
    source_path = Path(os.environ["MOC_GEBCO_FILE"])
    with xr.open_dataset(source_path, chunks={"lat": 240, "lon": -1}) as source:
        extracted = MultiBasinGeometry.from_bathymetry(
            source.elevation,
            H=H,
            basin_definitions={
                "atlantic": {
                    "longitude_bounds": (-100.0, 40.0),
                    "seed": (-30.0, 0.0),
                },
                "indian": {
                    "longitude_bounds": (20.0, 150.0),
                    "seed": (75.0, 0.0),
                },
                "pacific": {
                    "longitude_bounds": (105.0, 290.0),
                    "seed": (200.0, 0.0),
                },
            },
            region_definitions=REFERENCE_REGIONS,
            closures=(
                {
                    "name": "greenland_iceland_uk_europe",
                    "basins": ("atlantic",),
                    "points": ((5.0, 60.0), (-6.5, 62.0), (-18.5, 64.8), (-38.0, 66.0)),
                    "width_degrees": 0.45,
                },
                {
                    "name": "se_asia_to_australia",
                    "basins": ("indian", "pacific"),
                    "points": ((103.0, 1.0), (110.0, -6.5), (121.0, -8.8), (130.0, -10.5), (142.0, -12.0)),
                    "width_degrees": 0.45,
                },
                {
                    "name": "bering_strait",
                    "basins": ("pacific",),
                    "points": ((165.0, 65.5), (179.0, 65.5), (-166.0, 65.5)),
                    "width_degrees": 0.55,
                },
                {
                    "name": "caribbean_arc",
                    "basins": ("atlantic",),
                    "points": ((-81.5, 24.5), (-77.5, 21.5), (-73.0, 19.0), (-66.5, 18.0), (-61.5, 14.5)),
                    "width_degrees": 0.35,
                },
            ),
            ignored_features=(
                {
                    "name": "madagascar",
                    "basins": ("indian",),
                    "bounds": (42.0, 52.5, -27.0, -10.0),
                },
                {
                    "name": "new_zealand",
                    "basins": ("pacific",),
                    "bounds": (165.0, 180.0, -49.5, -33.0),
                },
            ),
            extraction_options={
                "positive": "up",
                "coarsen_factor": 4,
                "maximum_gap_degrees": 6.0,
                "smoothing_sigma_degrees": 0.45,
            },
        )

    reference_path = (
        Path(__file__).parents[1] / "data" / "tracked" / "isobath" / reference_name
    )
    with xr.open_dataset(reference_path) as reference:
        for trace, variable in TRACE_VARIABLES.items():
            expected = reference[variable].interp(latitude=extracted.dataset.latitude)
            actual = extracted.dataset.longitude.sel(trace=trace)
            common = np.isfinite(actual) & np.isfinite(expected)
            error = np.abs(actual.where(common, drop=True) - expected.where(common, drop=True))
            assert float(error.median()) < 0.25
            assert float(error.quantile(0.95)) < 1.0
