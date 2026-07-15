from __future__ import annotations

import numpy as np
import xarray as xr

from moc_adjustment_theory import MultiBasinTopology, extract_boundary_traces


def synthetic_global_elevation() -> xr.DataArray:
    """Simple global ocean separated by three continental barriers."""

    latitude = np.arange(-60.0, 66.0)
    longitude = np.arange(-180.0, 180.0)
    elevation = np.full((latitude.size, longitude.size), -2000.0)

    def add_land(
        west: float, east: float, south: float, north: float
    ) -> None:
        lat_mask = (latitude >= south) & (latitude <= north)
        lon_mask = (longitude >= west) & (longitude <= east)
        elevation[np.ix_(lat_mask, lon_mask)] = 0.0

    add_land(-80.0, -60.0, -56.0, 65.0)  # Americas
    add_land(-20.0, 30.0, -35.0, 65.0)  # Africa and Europe
    add_land(100.0, 155.0, -44.0, 65.0)  # Australia and Asia
    add_land(30.0, 100.0, 26.0, 65.0)  # closed northern Indian Ocean
    add_land(155.0, 179.0, 60.0, 65.0)  # closed northern Pacific
    add_land(-180.0, -100.0, 60.0, 65.0)
    add_land(45.0, 50.0, -25.0, -10.0)  # Madagascar-like island
    add_land(170.0, 175.0, -49.0, -35.0)  # New-Zealand-like island

    return xr.DataArray(
        elevation,
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
        name="elevation",
        attrs={"units": "m", "source": "analytic test ocean"},
    )


def test_extract_boundary_traces_builds_exact_shared_geometry() -> None:
    elevation = synthetic_global_elevation().chunk(
        {"latitude": 16, "longitude": 90}
    )
    traces = extract_boundary_traces(elevation, search_factor=1)
    topology = MultiBasinTopology.from_traces(traces)

    assert set(traces) == {
        "atlantic_west",
        "atlantic_east",
        "indian_west",
        "indian_east",
        "pacific_west",
        "pacific_east",
    }
    assert all(
        np.array_equal(trace.latitude, traces["atlantic_west"].latitude)
        for trace in traces.values()
    )
    assert all(
        traces["atlantic_west"].has_latitude(latitude)
        for latitude in (-56.0, -44.0, -35.0, 55.0)
    )
    assert topology.basin("atlantic_north").northern_boundary == 55.0
    assert topology.basin("indian_north").northern_boundary == 25.0
    assert topology.basin("pacific_north").northern_boundary == 59.0
    assert traces["pacific_east"].longitude_at(0.0) == 279.5
    assert traces["atlantic_west"].longitude_at(0.0) == -59.5
    assert traces["indian_west"].longitude_at(-15.0) == 30.5
    assert traces["pacific_west"].longitude_at(-40.0) == 155.5
    assert np.nanmax(np.abs(np.diff(traces["pacific_east"].longitude))) < 1.0
    assert not np.any(
        traces["pacific_east"].valid[
            traces["pacific_east"].latitude < -56.0
        ]
    )
    assert "indonesian_throughflow" in traces["indian_east"].provenance[
        "closures"
    ]


def test_extract_boundary_traces_is_deterministic() -> None:
    elevation = synthetic_global_elevation()
    first = extract_boundary_traces(elevation, search_factor=1)
    second = extract_boundary_traces(elevation, search_factor=1)

    for key in first:
        np.testing.assert_array_equal(first[key].longitude, second[key].longitude)
        np.testing.assert_array_equal(first[key].repaired, second[key].repaired)


def test_extract_boundary_traces_inserts_exact_requested_latitudes() -> None:
    traces = extract_boundary_traces(
        synthetic_global_elevation(),
        search_factor=1,
        southern_boundary=-55.9,
        pacific_gateway=-43.9,
        indian_gateway=-34.9,
        atlantic_north=55.1,
    )

    for latitude in (-55.9, -43.9, -34.9, 55.1):
        assert np.any(traces["atlantic_west"].latitude == latitude)
    assert traces["atlantic_west"].has_latitude(-55.9)
    assert traces["atlantic_east"].has_latitude(-34.9)


def test_extract_boundary_traces_records_one_row_repair() -> None:
    elevation = synthetic_global_elevation()
    elevation.loc[
        {"latitude": 10.0, "longitude": slice(-80.0, -60.0)}
    ] = -2000.0
    traces = extract_boundary_traces(elevation, search_factor=1)
    index = int(np.flatnonzero(traces["atlantic_west"].latitude == 10.0)[0])

    assert traces["atlantic_west"].repaired[index]
    assert np.isnan(traces["atlantic_west"].raw_longitude[index])
    assert traces["atlantic_west"].longitude[index] == -59.5


def test_extract_boundary_traces_rejects_long_gap() -> None:
    elevation = synthetic_global_elevation()
    elevation.loc[
        {"latitude": slice(10.0, 12.0), "longitude": slice(-80.0, -60.0)}
    ] = -2000.0

    with np.testing.assert_raises_regex(ValueError, "unrepaired boundary gap"):
        extract_boundary_traces(elevation, search_factor=1)
