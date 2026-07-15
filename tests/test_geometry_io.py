from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import (
    Basin,
    BoundaryTrace,
    MultiBasinTopology,
    topology_from_dataset,
    topology_to_dataset,
)


def test_topology_dataset_contains_geometry_contract(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    topology = MultiBasinTopology(non_itf_basins)
    dataset = topology_to_dataset(topology)

    assert dataset.sizes == {"trace": 6, "latitude": 117, "basin": 5}
    assert dataset.attrs["isobath_depth_m"] == 1000.0
    assert dataset.attrs["southern_boundary"] == -56.0
    assert dataset.attrs["atlantic_north"] == 55.0
    assert dataset.basin_west_trace.sel(
        basin="atlantic_indian_transition"
    ).item() == "atlantic_west"
    assert dataset.basin_east_trace.sel(
        basin="atlantic_pacific_transition"
    ).item() == "pacific_east"
    pacific = dataset.sel(trace="pacific_east")
    assert float(pacific.longitude.max()) == 290.0
    assert float(pacific.longitude_wrapped.max()) == -70.0


def test_topology_dataset_round_trip_preserves_scientific_geometry(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    original = MultiBasinTopology(non_itf_basins)
    restored = topology_from_dataset(topology_to_dataset(original))

    assert restored.connections == original.connections
    assert restored.eastern_boundary_groups == original.eastern_boundary_groups
    assert restored.basin("atlantic_north").northern_boundary == 55.0
    assert restored.basin("indian_north").northern_boundary == 58.0
    assert restored.basin("pacific_north").northern_boundary == 60.0
    assert (
        restored.basin("atlantic_north").western_boundary
        is restored.basin("atlantic_pacific_transition").western_boundary
    )
    np.testing.assert_allclose(
        restored.basin("pacific_north").eastern_boundary.longitude,
        original.basin("pacific_north").eastern_boundary.longitude,
        equal_nan=True,
    )
    assert restored.basin("atlantic_north").western_boundary.provenance == {
        "fixture": "synthetic"
    }


def test_topology_dataset_preserves_raw_and_repaired_samples(
    boundary_traces: dict[str, BoundaryTrace],
) -> None:
    traces = dict(boundary_traces)
    original = traces["atlantic_west"]
    raw = np.asarray(original.raw_longitude).copy()
    repaired = np.asarray(original.repaired).copy()
    index = int(np.flatnonzero(original.latitude == -50.0)[0])
    raw[index] = np.nan
    repaired[index] = True
    traces["atlantic_west"] = BoundaryTrace(
        key=original.key,
        side=original.side,
        latitude=original.latitude,
        longitude=original.longitude,
        depth=original.depth,
        raw_longitude=raw,
        valid=original.valid,
        repaired=repaired,
        provenance=original.provenance,
    )

    restored = topology_from_dataset(
        topology_to_dataset(MultiBasinTopology.from_traces(traces))
    )
    restored_trace = restored.basin("atlantic_north").western_boundary

    assert np.isnan(restored_trace.raw_longitude[index])
    assert restored_trace.repaired[index]
    assert restored_trace.longitude[index] == -70.0


def test_topology_dataset_round_trip_through_netcdf(
    non_itf_basins: tuple[Basin, ...], tmp_path
) -> None:
    path = tmp_path / "geometry.nc"
    topology_to_dataset(MultiBasinTopology(non_itf_basins)).to_netcdf(
        path, engine="h5netcdf"
    )

    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        restored = topology_from_dataset(dataset)

    assert restored.basin("atlantic_north").x_b(0.0) == -70.0
    assert restored.basin("pacific_north").x_e(0.0) == 290.0


def test_topology_from_dataset_rejects_unknown_format(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    dataset = topology_to_dataset(MultiBasinTopology(non_itf_basins))
    dataset.attrs["geometry_format_version"] = "future"

    with pytest.raises(ValueError, match="unsupported geometry format"):
        topology_from_dataset(dataset)


def test_topology_to_dataset_rejects_mismatched_trace_grids(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    basins = list(non_itf_basins)
    original = basins[0].eastern_boundary
    extended = BoundaryTrace(
        key=original.key,
        side=original.side,
        latitude=np.append(original.latitude, 61.0),
        longitude=np.append(original.longitude, np.nan),
        depth=original.depth,
        raw_longitude=np.append(original.raw_longitude, np.nan),
        valid=np.append(original.valid, False),
        repaired=np.append(original.repaired, False),
        provenance=original.provenance,
    )
    basins[0] = replace(basins[0], eastern_boundary=extended)

    with pytest.raises(ValueError, match="one exact latitude grid"):
        topology_to_dataset(MultiBasinTopology(basins))


def test_topology_from_dataset_rejects_corrupt_basin_mapping(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    dataset = topology_to_dataset(MultiBasinTopology(non_itf_basins))
    dataset["basin_east_trace"].loc[
        {"basin": "atlantic_north"}
    ] = "pacific_east"

    with pytest.raises(ValueError, match="eastern trace mapping"):
        topology_from_dataset(dataset)
