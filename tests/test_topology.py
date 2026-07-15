from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from moc_adjustment_theory import Basin, BoundaryTrace, MultiBasinTopology


def test_non_itf_topology_has_exact_graph(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    topology = MultiBasinTopology(non_itf_basins)

    assert topology.basin_keys == (
        "atlantic_north",
        "indian_north",
        "pacific_north",
        "atlantic_indian_transition",
        "atlantic_pacific_transition",
    )
    assert topology.connections == {
        "T_S": (None, "atlantic_pacific_transition"),
        "T_P": ("atlantic_pacific_transition", "pacific_north"),
        "T_T": (
            "atlantic_pacific_transition",
            "atlantic_indian_transition",
        ),
        "T_I": ("atlantic_indian_transition", "indian_north"),
        "T_A": ("atlantic_indian_transition", "atlantic_north"),
        "T_N": ("atlantic_north", None),
    }
    assert topology.children("atlantic_pacific_transition") == (
        "pacific_north",
        "atlantic_indian_transition",
    )
    assert topology.children("atlantic_indian_transition") == (
        "indian_north",
        "atlantic_north",
    )
    assert topology.parents("atlantic_north") == (
        "atlantic_indian_transition",
    )
    assert topology.parents("indian_north") == (
        "atlantic_indian_transition",
    )
    assert topology.children("indian_north") == ()
    assert topology.children("pacific_north") == ()
    assert (
        topology.basin("indian_north").northern_boundary
        > topology.basin("atlantic_north").northern_boundary
    )
    assert (
        topology.basin("pacific_north").northern_boundary
        > topology.basin("atlantic_north").northern_boundary
    )


def test_topology_factory_assembles_shared_traces_and_independent_norths(
    boundary_traces: dict[str, BoundaryTrace],
) -> None:
    topology = MultiBasinTopology.from_traces(boundary_traces)

    assert topology.basin("indian_north").northern_boundary == 58.0
    assert topology.basin("pacific_north").northern_boundary == 60.0
    assert topology.basin("atlantic_north").northern_boundary == 55.0
    assert (
        topology.basin("atlantic_north").western_boundary
        is topology.basin("atlantic_pacific_transition").western_boundary
    )
    assert (
        topology.basin("indian_north").eastern_boundary
        is topology.basin("atlantic_indian_transition").eastern_boundary
    )


def test_topology_factory_requires_exact_trace_set(
    boundary_traces: dict[str, BoundaryTrace],
) -> None:
    incomplete = dict(boundary_traces)
    incomplete.pop("pacific_west")

    with pytest.raises(ValueError, match="missing=.*pacific_west"):
        MultiBasinTopology.from_traces(incomplete)


def test_topology_preserves_boundary_identity_and_atlantic_path(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    topology = MultiBasinTopology(non_itf_basins)

    assert topology.atlantic_path == (
        "atlantic_pacific_transition",
        "atlantic_indian_transition",
        "atlantic_north",
    )
    assert topology.eastern_boundary_groups == {
        "atlantic_east": ("atlantic_north",),
        "indian_east": ("indian_north", "atlantic_indian_transition"),
        "pacific_east": ("pacific_north", "atlantic_pacific_transition"),
    }
    assert (
        topology.basin("atlantic_north").western_boundary
        is topology.basin("atlantic_indian_transition").western_boundary
    )
    assert (
        topology.basin("indian_north").eastern_boundary
        is topology.basin("atlantic_indian_transition").eastern_boundary
    )
    assert topology.depth == 1000.0


def test_basin_exposes_geometry_and_named_connections(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    basin = non_itf_basins[-1]

    assert basin.inflows == ("T_S",)
    assert basin.outflows == ("T_P", "T_T")
    assert basin.x_b(-50.5) == pytest.approx(-70.0)
    assert basin.x_e(-50.5) == pytest.approx(290.0)
    assert basin.latitude[[0, -1]].tolist() == [-56.0, -44.0]


def test_basin_rejects_nonpositive_width(
    boundary_traces: dict[str, BoundaryTrace],
) -> None:
    east = BoundaryTrace(
        "atlantic_east",
        "east",
        boundary_traces["atlantic_west"].latitude,
        np.full(boundary_traces["atlantic_west"].latitude.size, -80.0),
    )

    with pytest.raises(ValueError, match="east of western"):
        Basin(
            "bad",
            boundary_traces["atlantic_west"],
            east,
            -56.0,
            55.0,
        )


def test_topology_rejects_reversed_child_order(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    basins = list(non_itf_basins)
    basins[3] = replace(basins[3], outflows=("T_A", "T_I"))

    with pytest.raises(ValueError, match="child order"):
        MultiBasinTopology(basins)


def test_topology_rejects_mismatched_gateway(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    basins = list(non_itf_basins)
    basins[2] = replace(basins[2], southern_boundary=-43.0)

    with pytest.raises(ValueError, match="Pacific gateway"):
        MultiBasinTopology(basins)


@pytest.mark.parametrize(
    ("basin_index", "premature_limit", "match"),
    [
        (1, 57.0, "Indian northern closure"),
        (2, 59.0, "Pacific northern closure"),
    ],
)
def test_topology_rejects_premature_northern_closure(
    non_itf_basins: tuple[Basin, ...],
    basin_index: int,
    premature_limit: float,
    match: str,
) -> None:
    basins = list(non_itf_basins)
    basins[basin_index] = replace(
        basins[basin_index], northern_boundary=premature_limit
    )

    with pytest.raises(ValueError, match=match):
        MultiBasinTopology(basins)


def test_topology_distinguishes_nearby_sampled_closure_rows(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    premature_limit = 57.9996

    def insert_sample(trace: BoundaryTrace) -> BoundaryTrace:
        index = int(np.searchsorted(trace.latitude, premature_limit))
        return BoundaryTrace(
            key=trace.key,
            side=trace.side,
            latitude=np.insert(trace.latitude, index, premature_limit),
            longitude=np.insert(
                trace.longitude, index, trace.longitude_at(premature_limit)
            ),
            depth=trace.depth,
            raw_longitude=np.insert(
                trace.raw_longitude, index, trace.longitude_at(premature_limit)
            ),
            valid=np.insert(trace.valid, index, True),
            repaired=np.insert(trace.repaired, index, False),
            provenance=trace.provenance,
        )

    basins = list(non_itf_basins)
    indian_west = insert_sample(basins[1].western_boundary)
    indian_east = insert_sample(basins[1].eastern_boundary)
    basins[1] = replace(
        basins[1],
        western_boundary=indian_west,
        eastern_boundary=indian_east,
        northern_boundary=premature_limit,
    )
    basins[3] = replace(basins[3], eastern_boundary=indian_east)

    with pytest.raises(ValueError, match="Indian northern closure"):
        MultiBasinTopology(basins)


def test_topology_rejects_copied_shared_boundary(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    basins = list(non_itf_basins)
    original = basins[3].eastern_boundary
    copied = BoundaryTrace(
        original.key,
        original.side,
        original.latitude,
        original.longitude,
        depth=original.depth,
        valid=original.valid,
    )
    basins[3] = replace(basins[3], eastern_boundary=copied)

    with pytest.raises(ValueError, match="shared by identity"):
        MultiBasinTopology(basins)


def test_topology_rejects_itf_style_membership(
    non_itf_basins: tuple[Basin, ...],
) -> None:
    basins = list(non_itf_basins)
    basins[-1] = replace(basins[-1], outflows=("T_T", "T_P"))

    with pytest.raises(ValueError, match="child order"):
        MultiBasinTopology(basins)


def test_public_api_exports_scientific_objects() -> None:
    from moc_adjustment_theory import Basin as PublicBasin
    from moc_adjustment_theory import BoundaryTrace as PublicBoundaryTrace
    from moc_adjustment_theory import MultiBasinTopology as PublicTopology

    assert PublicBasin is Basin
    assert PublicBoundaryTrace is BoundaryTrace
    assert PublicTopology is MultiBasinTopology
