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

