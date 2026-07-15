"""Modular theory of meridional-overturning-circulation adjustment."""

from .geometry import BoundaryTrace
from .geometry_io import topology_from_dataset, topology_to_dataset
from .gebco import extract_boundary_traces, topology_from_gebco
from .topology import Basin, MultiBasinTopology

__all__ = [
    "Basin",
    "BoundaryTrace",
    "MultiBasinTopology",
    "extract_boundary_traces",
    "topology_from_gebco",
    "topology_from_dataset",
    "topology_to_dataset",
]
