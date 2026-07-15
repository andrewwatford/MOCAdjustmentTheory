"""Serialization for the boundary traces held by a multi-basin topology."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr

from .geometry import BoundaryTrace
from .topology import BOUNDARY_TRACE_KEYS, MultiBasinTopology


_FORMAT_VERSION = "1"


def _trace_lookup(topology: MultiBasinTopology) -> dict[str, BoundaryTrace]:
    traces: dict[str, BoundaryTrace] = {}
    for basin in topology:
        for trace in (basin.western_boundary, basin.eastern_boundary):
            previous = traces.setdefault(trace.key, trace)
            if previous is not trace:
                raise ValueError(f"trace {trace.key!r} is not shared by identity")
    if set(traces) != set(BOUNDARY_TRACE_KEYS):
        raise ValueError("topology does not contain the fixed six boundary traces")
    return traces


def topology_to_dataset(topology: MultiBasinTopology) -> xr.Dataset:
    """Serialize topology geometry to a compact trace-by-latitude dataset.

    The resulting dataset contains only geometry and provenance. Forcing and
    model results belong in separate datasets.

    Parameters
    ----------
    topology
        Valid fixed five-basin topology.

    Returns
    -------
    xarray.Dataset
        Six shared traces, five basin mappings, masks, and fixed junctions.
    """

    traces = _trace_lookup(topology)
    latitude = np.asarray(traces[BOUNDARY_TRACE_KEYS[0]].latitude)
    if any(
        not np.array_equal(traces[key].latitude, latitude)
        for key in BOUNDARY_TRACE_KEYS[1:]
    ):
        raise ValueError("all six traces must share one exact latitude grid")
    longitude = np.full((len(BOUNDARY_TRACE_KEYS), latitude.size), np.nan)
    longitude_raw = np.full_like(longitude, np.nan)
    valid = np.zeros_like(longitude, dtype=bool)
    repaired = np.zeros_like(longitude, dtype=bool)
    provenance = []

    for index, key in enumerate(BOUNDARY_TRACE_KEYS):
        trace = traces[key]
        longitude[index] = trace.longitude
        longitude_raw[index] = trace.raw_longitude
        valid[index] = trace.valid
        repaired[index] = trace.repaired
        provenance.append(json.dumps(dict(traces[key].provenance), sort_keys=True))

    basin_keys = topology.basin_keys
    basin_west = [topology.basin(key).western_boundary.key for key in basin_keys]
    basin_east = [topology.basin(key).eastern_boundary.key for key in basin_keys]
    basin_south = [topology.basin(key).southern_boundary for key in basin_keys]
    basin_north = [topology.basin(key).northern_boundary for key in basin_keys]

    common_provenance = {
        key: value
        for key, value in traces[BOUNDARY_TRACE_KEYS[0]].provenance.items()
        if all(
            traces[name].provenance.get(key) == value
            for name in BOUNDARY_TRACE_KEYS
        )
    }

    r1 = topology.basin("atlantic_north")
    r4 = topology.basin("atlantic_indian_transition")
    r5 = topology.basin("atlantic_pacific_transition")
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    wrapped[~valid] = np.nan

    return xr.Dataset(
        data_vars={
            "longitude_raw": (("trace", "latitude"), longitude_raw),
            "longitude": (("trace", "latitude"), longitude),
            "longitude_wrapped": (("trace", "latitude"), wrapped),
            "valid": (("trace", "latitude"), valid),
            "repaired": (("trace", "latitude"), repaired),
            "trace_provenance": ("trace", provenance),
            "basin_west_trace": ("basin", basin_west),
            "basin_east_trace": ("basin", basin_east),
            "basin_southern_latitude": ("basin", basin_south),
            "basin_northern_latitude": ("basin", basin_north),
        },
        coords={
            "trace": list(BOUNDARY_TRACE_KEYS),
            "latitude": latitude,
            "basin": list(basin_keys),
        },
        attrs={
            "geometry_format_version": _FORMAT_VERSION,
            "topology": "fixed_non_itf",
            "isobath_depth_m": topology.depth,
            "southern_boundary": r5.southern_boundary,
            "pacific_gateway": r5.northern_boundary,
            "indian_gateway": r4.northern_boundary,
            "atlantic_north": r1.northern_boundary,
            "longitude_convention": (
                "continuous degrees east; longitude_wrapped is display-only"
            ),
            **{
                f"provenance_{key}": value
                for key, value in common_provenance.items()
            },
        },
    )


def topology_from_dataset(dataset: xr.Dataset) -> MultiBasinTopology:
    """Reconstruct a topology from :func:`topology_to_dataset` output.

    Parameters
    ----------
    dataset
        Loaded or lazily opened geometry dataset.

    Returns
    -------
    MultiBasinTopology
        Five basins referencing the reconstructed six shared trace objects.
    """

    version = str(dataset.attrs.get("geometry_format_version", ""))
    if version != _FORMAT_VERSION:
        raise ValueError(f"unsupported geometry format version {version!r}")
    if set(map(str, dataset.trace.values)) != set(BOUNDARY_TRACE_KEYS):
        raise ValueError("dataset must contain the fixed six boundary traces")

    latitude = np.asarray(dataset.latitude.values, dtype=float)
    depth = float(dataset.attrs["isobath_depth_m"])
    traces: dict[str, BoundaryTrace] = {}
    for key in BOUNDARY_TRACE_KEYS:
        selected = dataset.sel(trace=key)
        raw_provenance = str(selected.trace_provenance.item())
        provenance = json.loads(raw_provenance) if raw_provenance else {}
        traces[key] = BoundaryTrace(
            key=key,
            side="west" if key.endswith("_west") else "east",
            latitude=latitude,
            longitude=np.asarray(selected.longitude.values, dtype=float),
            depth=depth,
            raw_longitude=np.asarray(selected.longitude_raw.values, dtype=float),
            valid=np.asarray(selected.valid.values, dtype=bool),
            repaired=np.asarray(selected.repaired.values, dtype=bool),
            provenance=provenance,
        )

    topology = MultiBasinTopology.from_traces(
        traces,
        southern_boundary=float(dataset.attrs["southern_boundary"]),
        pacific_gateway=float(dataset.attrs["pacific_gateway"]),
        indian_gateway=float(dataset.attrs["indian_gateway"]),
        atlantic_north=float(dataset.attrs["atlantic_north"]),
    )
    basin_keys = tuple(map(str, dataset.basin.values))
    if basin_keys != topology.basin_keys:
        raise ValueError("serialized basin order does not match the fixed topology")
    expected_west = tuple(
        topology.basin(key).western_boundary.key for key in basin_keys
    )
    expected_east = tuple(
        topology.basin(key).eastern_boundary.key for key in basin_keys
    )
    expected_south = np.array(
        [topology.basin(key).southern_boundary for key in basin_keys]
    )
    expected_north = np.array(
        [topology.basin(key).northern_boundary for key in basin_keys]
    )
    if tuple(map(str, dataset.basin_west_trace.values)) != expected_west:
        raise ValueError("serialized western trace mapping is inconsistent")
    if tuple(map(str, dataset.basin_east_trace.values)) != expected_east:
        raise ValueError("serialized eastern trace mapping is inconsistent")
    if not np.array_equal(dataset.basin_southern_latitude.values, expected_south):
        raise ValueError("serialized southern basin limits are inconsistent")
    if not np.array_equal(dataset.basin_northern_latitude.values, expected_north):
        raise ValueError("serialized northern basin limits are inconsistent")
    return topology
