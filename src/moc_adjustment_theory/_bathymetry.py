"""Generic bathymetry-to-boundary extraction helpers.

This module deliberately contains no named-ocean coordinates.  Basin search
windows, seeds, closures, and ignored features are scientific inputs supplied
by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json

import numpy as np
import xarray as xr
from scipy import ndimage


_EXTRACTION_OPTIONS = {
    "positive",
    "coarsen_factor",
    "maximum_gap_degrees",
    "smoothing_sigma_degrees",
}


def _normalise_elevation(
    elevation: xr.DataArray, *, positive: str | None
) -> tuple[xr.DataArray, bool]:
    rename: dict[str, str] = {}
    if "lat" in elevation.dims:
        rename["lat"] = "latitude"
    if "lon" in elevation.dims:
        rename["lon"] = "longitude"
    elevation = elevation.rename(rename)
    if set(elevation.dims) != {"latitude", "longitude"}:
        raise ValueError("bathymetry must have latitude and longitude dimensions")
    elevation = elevation.transpose("latitude", "longitude")

    units = str(elevation.attrs.get("units", "")).lower().replace(" ", "")
    if units not in {"m", "meter", "meters", "metre", "metres"}:
        raise ValueError("bathymetry units must be metres")

    if positive is None:
        positive = str(elevation.attrs.get("positive", "")).lower() or None
    if positive is None:
        standard_name = str(elevation.attrs.get("standard_name", "")).lower()
        name = str(elevation.name or "").lower()
        if "height" in standard_name or "elevation" in name:
            positive = "up"
        elif "depth" in standard_name or "depth" in name:
            positive = "down"
    if positive not in {"up", "down"}:
        raise ValueError(
            "bathymetry vertical sign is ambiguous; pass extraction_options="
            "{'positive': 'up'} or {'positive': 'down'}"
        )

    latitude = np.asarray(elevation.latitude, dtype=float)
    longitude = np.asarray(elevation.longitude, dtype=float)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("latitude and longitude coordinates must be one-dimensional")
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError("latitude and longitude coordinates must be finite")
    latitude_difference = np.diff(latitude)
    if np.any(latitude_difference == 0) or np.any(np.diff(longitude) == 0):
        raise ValueError("latitude and longitude coordinates must be unique")
    if not (np.all(latitude_difference > 0) or np.all(latitude_difference < 0)):
        raise ValueError("latitude must be strictly monotonic")
    if np.all(latitude_difference < 0):
        elevation = elevation.sortby("latitude")

    wrapped = (np.asarray(elevation.longitude, dtype=float) + 180.0) % 360.0 - 180.0
    order = np.argsort(wrapped)
    wrapped = wrapped[order]
    if np.any(np.diff(wrapped) <= 0):
        raise ValueError("wrapped longitude coordinates must be unique")
    elevation = elevation.isel(longitude=order).assign_coords(longitude=wrapped)
    spacing = float(np.median(np.diff(wrapped)))
    if wrapped[-1] - wrapped[0] + spacing < 359.0:
        raise ValueError("bathymetry must provide cyclic global longitude coverage")
    return elevation, positive == "down"


def _continuous_window(
    elevation: xr.DataArray, bounds: tuple[float, float]
) -> xr.DataArray:
    west, east = map(float, bounds)
    if not west < east or east - west >= 360.0:
        raise ValueError("each longitude window must satisfy west < east < west + 360")
    center = 0.5 * (west + east)
    longitude = np.asarray(elevation.longitude, dtype=float)
    continuous = (longitude - center + 180.0) % 360.0 + center - 180.0
    order = np.argsort(continuous)
    return (
        elevation.isel(longitude=order)
        .assign_coords(longitude=continuous[order])
        .sel(longitude=slice(west, east))
    )


def _densify(
    points: Sequence[Sequence[float]], *, spacing: float, center: float
) -> list[tuple[float, float]]:
    if len(points) < 2:
        raise ValueError("a closure path requires at least two points")
    unwrapped: list[tuple[float, float]] = []
    previous: float | None = None
    for raw_lon, raw_lat in points:
        lon = (float(raw_lon) - center + 180.0) % 360.0 + center - 180.0
        if previous is not None:
            lon += 360.0 * round((previous - lon) / 360.0)
        unwrapped.append((lon, float(raw_lat)))
        previous = lon

    dense: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(unwrapped[:-1], unwrapped[1:]):
        n = max(int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)) / spacing)), 1)
        dense.extend(
            (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            for t in np.linspace(0.0, 1.0, n, endpoint=False)
        )
    dense.append(unwrapped[-1])
    return dense


def _bounded_indices(values: np.ndarray, lower: float, upper: float) -> slice | None:
    start = int(np.searchsorted(values, lower, side="left"))
    stop = int(np.searchsorted(values, upper, side="right"))
    if start >= stop:
        return None
    return slice(start, stop)


def _apply_scientific_masks(
    deep: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    basin: str,
    center: float,
    closures: Sequence[Mapping[str, object]],
    ignored_features: Sequence[Mapping[str, object]],
) -> np.ndarray:
    deep = deep.copy()
    for feature in ignored_features:
        basins = tuple(map(str, feature.get("basins", ())))
        if basin not in basins:
            continue
        if "bounds" not in feature:
            raise ValueError("ignored features currently require rectangular bounds")
        west, east, south, north = map(float, feature["bounds"])  # type: ignore[arg-type]
        if not west < east or not south < north:
            raise ValueError("ignored-feature bounds must be (west, east, south, north)")
        y_slice = _bounded_indices(latitude, south, north)
        x_slice = _bounded_indices(longitude, west, east)
        if y_slice is not None and x_slice is not None:
            deep[y_slice, x_slice] = True

    dlat = float(np.median(np.diff(latitude)))
    dlon = float(np.median(np.diff(longitude)))
    for closure in closures:
        basins = tuple(map(str, closure.get("basins", ())))
        if basin not in basins:
            continue
        width = float(closure.get("width_degrees", 0.0))
        if not np.isfinite(width) or width <= 0:
            raise ValueError("closure width_degrees must be positive")
        points = closure.get("points")
        if not isinstance(points, Sequence):
            raise ValueError("closure points must be a coordinate sequence")
        spacing = max(min(dlat, dlon, width / 2.0), 0.02)
        for lon, lat in _densify(points, spacing=spacing, center=center):
            coslat = max(np.cos(np.deg2rad(lat)), 0.2)
            y_slice = _bounded_indices(latitude, lat - width, lat + width)
            x_radius = width / coslat
            x_slice = _bounded_indices(longitude, lon - x_radius, lon + x_radius)
            if y_slice is None or x_slice is None:
                continue
            local_latitude = latitude[y_slice]
            local_longitude = longitude[x_slice]
            mask = (
                ((local_latitude[:, None] - lat) / width) ** 2
                + ((local_longitude[None, :] - lon) * coslat / width) ** 2
                <= 1.0
            )
            local = deep[y_slice, x_slice]
            local[mask] = False
    return deep


def _crossing(
    x0: float, depth0: float, x1: float, depth1: float, H: float
) -> float:
    if (
        np.isfinite(depth0)
        and np.isfinite(depth1)
        and depth0 != depth1
        and (depth0 - H) * (depth1 - H) <= 0.0
    ):
        return float(x0 + (H - depth0) * (x1 - x0) / (depth1 - depth0))
    return np.nan


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))


def _repair_short_gaps(
    values: np.ndarray, latitude: np.ndarray, maximum_gap_degrees: float
) -> np.ndarray:
    repaired = np.zeros(values.size, dtype=bool)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size < 2:
        return repaired
    internal = ~np.isfinite(values[finite[0] : finite[-1] + 1])
    for local_start, local_end in _runs(internal):
        start = local_start + finite[0]
        end = local_end + finite[0]
        gap = latitude[end + 1] - latitude[start - 1]
        if gap > maximum_gap_degrees:
            raise ValueError(
                f"extraction left an internal boundary gap from "
                f"{latitude[start]:g} to {latitude[end]:g} degrees north"
            )
        values[start : end + 1] = np.interp(
            latitude[start : end + 1],
            [latitude[start - 1], latitude[end + 1]],
            [values[start - 1], values[end + 1]],
        )
        repaired[start : end + 1] = True
    return repaired


def _extract_side(
    depth: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    H: float,
    side: str,
    support: tuple[float, float],
    seed: tuple[float, float],
    maximum_gap_degrees: float,
    smoothing_sigma_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    south, north = support
    rows = np.flatnonzero((latitude >= south) & (latitude <= north))
    if rows.size < 2:
        raise ValueError(f"{side} trace support contains fewer than two rows")
    local_depth = depth[rows]
    deep = local_depth >= H
    labels, count = ndimage.label(deep, structure=np.ones((3, 3), dtype=np.int8))
    if count == 0:
        raise ValueError(f"no deep component found for {side} trace")

    seed_lon, seed_lat = seed
    if not south <= seed_lat <= north:
        raise ValueError(f"seed latitude is outside the {side} trace support")
    iy = int(np.argmin(np.abs(latitude[rows] - seed_lat)))
    ix = int(np.argmin(np.abs(longitude - seed_lon)))
    selected = int(labels[iy, ix])
    if selected == 0:
        raise ValueError(f"seed for {side} trace is not in water deeper than H")

    raw = np.full(latitude.size, np.nan)
    component = labels == selected
    for local_row, row in enumerate(component):
        wet = np.flatnonzero(row)
        if not wet.size:
            continue
        index = int(wet[0] if side == "west" else wet[-1])
        neighbour = index - 1 if side == "west" else index + 1
        if neighbour < 0 or neighbour >= longitude.size:
            continue
        raw[rows[local_row]] = _crossing(
            longitude[neighbour],
            local_depth[local_row, neighbour],
            longitude[index],
            local_depth[local_row, index],
            H,
        )

    values = raw.copy()
    repaired = _repair_short_gaps(values, latitude, maximum_gap_degrees)
    valid = np.isfinite(values) & (latitude >= south) & (latitude <= north)
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        raise ValueError(f"{side} trace has insufficient valid support")
    if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise ValueError(f"{side} trace valid support is not contiguous")

    if smoothing_sigma_degrees > 0:
        dlat = float(np.median(np.diff(latitude[indices])))
        sigma = smoothing_sigma_degrees / dlat
        original_ends = values[indices[[0, -1]]].copy()
        values[indices] = ndimage.gaussian_filter1d(values[indices], sigma=sigma)
        values[indices[[0, -1]]] = original_ends
    return values, raw, repaired


def extract_traces(
    elevation: xr.DataArray,
    *,
    H: float,
    basin_definitions: Mapping[str, Mapping[str, object]],
    trace_support: Mapping[str, tuple[float, float]],
    closures: Sequence[Mapping[str, object]],
    ignored_features: Sequence[Mapping[str, object]],
    extraction_options: Mapping[str, object],
) -> xr.Dataset:
    """Extract explicitly configured continuous boundary traces."""

    unknown_options = sorted(set(extraction_options) - _EXTRACTION_OPTIONS)
    if unknown_options:
        raise ValueError(f"unknown extraction options: {unknown_options}")
    basin_names = set(map(str, basin_definitions))
    for collection_name, collection in (
        ("closure", closures),
        ("ignored feature", ignored_features),
    ):
        for item in collection:
            selected = set(map(str, item.get("basins", ())))
            unknown_basins = sorted(selected - basin_names)
            if unknown_basins:
                raise ValueError(
                    f"{collection_name} references unknown basins: {unknown_basins}"
                )

    positive = extraction_options.get("positive")
    if positive is not None:
        positive = str(positive)
    elevation, positive_down = _normalise_elevation(elevation, positive=positive)

    raw_coarsen_factor = extraction_options.get("coarsen_factor", 1)
    if isinstance(raw_coarsen_factor, bool) or not isinstance(
        raw_coarsen_factor, (int, np.integer)
    ):
        raise ValueError("coarsen_factor must be an integer")
    coarsen_factor = int(raw_coarsen_factor)
    if coarsen_factor < 1:
        raise ValueError("coarsen_factor must be at least one")
    if coarsen_factor > 1:
        elevation = elevation.coarsen(
            latitude=coarsen_factor,
            longitude=coarsen_factor,
            boundary="trim",
        ).mean()

    maximum_gap = float(extraction_options.get("maximum_gap_degrees", 0.25))
    smoothing_sigma = float(extraction_options.get("smoothing_sigma_degrees", 0.0))
    if not np.isfinite(maximum_gap) or maximum_gap < 0:
        raise ValueError("maximum_gap_degrees must be finite and nonnegative")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError("smoothing_sigma_degrees must be finite and nonnegative")
    traces: list[xr.Dataset] = []

    for basin, definition in basin_definitions.items():
        unknown_definition = sorted(set(definition) - {"longitude_bounds", "seed"})
        if unknown_definition:
            raise ValueError(
                f"basin {basin!r} has unknown configuration keys {unknown_definition}"
            )
        if "longitude_bounds" not in definition or "seed" not in definition:
            raise ValueError(
                f"basin {basin!r} requires longitude_bounds and seed"
            )
        bounds = tuple(map(float, definition["longitude_bounds"]))  # type: ignore[arg-type]
        seed = tuple(map(float, definition["seed"]))  # type: ignore[arg-type]
        if len(bounds) != 2 or len(seed) != 2:
            raise ValueError("longitude_bounds and seed must each contain two values")
        if not bounds[0] <= seed[0] <= bounds[1]:
            raise ValueError(f"seed longitude for {basin!r} is outside its window")
        center = 0.5 * (bounds[0] + bounds[1])
        supports = [
            trace_support[f"{basin}_{side}"]
            for side in ("west", "east")
            if f"{basin}_{side}" in trace_support
        ]
        if not supports:
            continue
        south = min(support[0] for support in supports)
        north = max(support[1] for support in supports)
        if not south <= seed[1] <= north:
            raise ValueError(f"seed latitude for {basin!r} is outside trace support")
        native_latitude = np.asarray(elevation.latitude, dtype=float)
        first = max(int(np.searchsorted(native_latitude, south, side="left")) - 1, 0)
        last = min(int(np.searchsorted(native_latitude, north, side="right")) + 1, native_latitude.size)
        local_elevation = elevation.isel(latitude=slice(first, last))
        local_latitude = np.asarray(local_elevation.latitude, dtype=float)
        target_latitude = np.unique(
            np.r_[
                local_latitude,
                np.asarray(supports, dtype=float).ravel(),
                float(seed[1]),
            ]
        )
        target_latitude = target_latitude[
            (target_latitude >= local_latitude[0])
            & (target_latitude <= local_latitude[-1])
        ]
        local_elevation = local_elevation.interp(latitude=target_latitude)
        window = _continuous_window(local_elevation, bounds).compute()
        latitude = np.asarray(window.latitude, dtype=float)
        longitude = np.asarray(window.longitude, dtype=float)
        raw_values = np.asarray(window, dtype=float)
        depth = raw_values if positive_down else -raw_values
        deep = np.isfinite(depth) & (depth >= H)
        deep = _apply_scientific_masks(
            deep,
            latitude,
            longitude,
            basin=basin,
            center=center,
            closures=closures,
            ignored_features=ignored_features,
        )
        epsilon = max(1e-9 * H, np.finfo(float).eps)
        masked_depth = np.where(
            deep,
            np.maximum(depth, H + epsilon),
            np.minimum(depth, H - epsilon),
        )

        for side in ("west", "east"):
            key = f"{basin}_{side}"
            if key not in trace_support:
                continue
            values, raw, repaired = _extract_side(
                masked_depth,
                latitude,
                longitude,
                H=H,
                side=side,
                support=trace_support[key],
                seed=seed,
                maximum_gap_degrees=maximum_gap,
                smoothing_sigma_degrees=smoothing_sigma,
            )
            traces.append(
                xr.Dataset(
                    {
                        "longitude": ("latitude", values),
                        "longitude_raw": ("latitude", raw),
                        "repaired": ("latitude", repaired),
                    },
                    coords={"latitude": latitude},
                ).assign_coords(trace=key)
            )

    if set(map(str, trace_support)) != {str(ds.trace.item()) for ds in traces}:
        raise ValueError("basin definitions did not produce every referenced trace")
    result = xr.concat(traces, dim="trace", join="outer").sortby("latitude")
    result["valid"] = np.isfinite(result.longitude)
    result.attrs["extraction_options"] = json.dumps(
        dict(extraction_options), sort_keys=True, default=str
    )
    return result
