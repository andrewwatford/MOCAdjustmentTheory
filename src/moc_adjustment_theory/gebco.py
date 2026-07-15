"""GEBCO boundary extraction with native-grid contour refinement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import xarray as xr
from scipy import ndimage

from .geometry import BoundaryTrace
from .topology import MultiBasinTopology


GEBCO_2026_DOI = "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"
GEBCO_2026_SHA256 = (
    "9a338345b7a8b8614718ccd551be4be6be629e24cca50f1bc764bdf3ea6e9c3c"
)


@dataclass(frozen=True, slots=True)
class _Ocean:
    key: str
    center: float
    longitude_bounds: tuple[float, float]
    seed_longitude: float
    anchor_latitude: float
    west_south: float
    east_south: float


_OCEANS = (
    _Ocean("atlantic", 0.0, (-100.0, 40.0), -30.0, 0.0, -56.0, -35.0),
    _Ocean("indian", 75.0, (15.0, 155.0), 75.0, -5.0, -35.0, -44.0),
    _Ocean("pacific", 180.0, (100.0, 300.0), 200.0, 0.0, -44.0, -56.0),
)

_CLOSURES = (
    (
        "indonesian_throughflow",
        ((103.0, 1.0), (110.0, -6.5), (121.0, -8.8), (130.0, -10.5), (142.0, -12.0)),
        0.4,
        frozenset({"indian", "pacific"}),
    ),
    (
        "caribbean_bahamas",
        ((-81.5, 24.5), (-77.5, 21.5), (-73.0, 19.0), (-66.5, 18.0), (-61.5, 14.5)),
        0.3,
        frozenset({"atlantic"}),
    ),
    (
        "aleutian_arc",
        (
            (162.0, 56.0),
            (170.0, 53.0),
            (180.0, 51.5),
            (190.0, 52.0),
            (200.0, 54.0),
            (210.0, 58.0),
        ),
        0.4,
        frozenset({"pacific"}),
    ),
)


def _software_version() -> str:
    try:
        return version("moc-adjustment-theory")
    except PackageNotFoundError:
        return "uninstalled"


def _normalise_elevation(elevation: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "lat" in elevation.dims:
        rename["lat"] = "latitude"
    if "lon" in elevation.dims:
        rename["lon"] = "longitude"
    elevation = elevation.rename(rename)
    if set(elevation.dims) != {"latitude", "longitude"}:
        raise ValueError("elevation must have latitude and longitude dimensions")
    elevation = elevation.transpose("latitude", "longitude")

    latitude = np.asarray(elevation.latitude.values, dtype=float)
    longitude = np.asarray(elevation.longitude.values, dtype=float)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("latitude and longitude coordinates must be one-dimensional")
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError("latitude and longitude coordinates must be finite")
    if np.any(np.diff(latitude) == 0) or np.any(np.diff(longitude) == 0):
        raise ValueError("latitude and longitude coordinates must be unique")

    if np.any(np.diff(latitude) < 0):
        elevation = elevation.sortby("latitude")
    longitude = (np.asarray(elevation.longitude.values, dtype=float) + 180.0) % 360.0 - 180.0
    order = np.argsort(longitude)
    longitude = longitude[order]
    if np.any(np.diff(longitude) <= 0):
        raise ValueError("wrapped longitude coordinates must be unique")
    elevation = elevation.isel(longitude=order).assign_coords(longitude=longitude)

    spacing = float(np.median(np.diff(longitude)))
    if longitude[-1] - longitude[0] + spacing < 359.0:
        raise ValueError("elevation must provide cyclic global longitude coverage")
    return elevation


def _window(elevation: xr.DataArray, ocean: _Ocean) -> xr.DataArray:
    longitude = np.asarray(elevation.longitude.values)
    continuous = (longitude - ocean.center + 180.0) % 360.0 + ocean.center - 180.0
    order = np.argsort(continuous)
    window = elevation.isel(longitude=order).assign_coords(
        longitude=continuous[order]
    )
    return window.sel(longitude=slice(*ocean.longitude_bounds))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _densify(
    points: Iterable[tuple[float, float]], spacing: float
) -> list[tuple[float, float]]:
    points = list(points)
    dense: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        dx = (x1 - x0 + 180.0) % 360.0 - 180.0
        steps = max(int(np.ceil(max(abs(dx), abs(y1 - y0)) / spacing)), 1)
        dense.extend(
            (x0 + fraction * dx, y0 + fraction * (y1 - y0))
            for fraction in np.linspace(0.0, 1.0, steps, endpoint=False)
        )
    dense.append(points[-1])
    return dense


def _apply_closures(
    deep: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, ocean: _Ocean
) -> tuple[dict[str, int], np.ndarray]:
    affected: dict[str, int] = {}
    protected = np.zeros_like(deep, dtype=bool)
    dlat = float(np.median(np.diff(latitude)))
    dlon = float(np.median(np.diff(longitude)))
    for name, points, width, oceans in _CLOSURES:
        if ocean.key not in oceans:
            continue
        before = deep.copy()
        for raw_x, y in _densify(points, spacing=max(min(dlat, dlon), 0.05)):
            x = (raw_x - ocean.center + 180.0) % 360.0 + ocean.center - 180.0
            y_index = int(np.argmin(np.abs(latitude - y)))
            x_index = int(np.argmin(np.abs(longitude - x)))
            y_radius = max(int(np.ceil(width / dlat)), 1)
            x_radius = max(
                int(np.ceil(width / (dlon * max(np.cos(np.deg2rad(y)), 0.2)))),
                1,
            )
            deep[
                max(0, y_index - y_radius) : y_index + y_radius + 1,
                max(0, x_index - x_radius) : x_index + x_radius + 1,
            ] = False
        newly_closed = before & ~deep
        protected |= newly_closed
        affected[name] = int(np.count_nonzero(newly_closed))
    return affected, protected


def _component_boundary(
    deep: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    ocean: _Ocean,
    side: str,
    southern_limit: float,
    protected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one outer edge of the seeded 2-D ocean component.

    The component is labelled independently for each boundary after clipping
    away latitudes south of that boundary's gateway.  This prevents a route
    through the Southern Ocean from joining nominally separate basins.  All
    row runs carrying the selected component label are treated as one basin,
    so islands and internal ridges do not become basin boundaries.
    """

    start = int(np.argmin(np.abs(latitude - southern_limit)))
    active = deep[start:]
    labels, count = ndimage.label(active, structure=np.ones((3, 3), dtype=np.int8))
    if count == 0:
        raise ValueError(f"no deep component found for {ocean.key}_{side}")

    anchor = int(np.argmin(np.abs(latitude[start:] - ocean.anchor_latitude)))
    deep_at_anchor = np.flatnonzero(active[anchor])
    if not deep_at_anchor.size:
        raise ValueError(f"no deep anchor row found for {ocean.key}_{side}")
    seed = int(
        deep_at_anchor[
            np.argmin(np.abs(longitude[deep_at_anchor] - ocean.seed_longitude))
        ]
    )
    selected_label = int(labels[anchor, seed])
    if selected_label == 0:
        raise ValueError(f"the {ocean.key}_{side} seed is not in deep water")

    component = labels == selected_label
    values = np.full(latitude.size, np.nan)
    closure_edge = np.zeros(latitude.size, dtype=bool)
    for offset, row in enumerate(component):
        indices = np.flatnonzero(row)
        if not indices.size:
            continue
        index = int(indices[0] if side == "west" else indices[-1])
        neighbour = index - 1 if side == "west" else index + 1
        if neighbour < 0 or neighbour >= longitude.size or deep[start + offset, neighbour]:
            continue
        values[start + offset] = 0.5 * (longitude[index] + longitude[neighbour])
        closure_edge[start + offset] = protected[start + offset, neighbour]
    return values, closure_edge


def _fill_short_gaps(
    west: np.ndarray, east: np.ndarray, maximum_gap_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    repaired_west = np.zeros(west.size, dtype=bool)
    repaired_east = np.zeros(east.size, dtype=bool)
    for values, repaired in ((west, repaired_west), (east, repaired_east)):
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size < 2:
            continue
        for start, end in _runs(~np.isfinite(values[finite[0] : finite[-1] + 1])):
            start += finite[0]
            end += finite[0]
            if end - start + 1 <= maximum_gap_rows:
                values[start : end + 1] = np.interp(
                    np.arange(start, end + 1),
                    [start - 1, end + 1],
                    [values[start - 1], values[end + 1]],
                )
                repaired[start : end + 1] = True
    return repaired_west, repaired_east


def _linear_crossings(
    longitude: np.ndarray, elevation: np.ndarray, depth: float
) -> np.ndarray:
    residual = elevation + depth
    valid = np.isfinite(residual[:-1]) & np.isfinite(residual[1:])
    bracket = valid & (residual[:-1] * residual[1:] <= 0.0) & (
        residual[:-1] != residual[1:]
    )
    index = np.flatnonzero(bracket)
    if not index.size:
        return np.empty(0)
    return longitude[index] - residual[index] * (
        longitude[index + 1] - longitude[index]
    ) / (residual[index + 1] - residual[index])


def _refine_guesses(
    elevation: xr.DataArray,
    latitude: np.ndarray,
    guesses: np.ndarray,
    depth: float,
    *,
    band_size: int = 256,
    window: float = 0.3,
) -> np.ndarray:
    refined = np.full(guesses.size, np.nan)
    native_latitude = np.asarray(elevation.latitude.values, dtype=float)
    native_longitude = np.asarray(elevation.longitude.values, dtype=float)
    dlat = float(np.median(np.diff(native_latitude)))
    dlon = float(np.median(np.diff(native_longitude)))
    window = max(window, 1.5 * dlon)

    def tile_for(
        y_min: float, y_max: float, x_min: float, x_max: float
    ) -> xr.DataArray:
        latitude_slice = slice(y_min - 1.1 * dlat, y_max + 1.1 * dlat)
        if x_min < -180.0:
            left = elevation.sel(
                latitude=latitude_slice,
                longitude=slice(x_min + 360.0, 180.0),
            ).assign_coords(longitude=lambda value: value.longitude - 360.0)
            right = elevation.sel(
                latitude=latitude_slice,
                longitude=slice(-180.0, x_max),
            )
            return xr.concat((left, right), dim="longitude").compute()
        if x_max > 180.0:
            left = elevation.sel(
                latitude=latitude_slice,
                longitude=slice(x_min, 180.0),
            )
            right = elevation.sel(
                latitude=latitude_slice,
                longitude=slice(-180.0, x_max - 360.0),
            ).assign_coords(longitude=lambda value: value.longitude + 360.0)
            return xr.concat((left, right), dim="longitude").compute()
        return elevation.sel(
            latitude=latitude_slice,
            longitude=slice(x_min, x_max),
        ).compute()

    for start in range(0, guesses.size, band_size):
        end = min(start + band_size, guesses.size)
        y = latitude[start:end]
        guess = guesses[start:end]
        finite = np.isfinite(guess)
        if not np.any(finite):
            continue
        wrapped = (guess[finite] + 180.0) % 360.0 - 180.0
        finite_indices = np.flatnonzero(finite)
        breaks = np.r_[0, np.flatnonzero(np.abs(np.diff(wrapped)) > 180.0) + 1, wrapped.size]
        for group_start, group_end in zip(breaks[:-1], breaks[1:]):
            group_indices = finite_indices[group_start:group_end]
            group_wrapped = wrapped[group_start:group_end]
            if np.ptp(group_wrapped) > 90.0:
                raise ValueError("a refinement band is not longitude-continuous")
            group_y = y[group_indices]
            tile = tile_for(
                float(group_y.min()),
                float(group_y.max()),
                float(group_wrapped.min() - window),
                float(group_wrapped.max() + window),
            )
            tile_y = np.asarray(tile.latitude.values, dtype=float)
            tile_x = np.asarray(tile.longitude.values, dtype=float)
            tile_z = np.asarray(tile.values, dtype=float)

            for target_index, local_index in enumerate(group_indices):
                target_y = y[local_index]
                target_guess = group_wrapped[target_index]
                upper = int(np.searchsorted(tile_y, target_y))
                upper = min(max(upper, 1), tile_y.size - 1)
                lower = upper - 1
                fraction = (target_y - tile_y[lower]) / (
                    tile_y[upper] - tile_y[lower]
                )
                row = (1.0 - fraction) * tile_z[lower] + fraction * tile_z[upper]
                local = np.abs(tile_x - target_guess) <= window
                crossings = _linear_crossings(tile_x[local], row[local], depth)
                if crossings.size:
                    crossing = float(
                        crossings[np.argmin(np.abs(crossings - target_guess))]
                    )
                    original_guess = guess[local_index]
                    crossing += 360.0 * round(
                        (original_guess - crossing) / 360.0
                    )
                    refined[start + local_index] = crossing
    return refined


def _repair_refinement(
    values: np.ndarray,
    repaired: np.ndarray,
    maximum_gap_rows: int,
    *,
    latitude: np.ndarray | None = None,
    label: str = "boundary",
) -> None:
    missing = ~np.isfinite(values)
    if not np.any(missing):
        return
    for start, end in _runs(missing):
        length = end - start + 1
        if (
            length > maximum_gap_rows
            or start == 0
            or end == values.size - 1
            or not np.isfinite(values[start - 1])
            or not np.isfinite(values[end + 1])
        ):
            location = (
                f" from {latitude[start]:g} to {latitude[end]:g} degrees north"
                if latitude is not None
                else ""
            )
            raise ValueError(
                f"native refinement left an unrepaired {label} gap{location}"
            )
        values[start : end + 1] = np.interp(
            np.arange(start, end + 1),
            [start - 1, end + 1],
            [values[start - 1], values[end + 1]],
        )
        repaired[start : end + 1] = True


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    median = np.full(values.size, np.nan)
    half = window // 2
    for index in range(values.size):
        local = values[max(0, index - half) : index + half + 1]
        local = local[np.isfinite(local)]
        if local.size >= 3:
            median[index] = np.median(local)
    return median


def _regularization_parameters(key: str) -> dict[str, float]:
    parameters = {
        "rolling_window_degrees": 205.0 / 60.0,
        "outlier_threshold_degrees": 6.0,
        "maximum_step_degrees": 7.0,
        "maximum_gap_degrees": 4.0,
        "smoothing_sigma_degrees": 0.35,
    }
    ocean, side = key.split("_")
    if ocean == "atlantic":
        parameters.update(
            outlier_threshold_degrees=5.0,
            maximum_step_degrees=5.0,
            smoothing_sigma_degrees=0.45,
        )
        if side == "west":
            parameters["maximum_gap_degrees"] = 6.0
    elif ocean == "indian":
        parameters.update(
            outlier_threshold_degrees=5.0,
            maximum_step_degrees=5.0,
            smoothing_sigma_degrees=0.45,
        )
    elif ocean == "pacific" and side == "west":
        parameters.update(
            outlier_threshold_degrees=5.0,
            maximum_step_degrees=5.0,
            maximum_gap_degrees=5.0,
            smoothing_sigma_degrees=0.55,
        )
    return parameters


def _regularize_boundary(
    latitude: np.ndarray,
    values: np.ndarray,
    key: str,
    anchor_latitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, float]]:
    """Remove branch switches and lightly smooth one single-valued trace."""

    parameters = _regularization_parameters(key)
    cleaned = values.copy()
    repaired = np.zeros(values.size, dtype=bool)
    spacing = float(np.median(np.diff(latitude)))
    window = max(
        int(round(parameters["rolling_window_degrees"] / spacing)), 3
    )
    if window % 2 == 0:
        window += 1

    for _ in range(3):
        rolling = _rolling_median(cleaned, window)
        outlier = (
            np.isfinite(cleaned)
            & np.isfinite(rolling)
            & (
                np.abs(cleaned - rolling)
                > parameters["outlier_threshold_degrees"]
            )
        )
        cleaned[outlier] = np.nan
        repaired |= outlier

        finite = np.flatnonzero(np.isfinite(cleaned))
        if finite.size < 2:
            continue
        jumps = (
            np.abs(np.diff(cleaned[finite]))
            > parameters["maximum_step_degrees"]
        )
        for jump in np.flatnonzero(jumps):
            left = int(finite[jump])
            right = int(finite[jump + 1])
            left_score = (
                abs(cleaned[left] - rolling[left])
                if np.isfinite(rolling[left])
                else np.inf
            )
            right_score = (
                abs(cleaned[right] - rolling[right])
                if np.isfinite(rolling[right])
                else np.inf
            )
            remove = left if left_score > right_score else right
            cleaned[remove] = np.nan
            repaired[remove] = True

    finite = np.flatnonzero(np.isfinite(cleaned))
    if finite.size < 2:
        raise ValueError(f"regularization removed the full {key} trace")
    first, last = int(finite[0]), int(finite[-1])
    for start, end in _runs(~np.isfinite(cleaned[first : last + 1])):
        start += first
        end += first
        gap_span = latitude[end + 1] - latitude[start - 1]
        if gap_span <= parameters["maximum_gap_degrees"]:
            cleaned[start : end + 1] = np.interp(
                latitude[start : end + 1],
                [latitude[start - 1], latitude[end + 1]],
                [cleaned[start - 1], cleaned[end + 1]],
            )
            repaired[start : end + 1] = True

    segments = _runs(np.isfinite(cleaned))
    selected = min(
        segments,
        key=lambda bounds: max(
            latitude[bounds[0]] - anchor_latitude,
            anchor_latitude - latitude[bounds[1]],
            0.0,
        ),
    )
    keep = np.zeros(values.size, dtype=bool)
    keep[selected[0] : selected[1] + 1] = True
    cleaned[~keep] = np.nan
    repaired &= keep

    sigma = parameters["smoothing_sigma_degrees"] / spacing
    segment = cleaned[selected[0] : selected[1] + 1]
    if segment.size >= 7:
        smoothed = ndimage.gaussian_filter1d(segment, sigma=sigma, mode="nearest")
        smoothed[0] = segment[0]
        smoothed[-1] = segment[-1]
        cleaned[selected[0] : selected[1] + 1] = smoothed
    regularized = (
        np.isfinite(values)
        & np.isfinite(cleaned)
        & ~np.isclose(values, cleaned, rtol=0.0, atol=1e-12)
    )
    return cleaned, repaired, regularized, parameters


def _latitude_ranges(latitude: np.ndarray, mask: np.ndarray) -> str:
    ranges = [
        [float(latitude[start]), float(latitude[end])]
        for start, end in _runs(mask)
    ]
    return json.dumps(ranges)


def extract_boundary_traces(
    elevation: xr.DataArray,
    *,
    depth: float = 1000.0,
    southern_boundary: float = -56.0,
    pacific_gateway: float = -44.0,
    indian_gateway: float = -35.0,
    atlantic_north: float = 55.0,
    indian_north: float = 20.0,
    search_north: float = 70.0,
    search_factor: int = 4,
    deep_fraction_threshold: float = 1.0,
    maximum_gap_rows: int = 2,
    provenance: Mapping[str, str] | None = None,
) -> dict[str, BoundaryTrace]:
    """Extract six shared boundary traces from positive-up bathymetry.

    Coarsened deep-water fractions identify seeded two-dimensional ocean
    components independently above each gateway.  Their rowwise outer edges
    are then recomputed from the native elevation grid.

    Parameters
    ----------
    elevation
        Global, cyclic positive-up elevation with latitude/longitude
        coordinates. NumPy- and Dask-backed arrays are both supported.
    depth
        Positive isobath and active-layer depth in metres.
    indian_north
        Closed Indian northern latitude. The default stops before the Bay of
        Bengal edge jumps from Southeast Asia onto India.
    search_factor
        Integer coarsening factor used only for candidate search.

    Returns
    -------
    dict[str, BoundaryTrace]
        Six traces on one master latitude grid with exact gateway samples.
    """

    if search_factor < 1:
        raise ValueError("search_factor must be a positive integer")
    if not 0.0 < deep_fraction_threshold <= 1.0:
        raise ValueError("deep_fraction_threshold must lie in (0, 1]")
    elevation = _normalise_elevation(elevation)
    native_dlat = float(np.median(np.diff(elevation.latitude.values)))
    work = elevation.sel(
        latitude=slice(
            southern_boundary - 2 * native_dlat,
            search_north + 2 * native_dlat,
        )
    )
    tracks = {}
    closure_counts: dict[str, int] = {}

    for ocean in _OCEANS:
        east_south = {
            "atlantic": indian_gateway,
            "indian": pacific_gateway,
            "pacific": southern_boundary,
        }[ocean.key]
        ocean = _Ocean(
            ocean.key,
            ocean.center,
            ocean.longitude_bounds,
            ocean.seed_longitude,
            ocean.anchor_latitude,
            (
                southern_boundary
                if ocean.key == "atlantic"
                else indian_gateway if ocean.key == "indian" else pacific_gateway
            ),
            east_south,
        )
        window = _window(work, ocean)
        deep_fraction = (window <= -float(depth)).astype(np.float32)
        if search_factor > 1:
            deep_fraction = deep_fraction.coarsen(
                latitude=search_factor,
                longitude=search_factor,
                boundary="trim",
            ).mean()
        deep = (deep_fraction >= deep_fraction_threshold).compute()
        latitude = np.asarray(deep.latitude.values, dtype=float)
        longitude = np.asarray(deep.longitude.values, dtype=float)
        deep_values = np.asarray(deep.values, dtype=bool)
        counts, protected = _apply_closures(
            deep_values, latitude, longitude, ocean
        )
        closure_counts.update(
            {
                name: closure_counts.get(name, 0) + count
                for name, count in counts.items()
            }
        )
        west, west_closure = _component_boundary(
            deep_values,
            latitude,
            longitude,
            ocean,
            "west",
            ocean.west_south,
            protected,
        )
        east, east_closure = _component_boundary(
            deep_values,
            latitude,
            longitude,
            ocean,
            "east",
            ocean.east_south,
            protected,
        )
        repaired_west, repaired_east = _fill_short_gaps(
            west, east, maximum_gap_rows
        )
        tracks[ocean.key] = (
            latitude,
            west,
            east,
            repaired_west,
            repaired_east,
            west_closure & np.isfinite(west),
            east_closure & np.isfinite(east),
        )

    atlantic_end = atlantic_north
    # North of 20 N the Bay of Bengal branch disappears and the row-wise
    # eastern edge jumps onto India. Close the model basin while its eastern
    # wall is still Southeast Asia instead of treating India as that wall.
    indian_end = float(indian_north)
    pacific_end = float(
        tracks["pacific"][0][
            np.flatnonzero(
                np.isfinite(tracks["pacific"][1]) & np.isfinite(tracks["pacific"][2])
            )[-1]
        ]
    )
    master = np.unique(
        np.concatenate(
            [
                *(track[0] for track in tracks.values()),
                np.array(
                    [
                        southern_boundary,
                        pacific_gateway,
                        indian_gateway,
                        atlantic_north,
                        indian_north,
                    ]
                ),
            ]
        )
    )
    master = master[
        (master >= southern_boundary)
        & (master <= max(atlantic_end, indian_end, pacific_end))
    ]

    closure_definitions = {
        name: {
            "points": points,
            "width_degrees": width,
            "oceans": sorted(oceans),
        }
        for name, points, width, oceans in _CLOSURES
    }
    extraction_configuration = {
        "depth_m": depth,
        "southern_boundary": southern_boundary,
        "pacific_gateway": pacific_gateway,
        "indian_gateway": indian_gateway,
        "atlantic_north": atlantic_north,
        "indian_north": indian_north,
        "search_north": search_north,
        "search_factor": search_factor,
        "deep_fraction_threshold": deep_fraction_threshold,
        "maximum_gap_rows": maximum_gap_rows,
    }
    common_provenance = {
        **({} if provenance is None else dict(provenance)),
        "algorithm": "connected_component_native_refinement_v2",
        "depth_m": f"{float(depth):g}",
        "search_factor": str(search_factor),
        "deep_fraction_threshold": f"{deep_fraction_threshold:g}",
        "closure_snap_window_degrees": "2",
        "closure_affected_cells": json.dumps(closure_counts, sort_keys=True),
        "closure_definitions": json.dumps(closure_definitions, sort_keys=True),
        "extraction_configuration": json.dumps(
            extraction_configuration, sort_keys=True
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software_version": _software_version(),
        "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
        "source": str(elevation.attrs.get("source", elevation.name or "elevation")),
    }
    definitions = {
        "atlantic_west": ("atlantic", 1, southern_boundary, atlantic_end, "west"),
        "atlantic_east": ("atlantic", 2, indian_gateway, atlantic_end, "east"),
        "indian_west": ("indian", 1, indian_gateway, indian_end, "west"),
        "indian_east": ("indian", 2, pacific_gateway, indian_end, "east"),
        "pacific_west": ("pacific", 1, pacific_gateway, pacific_end, "west"),
        "pacific_east": ("pacific", 2, southern_boundary, pacific_end, "east"),
    }
    traces = {}
    for key, (ocean, value_index, south, north, side) in definitions.items():
        source_latitude = tracks[ocean][0]
        source_value = tracks[ocean][value_index]
        source_repaired = tracks[ocean][value_index + 2]
        source_closure = tracks[ocean][value_index + 4]
        valid = (master >= south) & (master <= north)
        finite = np.isfinite(source_value)
        finite_latitude = source_latitude[finite]
        source_spacing = float(np.median(np.diff(source_latitude)))
        if (
            not finite_latitude.size
            or finite_latitude[0] > south + 1.01 * source_spacing
            or finite_latitude[-1] < north - 1.01 * source_spacing
        ):
            raise ValueError(
                f"coarse component does not cover the full {key} interval"
            )
        source_first = int(np.flatnonzero(finite)[0])
        source_last = int(np.flatnonzero(finite)[-1])
        if np.any(~finite[source_first : source_last + 1]):
            raise ValueError(f"coarse component has an internal {key} gap")
        guesses = np.full(master.size, np.nan)
        guesses[valid] = np.interp(
            master[valid], finite_latitude, source_value[finite]
        )
        closure_edge = np.interp(
            master,
            source_latitude,
            source_closure.astype(float),
        ) > 0.0
        repaired = np.zeros(master.size, dtype=bool)
        native = np.full(master.size, np.nan)
        native[valid] = _refine_guesses(
            elevation, master[valid], guesses[valid], float(depth)
        )
        closure_valid = valid & closure_edge
        if np.any(closure_valid & ~np.isfinite(native)):
            retry = closure_valid & ~np.isfinite(native)
            native[retry] = _refine_guesses(
                elevation,
                master[retry],
                guesses[retry],
                float(depth),
                window=2.0,
            )
        raw = native.copy()
        local_repaired = repaired[valid].copy()
        local_final = native[valid].copy()
        _repair_refinement(
            local_final,
            local_repaired,
            maximum_gap_rows,
            latitude=master[valid],
            label=key,
        )
        native[valid] = local_final
        repaired[valid] = local_repaired
        ocean_index = {"atlantic": 0, "indian": 1, "pacific": 2}[ocean]
        final, branch_repaired, regularized, regularization = _regularize_boundary(
            master, native, key, _OCEANS[ocean_index].anchor_latitude
        )
        valid &= np.isfinite(final)
        repaired |= branch_repaired | (closure_valid & valid)
        repaired &= valid

        final_indices = np.flatnonzero(valid)
        final_steps = np.abs(np.diff(final[final_indices]))
        maximum_final_step = float(np.max(final_steps, initial=0.0))
        final_step_km = (
            final_steps
            * 111.2
            * np.maximum(
                np.cos(np.deg2rad(master[final_indices[1:]])), 0.2
            )
        )
        maximum_final_step_km = float(np.max(final_step_km, initial=0.0))
        if maximum_final_step_km > 120.0:
            step_index = int(np.argmax(final_step_km))
            step_latitude = float(master[final_indices[step_index + 1]])
            raise ValueError(
                f"regularized {key} has a {maximum_final_step_km:g}-km branch jump"
                f" at {step_latitude:g} degrees north"
            )
        comparable = valid & np.isfinite(raw)
        displacement_km = (
            np.abs(final[comparable] - raw[comparable])
            * 111.2
            * np.maximum(np.cos(np.deg2rad(master[comparable])), 0.2)
        )
        closure_comparable = closure_valid & np.isfinite(raw)
        closure_displacement_km = (
            np.abs(raw[closure_comparable] - guesses[closure_comparable])
            * 111.2
            * np.maximum(
                np.cos(np.deg2rad(master[closure_comparable])), 0.2
            )
        )
        native_audit = np.full(master.size, np.nan)
        native_audit[comparable] = raw[comparable]
        audit = valid & (~np.isfinite(raw) | (np.abs(final - raw) > 0.3))
        if np.any(audit):
            native_audit[audit] = _refine_guesses(
                elevation,
                master[audit],
                final[audit],
                float(depth),
                window=3.0,
            )
        native_comparable = valid & np.isfinite(native_audit)
        native_missing = valid & ~np.isfinite(native_audit)
        native_displacement_km = (
            np.abs(final[native_comparable] - native_audit[native_comparable])
            * 111.2
            * np.maximum(
                np.cos(np.deg2rad(master[native_comparable])), 0.2
            )
        )
        trace_provenance = {
            **common_provenance,
            "coarse_search_repaired_rows": str(np.count_nonzero(source_repaired)),
            "closure_snap_rows": str(np.count_nonzero(closure_valid)),
            "closure_derived_latitude_ranges": _latitude_ranges(
                master, closure_valid & valid
            ),
            "closure_snap_displacement_km_max": f"{np.max(closure_displacement_km, initial=0.0):g}",
            "branch_repaired_latitude_ranges": _latitude_ranges(
                master, branch_repaired & valid
            ),
            "regularization_parameters": json.dumps(
                regularization, sort_keys=True
            ),
            "regularized_rows": str(np.count_nonzero(regularized & valid)),
            "regularization_displacement_km_p90": f"{np.percentile(displacement_km, 90) if displacement_km.size else 0.0:g}",
            "regularization_displacement_km_max": f"{np.max(displacement_km, initial=0.0):g}",
            "native_isobath_displacement_km_p90": f"{np.percentile(native_displacement_km, 90):g}",
            "native_isobath_displacement_km_max": f"{np.max(native_displacement_km, initial=0.0):g}",
            "native_isobath_missing_rows": str(np.count_nonzero(native_missing)),
            "native_isobath_missing_latitude_ranges": _latitude_ranges(
                master, native_missing
            ),
            "maximum_final_step_degrees": f"{maximum_final_step:g}",
            "maximum_final_step_km": f"{maximum_final_step_km:g}",
        }
        traces[key] = BoundaryTrace(
            key=key,
            side=side,
            latitude=master,
            longitude=final,
            depth=depth,
            raw_longitude=raw,
            valid=valid,
            repaired=repaired,
            provenance=trace_provenance,
        )
    return traces


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topology_from_gebco(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    chunks: tuple[int, int] = (240, 2400),
    **extraction_kwargs,
) -> MultiBasinTopology:
    """Extract and assemble the production topology from GEBCO 2026.

    This strict wrapper validates the published source identity and uses Dask
    chunks while :func:`extract_boundary_traces` processes bounded ocean
    windows and native-refinement bands.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_checksum = _file_sha256(path) if verify_checksum else "not_verified"
    if verify_checksum and actual_checksum != GEBCO_2026_SHA256:
        raise ValueError("GEBCO source checksum does not match the manifest")
    with xr.open_dataset(
        path,
        engine="h5netcdf",
        chunks={"lat": chunks[0], "lon": chunks[1]},
    ) as dataset:
        if dataset.sizes.get("lat") != 43200 or dataset.sizes.get("lon") != 86400:
            raise ValueError("GEBCO source must have lat=43200 and lon=86400")
        if "elevation" not in dataset:
            raise ValueError("GEBCO source must contain elevation(lat, lon)")
        identity = " ".join(
            str(dataset.attrs.get(name, ""))
            for name in ("id", "identifier_product_doi", "references")
        )
        if GEBCO_2026_DOI not in identity:
            raise ValueError("GEBCO source DOI is not the expected 2026 product")
        if dataset.elevation.attrs.get("units") != "m":
            raise ValueError("GEBCO elevation units must be metres")
        traces = extract_boundary_traces(
            dataset.elevation,
            provenance={
                "source_product": "GEBCO 2026 sub-ice global grid",
                "source_doi": GEBCO_2026_DOI,
                "source_sha256": actual_checksum,
                "source_checksum_verified": str(verify_checksum).lower(),
                "source_path": str(path.resolve()),
                "source_dimensions": "lat=43200,lon=86400",
            },
            **extraction_kwargs,
        )
    return MultiBasinTopology.from_traces(
        traces,
        southern_boundary=float(extraction_kwargs.get("southern_boundary", -56.0)),
        pacific_gateway=float(extraction_kwargs.get("pacific_gateway", -44.0)),
        indian_gateway=float(extraction_kwargs.get("indian_gateway", -35.0)),
        atlantic_north=float(extraction_kwargs.get("atlantic_north", 55.0)),
    )
