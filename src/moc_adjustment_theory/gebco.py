"""GEBCO boundary extraction with native-grid contour refinement."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

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


@dataclass(frozen=True, slots=True)
class _Candidate:
    west: float
    east: float
    west_physical: bool
    east_physical: bool

    @property
    def width(self) -> float:
        return self.east - self.west


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
)

_EXCLUDED_ISLANDS = {
    "indian": ((42.0, 52.5, -27.0, -10.0),),
    "pacific": ((165.0, 180.0, -49.5, -33.0),),
}


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


def _merge_excluded_islands(
    runs: list[tuple[int, int]], longitude: np.ndarray, latitude: float, ocean: str
) -> list[tuple[int, int]]:
    boxes = _EXCLUDED_ISLANDS.get(ocean, ())
    merged = list(runs)
    for west, east, south, north in boxes:
        if not south <= latitude <= north:
            continue
        index = 0
        while index + 1 < len(merged):
            left, right = merged[index], merged[index + 1]
            gap_west = longitude[left[1] + 1]
            gap_east = longitude[right[0] - 1]
            if west <= gap_west and gap_east <= east:
                merged[index : index + 2] = [(left[0], right[1])]
            else:
                index += 1
    return merged


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
) -> dict[str, int]:
    affected: dict[str, int] = {}
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
        affected[name] = int(np.count_nonzero(before & ~deep))
    return affected


def _candidate_rows(
    deep: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    ocean: _Ocean,
    *,
    minimum_width: float,
) -> list[list[_Candidate]]:
    rows: list[list[_Candidate]] = []
    for row, y in zip(deep, latitude):
        candidates = []
        for start, end in _merge_excluded_islands(
            _runs(row), longitude, float(y), ocean.key
        ):
            west_physical = start > 0 and not row[start - 1]
            east_physical = end + 1 < row.size and not row[end + 1]
            west = (
                0.5 * (longitude[start - 1] + longitude[start])
                if west_physical
                else longitude[start]
            )
            east = (
                0.5 * (longitude[end] + longitude[end + 1])
                if east_physical
                else longitude[end]
            )
            if east - west < minimum_width:
                continue
            if y >= ocean.west_south and not west_physical:
                continue
            if y >= ocean.east_south and not east_physical:
                continue
            candidates.append(
                _Candidate(float(west), float(east), west_physical, east_physical)
            )
        rows.append(candidates)
    return rows


def _transition_cost(
    previous: _Candidate, current: _Candidate, latitude: float
) -> float:
    scale = 111.2 * max(np.cos(np.deg2rad(latitude)), 0.2)
    changes = []
    if previous.west_physical and current.west_physical:
        changes.append(abs(current.west - previous.west) * scale)
    if previous.east_physical and current.east_physical:
        changes.append(abs(current.east - previous.east) * scale)
    if not changes:
        changes.append(
            abs(
                0.5 * (current.west + current.east)
                - 0.5 * (previous.west + previous.east)
            )
            * scale
        )
    width_change = abs(current.width - previous.width) * scale
    return float(sum(changes) + 0.2 * width_change)


def _extend_path(
    rows: list[list[_Candidate]],
    latitude: np.ndarray,
    anchor_row: int,
    anchor_candidate: int,
    direction: int,
    *,
    maximum_gap_rows: int,
    maximum_step_km: float,
) -> dict[int, int]:
    selected = {anchor_row: anchor_candidate}
    costs = {anchor_row: np.array([0.0 if i == anchor_candidate else np.inf for i in range(len(rows[anchor_row]))])}
    back: dict[tuple[int, int], tuple[int, int]] = {}
    last_reachable = anchor_row
    indices = range(anchor_row + direction, len(rows) if direction > 0 else -1, direction)

    for row_index in indices:
        if abs(row_index - last_reachable) > maximum_gap_rows + 1:
            break
        if not rows[row_index]:
            continue
        previous_index = last_reachable
        previous_cost = costs[previous_index]
        current_cost = np.full(len(rows[row_index]), np.inf)
        for current_index, current in enumerate(rows[row_index]):
            options = []
            for previous_candidate, previous in enumerate(rows[previous_index]):
                jump = _transition_cost(previous, current, float(latitude[row_index]))
                if jump <= maximum_step_km:
                    options.append((previous_cost[previous_candidate] + jump, previous_candidate))
            if options:
                value, previous_candidate = min(options)
                current_cost[current_index] = value
                back[(row_index, current_index)] = (
                    previous_index,
                    previous_candidate,
                )
        if np.any(np.isfinite(current_cost)):
            costs[row_index] = current_cost
            last_reachable = row_index

    state = int(np.argmin(costs[last_reachable]))
    row_index = last_reachable
    while row_index != anchor_row:
        selected[row_index] = state
        row_index, state = back[(row_index, state)]
    return selected


def _track_candidates(
    rows: list[list[_Candidate]],
    latitude: np.ndarray,
    ocean: _Ocean,
    *,
    maximum_gap_rows: int,
    maximum_step_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    available = np.array([bool(row) for row in rows])
    if not np.any(available):
        raise ValueError(f"no boundary candidates found for the {ocean.key}")
    possible = np.flatnonzero(available)
    anchor_row = int(possible[np.argmin(np.abs(latitude[possible] - ocean.anchor_latitude))])

    def anchor_score(candidate: _Candidate) -> tuple[float, float]:
        distance = max(
            candidate.west - ocean.seed_longitude,
            ocean.seed_longitude - candidate.east,
            0.0,
        )
        return distance, -candidate.width

    anchor_candidate = min(
        range(len(rows[anchor_row])), key=lambda index: anchor_score(rows[anchor_row][index])
    )
    selected = _extend_path(
        rows,
        latitude,
        anchor_row,
        anchor_candidate,
        1,
        maximum_gap_rows=maximum_gap_rows,
        maximum_step_km=maximum_step_km,
    )
    selected.update(
        _extend_path(
            rows,
            latitude,
            anchor_row,
            anchor_candidate,
            -1,
            maximum_gap_rows=maximum_gap_rows,
            maximum_step_km=maximum_step_km,
        )
    )
    west = np.full(latitude.size, np.nan)
    east = np.full(latitude.size, np.nan)
    for row_index, candidate_index in selected.items():
        candidate = rows[row_index][candidate_index]
        if candidate.west_physical:
            west[row_index] = candidate.west
        if candidate.east_physical:
            east[row_index] = candidate.east
    return west, east, *_fill_short_gaps(west, east, maximum_gap_rows)


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

    for start in range(0, guesses.size, band_size):
        end = min(start + band_size, guesses.size)
        y = latitude[start:end]
        guess = guesses[start:end]
        finite = np.isfinite(guess)
        if not np.any(finite):
            continue
        wrapped = (guess[finite] + 180.0) % 360.0 - 180.0
        if np.ptp(wrapped) > 90.0:
            raise ValueError("a refinement band crosses the longitude seam")
        tile = elevation.sel(
            latitude=slice(float(y[finite].min() - 1.1 * dlat), float(y[finite].max() + 1.1 * dlat)),
            longitude=slice(float(wrapped.min() - window), float(wrapped.max() + window)),
        ).compute()
        tile_y = np.asarray(tile.latitude.values, dtype=float)
        tile_x = np.asarray(tile.longitude.values, dtype=float)
        tile_z = np.asarray(tile.values, dtype=float)

        finite_indices = np.flatnonzero(finite)
        for wrapped_index, local_index in enumerate(finite_indices):
            target_y = y[local_index]
            target_guess = wrapped[wrapped_index]
            upper = int(np.searchsorted(tile_y, target_y))
            upper = min(max(upper, 1), tile_y.size - 1)
            lower = upper - 1
            fraction = (target_y - tile_y[lower]) / (tile_y[upper] - tile_y[lower])
            row = (1.0 - fraction) * tile_z[lower] + fraction * tile_z[upper]
            local = np.abs(tile_x - target_guess) <= window
            crossings = _linear_crossings(tile_x[local], row[local], depth)
            if crossings.size:
                crossing = float(crossings[np.argmin(np.abs(crossings - target_guess))])
                original_guess = guess[local_index]
                crossing += 360.0 * round((original_guess - crossing) / 360.0)
                refined[start + local_index] = crossing
    return refined


def _repair_refinement(
    values: np.ndarray, repaired: np.ndarray, maximum_gap_rows: int
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
            raise ValueError("native refinement left an unrepaired boundary gap")
        values[start : end + 1] = np.interp(
            np.arange(start, end + 1),
            [start - 1, end + 1],
            [values[start - 1], values[end + 1]],
        )
        repaired[start : end + 1] = True


def extract_boundary_traces(
    elevation: xr.DataArray,
    *,
    depth: float = 1000.0,
    southern_boundary: float = -56.0,
    pacific_gateway: float = -44.0,
    indian_gateway: float = -35.0,
    atlantic_north: float = 55.0,
    search_north: float = 70.0,
    search_factor: int = 4,
    minimum_width: float = 2.0,
    maximum_gap_rows: int = 2,
    maximum_step_km: float = 500.0,
) -> dict[str, BoundaryTrace]:
    """Extract six shared boundary traces from positive-up bathymetry.

    Coarsened deep-water fractions locate candidate branches. Paired western
    and eastern candidates are tracked continuously, then every selected
    crossing is recomputed from the native elevation grid.

    Parameters
    ----------
    elevation
        Global, cyclic positive-up elevation with latitude/longitude
        coordinates. NumPy- and Dask-backed arrays are both supported.
    depth
        Positive isobath and active-layer depth in metres.
    search_factor
        Integer coarsening factor used only for candidate search.

    Returns
    -------
    dict[str, BoundaryTrace]
        Six traces on one master latitude grid with exact gateway samples.
    """

    if search_factor < 1:
        raise ValueError("search_factor must be a positive integer")
    elevation = _normalise_elevation(elevation)
    native_dlat = float(np.median(np.diff(elevation.latitude.values)))
    work = elevation.sel(
        latitude=slice(southern_boundary - 2 * native_dlat, search_north + 2 * native_dlat)
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
            southern_boundary if ocean.key == "atlantic" else (indian_gateway if ocean.key == "indian" else pacific_gateway),
            east_south,
        )
        window = _window(work, ocean)
        deep = (window <= -float(depth)).astype(np.float32)
        if search_factor > 1:
            deep = deep.coarsen(
                latitude=search_factor,
                longitude=search_factor,
                boundary="trim",
            ).mean()
        deep = deep.compute()
        latitude = np.asarray(deep.latitude.values, dtype=float)
        longitude = np.asarray(deep.longitude.values, dtype=float)
        deep_values = np.asarray(deep.values >= 0.5, dtype=bool)
        counts = _apply_closures(deep_values, latitude, longitude, ocean)
        closure_counts.update(
            {name: closure_counts.get(name, 0) + count for name, count in counts.items()}
        )
        rows = _candidate_rows(
            deep_values,
            latitude,
            longitude,
            ocean,
            minimum_width=minimum_width,
        )
        west, east, repaired_west, repaired_east = _track_candidates(
            rows,
            latitude,
            ocean,
            maximum_gap_rows=maximum_gap_rows,
            maximum_step_km=maximum_step_km,
        )
        tracks[ocean.key] = (latitude, west, east, repaired_west, repaired_east)

    atlantic_end = atlantic_north
    indian_end = float(
        tracks["indian"][0][
            np.flatnonzero(
                np.isfinite(tracks["indian"][1]) & np.isfinite(tracks["indian"][2])
            )[-1]
        ]
    )
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
                np.array([southern_boundary, pacific_gateway, indian_gateway, atlantic_north]),
            ]
        )
    )
    master = master[(master >= southern_boundary) & (master <= max(atlantic_end, indian_end, pacific_end))]

    provenance = {
        "algorithm": "paired_continuity_native_refinement_v1",
        "depth_m": f"{float(depth):g}",
        "search_factor": str(search_factor),
        "closures": ",".join(f"{name}:{count}" for name, count in sorted(closure_counts.items())),
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
        valid = (master >= south) & (master <= north)
        finite = np.isfinite(source_value)
        guesses = np.full(master.size, np.nan)
        guesses[valid] = np.interp(master[valid], source_latitude[finite], source_value[finite])
        repaired = np.zeros(master.size, dtype=bool)
        repaired[np.isin(master, source_latitude[source_repaired])] = True
        final = np.full(master.size, np.nan)
        final[valid] = _refine_guesses(
            elevation, master[valid], guesses[valid], float(depth)
        )
        local_repaired = repaired[valid].copy()
        local_final = final[valid].copy()
        _repair_refinement(local_final, local_repaired, maximum_gap_rows)
        final[valid] = local_final
        repaired[valid] = local_repaired
        raw = final.copy()
        raw[repaired] = np.nan
        traces[key] = BoundaryTrace(
            key=key,
            side=side,
            latitude=master,
            longitude=final,
            depth=depth,
            raw_longitude=raw,
            valid=valid,
            repaired=repaired,
            provenance=provenance,
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
    if verify_checksum and _file_sha256(path) != GEBCO_2026_SHA256:
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
        traces = extract_boundary_traces(dataset.elevation, **extraction_kwargs)
    return MultiBasinTopology.from_traces(
        traces,
        southern_boundary=float(extraction_kwargs.get("southern_boundary", -56.0)),
        pacific_gateway=float(extraction_kwargs.get("pacific_gateway", -44.0)),
        indian_gateway=float(extraction_kwargs.get("indian_gateway", -35.0)),
        atlantic_north=float(extraction_kwargs.get("atlantic_north", 55.0)),
    )
