"""Xarray-backed geometry for the fixed five-region global model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json

import numpy as np
import xarray as xr

REGION_KEYS = (
    "atlantic_north",
    "indian_north",
    "pacific_north",
    "atlantic_indian_transition",
    "atlantic_pacific_transition",
)

TRACE_NAMES = (
    "atlantic_west",
    "atlantic_east",
    "indian_west",
    "indian_east",
    "pacific_west",
    "pacific_east",
)

_REGION_METADATA_VARIABLES = (
    "region_west_trace",
    "region_east_trace",
    "region_south",
    "region_north",
)


def _embedded_region_definitions(
    dataset: xr.Dataset,
) -> dict[str, dict[str, object]] | None:
    present = [name in dataset for name in _REGION_METADATA_VARIABLES]
    if not any(present):
        return None
    if not all(present) or "region" not in dataset.coords:
        raise ValueError(
            "isobath dataset contains incomplete region metadata; expected "
            "region coordinate plus west/east trace and south/north variables"
        )
    if set(map(str, dataset.region.values)) != set(REGION_KEYS):
        raise ValueError("isobath dataset region metadata must define the five regions")
    return {
        region: {
            "west": str(dataset.region_west_trace.sel(region=region).item()),
            "east": str(dataset.region_east_trace.sel(region=region).item()),
            "south": float(dataset.region_south.sel(region=region)),
            "north": float(dataset.region_north.sel(region=region)),
        }
        for region in REGION_KEYS
    }


def _normalise_regions(
    definitions: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if set(definitions) != set(REGION_KEYS):
        missing = sorted(set(REGION_KEYS) - set(definitions))
        extra = sorted(set(definitions) - set(REGION_KEYS))
        raise ValueError(
            "region_definitions must contain the fixed five model regions; "
            f"missing={missing}, extra={extra}"
        )
    normalised: dict[str, dict[str, object]] = {}
    for key in REGION_KEYS:
        definition = definitions[key]
        west = str(definition.get("west", "")).strip()
        east = str(definition.get("east", "")).strip()
        if not west.endswith("_west") or not east.endswith("_east"):
            raise ValueError(f"region {key!r} must reference west/east trace names")
        south = float(definition["south"])
        if "north" not in definition:
            raise ValueError(f"region {key!r} requires an explicit north bound")
        north = float(definition["north"])
        if not np.isfinite(south) or not np.isfinite(north) or south >= north:
            raise ValueError(f"region {key!r} requires finite south < north")
        normalised[key] = {
            "west": west,
            "east": east,
            "south": south,
            "north": north,
        }
    r1 = normalised["atlantic_north"]
    r2 = normalised["indian_north"]
    r3 = normalised["pacific_north"]
    r4 = normalised["atlantic_indian_transition"]
    r5 = normalised["atlantic_pacific_transition"]
    if not np.isclose(float(r5["north"]), float(r4["south"])) or not np.isclose(
        float(r5["north"]), float(r3["south"])
    ):
        raise ValueError("Pacific gateway bounds must satisfy r5.north = r4.south = r3.south")
    if not np.isclose(float(r4["north"]), float(r1["south"])) or not np.isclose(
        float(r4["north"]), float(r2["south"])
    ):
        raise ValueError("Indian gateway bounds must satisfy r4.north = r1.south = r2.south")
    if not (
        r1["west"] == r4["west"] == r5["west"]
        and r2["east"] == r4["east"]
        and r3["east"] == r5["east"]
    ):
        raise ValueError("region trace mappings are inconsistent with the five-region stitching")
    return normalised


def _insert_endpoints(
    traces: xr.Dataset, regions: Mapping[str, Mapping[str, object]]
) -> xr.Dataset:
    endpoints = {
        float(value)
        for definition in regions.values()
        for value in (definition["south"], definition["north"])
        if value is not None
    }
    latitude = np.unique(
        np.r_[np.asarray(traces.latitude, dtype=float), np.fromiter(endpoints, float)]
    )
    if np.array_equal(latitude, np.asarray(traces.latitude, dtype=float)):
        return traces
    original_latitude = np.asarray(traces.latitude, dtype=float)
    expanded = traces.reindex(latitude=latitude)
    inserted = ~np.isin(latitude, original_latitude)
    for variable in ("longitude", "longitude_raw"):
        values = np.asarray(expanded[variable], dtype=float).copy()
        original_values = np.asarray(traces[variable], dtype=float)
        for column, target in enumerate(latitude):
            if not inserted[column]:
                continue
            right = int(np.searchsorted(original_latitude, target))
            left = right - 1
            if left < 0 or right >= original_latitude.size:
                continue
            finite = np.isfinite(original_values[:, left]) & np.isfinite(
                original_values[:, right]
            )
            fraction = (target - original_latitude[left]) / (
                original_latitude[right] - original_latitude[left]
            )
            values[finite, column] = original_values[finite, left] + fraction * (
                original_values[finite, right] - original_values[finite, left]
            )
        expanded[variable] = (expanded[variable].dims, values)
    inserted_da = xr.DataArray(inserted, dims="latitude", coords={"latitude": latitude})
    repaired = expanded.repaired.fillna(False).astype(bool)
    expanded["repaired"] = repaired | (inserted_da & np.isfinite(expanded.longitude))
    expanded["valid"] = np.isfinite(expanded.longitude)
    expanded.attrs.update(traces.attrs)
    return expanded


def _assemble_dataset(
    traces: xr.Dataset,
    *,
    H: float,
    region_definitions: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, object],
) -> xr.Dataset:
    regions = _normalise_regions(region_definitions)
    trace_names = set(map(str, traces.trace.values))
    referenced = {
        str(definition[side])
        for definition in regions.values()
        for side in ("west", "east")
    }
    if len(referenced) != 6:
        raise ValueError("the fixed geometry requires six distinct boundary traces")
    if trace_names != referenced:
        raise ValueError(
            "trace data must contain exactly the traces referenced by regions"
        )

    traces = _insert_endpoints(traces, regions)

    dataset = traces.assign_coords(trace=list(map(str, traces.trace.values)))
    dataset = dataset.assign_coords(region=list(REGION_KEYS))
    dataset["region_west_trace"] = (
        "region",
        [str(regions[key]["west"]) for key in REGION_KEYS],
    )
    dataset["region_east_trace"] = (
        "region",
        [str(regions[key]["east"]) for key in REGION_KEYS],
    )
    dataset["region_south"] = (
        "region",
        [float(regions[key]["south"]) for key in REGION_KEYS],
    )
    dataset["region_north"] = (
        "region",
        [float(regions[key]["north"]) for key in REGION_KEYS],
    )
    dataset.attrs.update(
        {
            "H": float(H),
            "longitude_convention": "continuous degrees east, unwrapped by basin",
            "scientific_configuration": json.dumps(
                provenance, sort_keys=True, default=str
            ),
        }
    )
    return dataset


@dataclass(frozen=True, slots=True)
class MultiBasinGeometry:
    """Boundary geometry and active-layer depth for the global model.

    The wrapped dataset keeps six physical isobath traces and an explicit map
    from the five model regions to those traces.  It contains no forcing-grid
    masks or integration weights; the model derives those after seeing the
    wind grid.
    """

    _dataset: xr.Dataset

    def __post_init__(self) -> None:
        dataset = self._dataset
        required_dims = {"trace", "latitude", "region"}
        if not required_dims.issubset(dataset.dims):
            raise ValueError("geometry requires trace, latitude, and region dimensions")
        if tuple(map(str, dataset.region.values)) != REGION_KEYS:
            raise ValueError("geometry regions are not in fixed model order")
        for name in (
            "longitude",
            "valid",
            "region_west_trace",
            "region_east_trace",
            "region_south",
            "region_north",
        ):
            if name not in dataset:
                raise ValueError(f"geometry is missing {name!r}")
        latitude = np.asarray(dataset.latitude, dtype=float)
        if latitude.size < 2 or not np.all(np.diff(latitude) > 0):
            raise ValueError("geometry latitude must be strictly increasing")
        H = float(dataset.attrs.get("H", np.nan))
        if not np.isfinite(H) or H <= 0:
            raise ValueError("geometry H must be positive and finite")

        west = self._boundary_view("west")
        east = self._boundary_view("east")
        in_region = self.region_mask
        if bool(((~np.isfinite(west)) & in_region).any()):
            raise ValueError("western traces do not cover every region latitude")
        if bool(((~np.isfinite(east)) & in_region).any()):
            raise ValueError("eastern traces do not cover every region latitude")
        if bool(((east - west).where(in_region) <= 0).any()):
            raise ValueError("every eastern boundary must lie east of its west")

    @classmethod
    def from_isobath_dataset(
        cls,
        dataset: xr.Dataset,
        *,
        H: float | None = None,
    ) -> MultiBasinGeometry:
        """Load the canonical geometry product written by the extraction notebook.

        The product must contain the six variables listed in ``TRACE_NAMES`` and
        the five-region metadata variables.  Their names are the file format:
        no trace roles or region layouts are inferred at load time.
        """

        if "latitude" not in dataset.coords:
            raise ValueError("isobath dataset requires a latitude coordinate")
        source_H = float(dataset.attrs.get("isobath_depth_m", np.nan))
        if H is None:
            H = source_H
        if not np.isfinite(H) or H <= 0:
            raise ValueError("H must be supplied or present as isobath_depth_m")
        if np.isfinite(source_H) and not np.isclose(float(H), source_H):
            raise ValueError("supplied H conflicts with the isobath dataset depth")
        missing = sorted(set(TRACE_NAMES) - set(dataset.data_vars))
        if missing:
            raise ValueError(f"isobath dataset is missing canonical traces {missing}")

        region_definitions = _embedded_region_definitions(dataset)
        if region_definitions is None:
            raise ValueError(
                "isobath dataset does not contain authoritative region metadata"
            )

        arrays = []
        for trace in TRACE_NAMES:
            values = dataset[trace].astype(float)
            arrays.append(
                xr.Dataset(
                    {
                        "longitude": values,
                        "longitude_raw": values,
                        "repaired": xr.zeros_like(values, dtype=bool),
                    }
                ).assign_coords(trace=str(trace))
            )
        traces = xr.concat(arrays, dim="trace")
        traces["valid"] = np.isfinite(traces.longitude)
        provenance = {
            "source_attrs": dict(dataset.attrs),
        }
        return cls(
            _assemble_dataset(
                traces,
                H=float(H),
                region_definitions=region_definitions,
                provenance=provenance,
            )
        )

    @property
    def dataset(self) -> xr.Dataset:
        """The compact labelled geometry dataset."""

        return self._dataset

    @property
    def H(self) -> float:
        """Active-layer/isobath depth in metres."""

        return float(self._dataset.attrs["H"])

    @property
    def region_mask(self) -> xr.DataArray:
        """Boolean latitude support of each region on the geometry grid."""

        return (
            (self._dataset.latitude >= self._dataset.region_south)
            & (self._dataset.latitude <= self._dataset.region_north)
        ).transpose("region", "latitude")

    def _boundary_view(self, side: str) -> xr.DataArray:
        mapping = self._dataset[f"region_{side}_trace"]
        arrays = [
            self._dataset.longitude.sel(trace=str(mapping.sel(region=region).item()))
            for region in REGION_KEYS
        ]
        result = xr.concat(
            arrays, dim=xr.IndexVariable("region", list(REGION_KEYS))
        ).where(self.region_mask)
        if "trace" in result.coords:
            result = result.drop_vars("trace")
        return result

    @property
    def x_b(self) -> xr.DataArray:
        """Western-interior longitude by region and latitude."""

        return self._boundary_view("west").rename("x_b")

    @property
    def x_e(self) -> xr.DataArray:
        """Eastern-boundary longitude by region and latitude."""

        return self._boundary_view("east").rename("x_e")

    def boundaries_on(self, latitude: xr.DataArray | np.ndarray) -> xr.Dataset:
        """Interpolate region boundaries to a model latitude grid."""

        target = np.asarray(latitude, dtype=float)
        if target.ndim != 1 or not np.all(np.diff(target) > 0):
            raise ValueError("target latitude must be one-dimensional and increasing")
        variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
        for name, source in (("x_b", self.x_b), ("x_e", self.x_e)):
            values = np.full((len(REGION_KEYS), target.size), np.nan)
            for index, region in enumerate(REGION_KEYS):
                trace = source.sel(region=region).dropna("latitude")
                source_latitude = np.asarray(trace.latitude, dtype=float)
                inside = (target >= source_latitude[0]) & (
                    target <= source_latitude[-1]
                )
                values[index, inside] = np.interp(
                    target[inside], source_latitude, np.asarray(trace, dtype=float)
                )
            variables[name] = (("region", "latitude"), values)
        return xr.Dataset(
            variables,
            coords={"region": list(REGION_KEYS), "latitude": target},
        )
