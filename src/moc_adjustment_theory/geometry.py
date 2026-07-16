"""Canonical six-trace geometry for the fixed five-region global model.

The isobath-extraction notebook writes the geometry product consumed here.  It
is deliberately an authoritative data product: this module validates its
schema but does not infer trace roles, repair gaps, or apply forcing-grid
masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr


REGION_KEYS = (
    "atlantic_north",
    "indian_north",
    "pacific_north",
    "atlantic_indian_transition",
    "atlantic_pacific_transition",
)
"""Fixed regional order used by the global three-thickness solve."""

TRACE_NAMES = (
    "atlantic_west",
    "atlantic_east",
    "indian_west",
    "indian_east",
    "pacific_west",
    "pacific_east",
)
"""The six named isobath traces required in a canonical geometry product."""

_REGION_METADATA_VARIABLES = (
    "region_west_trace",
    "region_east_trace",
    "region_south",
    "region_north",
)


def _embedded_region_definitions(
    dataset: xr.Dataset,
) -> dict[str, dict[str, object]] | None:
    """Return the explicit region-to-trace map stored in a geometry product.

    A product either supplies all of the region metadata or none of it.  The
    latter is rejected by :meth:`MultiBasinGeometry.from_isobath_dataset`,
    because this package intentionally has no fallback role inference.
    """

    present = [name in dataset for name in _REGION_METADATA_VARIABLES]
    if not any(present):
        return None
    if not all(present) or "region" not in dataset.coords:
        raise ValueError(
            "isobath dataset contains incomplete region metadata; expected "
            "region coordinate plus west/east trace and south/north variables"
        )
    if tuple(map(str, dataset.region.values)) != REGION_KEYS:
        raise ValueError("isobath dataset regions are not in fixed model order")
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
    """Validate region bounds and the prescribed five-region stitching.

    The global theory has a fixed graph, not a generic graph compiler.  This
    check makes the two ordered gateways and their shared traces explicit at
    load time.
    """

    if set(definitions) != set(REGION_KEYS):
        missing = sorted(set(REGION_KEYS) - set(definitions))
        extra = sorted(set(definitions) - set(REGION_KEYS))
        raise ValueError(
            "region metadata must contain the fixed five model regions; "
            f"missing={missing}, extra={extra}"
        )
    normalised: dict[str, dict[str, object]] = {}
    for key in REGION_KEYS:
        definition = definitions[key]
        west = str(definition.get("west", "")).strip()
        east = str(definition.get("east", "")).strip()
        if west not in TRACE_NAMES or east not in TRACE_NAMES:
            raise ValueError(f"region {key!r} references an unknown canonical trace")
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
        raise ValueError(
            "Pacific gateway bounds must satisfy r5.north = r4.south = r3.south"
        )
    if not np.isclose(float(r4["north"]), float(r1["south"])) or not np.isclose(
        float(r4["north"]), float(r2["south"])
    ):
        raise ValueError(
            "Indian gateway bounds must satisfy r4.north = r1.south = r2.south"
        )
    if not (
        r1["west"] == r4["west"] == r5["west"]
        and r2["east"] == r4["east"]
        and r3["east"] == r5["east"]
    ):
        raise ValueError("region trace mappings are inconsistent with five-region stitching")
    return normalised


@dataclass(frozen=True, slots=True)
class MultiBasinGeometry:
    """Validated canonical geometry and active-layer depth.

    Parameters
    ----------
    dataset
        The canonical xarray product: six one-dimensional longitude traces on
        a common ``latitude`` coordinate and explicit five-region metadata.
        It is retained unchanged, so its provenance remains available to users.
    H
        Active-layer depth in metres.  Usually this is the product's
        ``isobath_depth_m`` attribute, passed through by
        :meth:`from_isobath_dataset`.

    Notes
    -----
    This object owns geometry only.  It does not contain wind masks,
    integration weights, forcing data, or a topology inferred from labels.
    """

    dataset: xr.Dataset
    H: float

    def __post_init__(self) -> None:
        """Validate the canonical product without modifying its traces."""

        if not isinstance(self.dataset, xr.Dataset):
            raise TypeError("geometry dataset must be an xarray.Dataset")
        if "latitude" not in self.dataset.coords:
            raise ValueError("isobath dataset requires a latitude coordinate")
        if not np.isfinite(self.H) or self.H <= 0:
            raise ValueError("geometry H must be positive and finite")
        latitude = np.asarray(self.dataset.latitude, dtype=float)
        if latitude.size < 2 or not np.all(np.isfinite(latitude)):
            raise ValueError("geometry latitude requires at least two finite values")
        if not np.all(np.diff(latitude) > 0):
            raise ValueError("geometry latitude must be strictly increasing")

        missing = sorted(set(TRACE_NAMES) - set(self.dataset.data_vars))
        if missing:
            raise ValueError(f"isobath dataset is missing canonical traces {missing}")
        for name in TRACE_NAMES:
            if self.dataset[name].dims != ("latitude",):
                raise ValueError(f"canonical trace {name!r} must use only latitude")

        definitions = _embedded_region_definitions(self.dataset)
        if definitions is None:
            raise ValueError(
                "isobath dataset does not contain authoritative region metadata"
            )
        regions = _normalise_regions(definitions)
        for region, definition in regions.items():
            region_rows = (latitude >= float(definition["south"])) & (
                latitude <= float(definition["north"])
            )
            for side in ("west", "east"):
                trace_name = str(definition[side])
                values = np.asarray(self.dataset[trace_name], dtype=float)
                if not np.all(np.isfinite(values[region_rows])):
                    raise ValueError(
                        f"{side}ern traces do not cover every {region!r} latitude"
                    )
            west = np.asarray(self.dataset[str(definition["west"])], dtype=float)
            east = np.asarray(self.dataset[str(definition["east"])], dtype=float)
            if np.any(east[region_rows] <= west[region_rows]):
                raise ValueError("every eastern boundary must lie east of its west")

    @classmethod
    def from_isobath_dataset(
        cls,
        dataset: xr.Dataset,
        *,
        H: float | None = None,
    ) -> MultiBasinGeometry:
        """Load a geometry product written in the extraction-notebook schema.

        The product names all six traces and includes the region metadata that
        defines their stitching.  ``H`` defaults to ``isobath_depth_m`` and
        can only be supplied when it agrees with that source attribute.
        """

        if not isinstance(dataset, xr.Dataset):
            raise TypeError("isobath dataset must be an xarray.Dataset")
        source_H = float(dataset.attrs.get("isobath_depth_m", np.nan))
        if H is None:
            H = source_H
        if not np.isfinite(H) or H <= 0:
            raise ValueError("H must be supplied or present as isobath_depth_m")
        if np.isfinite(source_H) and not np.isclose(float(H), source_H):
            raise ValueError("supplied H conflicts with the isobath dataset depth")
        return cls(dataset=dataset, H=float(H))

    def boundaries_on(self, latitude: xr.DataArray | np.ndarray) -> xr.Dataset:
        """Interpolate the western and eastern trace of each region to rows.

        Parameters
        ----------
        latitude
            Strictly increasing target latitudes in degrees north.

        Returns
        -------
        xarray.Dataset
            ``x_b(region, latitude)`` and ``x_e(region, latitude)``.  Values
            outside a region's declared latitude range are ``NaN``.

        Raises
        ------
        ValueError
            If target latitudes are not a finite, strictly increasing vector.
        """

        target = np.asarray(latitude, dtype=float)
        if target.ndim != 1 or target.size == 0 or not np.all(np.isfinite(target)):
            raise ValueError("target latitude must be a finite one-dimensional vector")
        if target.size > 1 and not np.all(np.diff(target) > 0):
            raise ValueError("target latitude must be strictly increasing")
        definitions = _embedded_region_definitions(self.dataset)
        assert definitions is not None  # checked in __post_init__
        regions = _normalise_regions(definitions)
        variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
        for output_name, side in (("x_b", "west"), ("x_e", "east")):
            values = np.full((len(REGION_KEYS), target.size), np.nan)
            for index, region in enumerate(REGION_KEYS):
                definition = regions[region]
                trace = self.dataset[str(definition[side])].dropna("latitude")
                source_latitude = np.asarray(trace.latitude, dtype=float)
                region_target = (
                    (target >= float(definition["south"]))
                    & (target <= float(definition["north"]))
                    & (target >= source_latitude[0])
                    & (target <= source_latitude[-1])
                )
                values[index, region_target] = np.interp(
                    target[region_target],
                    source_latitude,
                    np.asarray(trace, dtype=float),
                )
            variables[output_name] = (("region", "latitude"), values)
        return xr.Dataset(
            variables,
            coords={"region": list(REGION_KEYS), "latitude": target},
        )
