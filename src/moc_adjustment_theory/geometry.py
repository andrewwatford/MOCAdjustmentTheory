"""Geometry value objects used by the adjustment models."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _readonly(values: ArrayLike, *, dtype: np.dtype, name: str) -> NDArray:
    array = np.asarray(values, dtype=dtype).copy()
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, eq=False)
class BoundaryTrace:
    """A continuous western or eastern isobath trace.

    Parameters
    ----------
    key
        Stable trace name, such as ``"atlantic_west"``.
    side
        Whether the trace is used as a western (``x_b``) or eastern
        (``x_e``) basin boundary.
    latitude, longitude
        One-dimensional coordinates in degrees north and continuous degrees
        east. Longitudes are not wrapped at the antimeridian.
    depth
        Isobath and active-layer depth in metres. They are the same quantity
        in this theory.
    raw_longitude
        Optional unsmoothed extraction result. Defaults to ``longitude``.
    valid
        Optional mask marking the trace's continuous valid latitude domain.
        Defaults to finite final longitudes.
    repaired
        Optional mask marking samples repaired during extraction.
    provenance
        Small string-valued source metadata carried into model outputs.

    Notes
    -----
    Arrays are copied and made read-only so that basins can safely share the
    same trace object.
    """

    key: str
    side: Literal["west", "east"]
    latitude: ArrayLike
    longitude: ArrayLike
    depth: float = 1000.0
    raw_longitude: ArrayLike | None = None
    valid: ArrayLike | None = None
    repaired: ArrayLike | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = self.key.strip()
        if not key:
            raise ValueError("trace key must not be empty")
        if self.side not in {"west", "east"}:
            raise ValueError("trace side must be 'west' or 'east'")

        latitude = _readonly(self.latitude, dtype=np.dtype(float), name="latitude")
        longitude = _readonly(
            self.longitude, dtype=np.dtype(float), name="longitude"
        )
        if latitude.size != longitude.size:
            raise ValueError("latitude and longitude must have the same length")
        if latitude.size < 2:
            raise ValueError("a boundary trace requires at least two samples")
        if not np.all(np.isfinite(latitude)):
            raise ValueError("latitude must be finite")
        if not np.all(np.diff(latitude) > 0):
            raise ValueError("latitude must be strictly increasing")

        if self.valid is None:
            valid = _readonly(
                np.isfinite(longitude), dtype=np.dtype(bool), name="valid"
            )
        else:
            valid = _readonly(self.valid, dtype=np.dtype(bool), name="valid")
        if valid.size != latitude.size:
            raise ValueError("valid must have the same length as latitude")
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < 2:
            raise ValueError("a boundary trace requires at least two valid samples")
        if not np.array_equal(
            valid_indices,
            np.arange(valid_indices[0], valid_indices[-1] + 1),
        ):
            raise ValueError("the valid latitude domain must be continuous")
        if not np.all(np.isfinite(longitude[valid])):
            raise ValueError("longitude must be finite in the valid domain")

        if self.raw_longitude is None:
            raw_longitude = _readonly(
                longitude, dtype=np.dtype(float), name="raw_longitude"
            )
        else:
            raw_longitude = _readonly(
                self.raw_longitude,
                dtype=np.dtype(float),
                name="raw_longitude",
            )
        if raw_longitude.size != latitude.size:
            raise ValueError("raw_longitude must have the same length as latitude")

        if self.repaired is None:
            repaired = _readonly(
                np.zeros(latitude.size, dtype=bool),
                dtype=np.dtype(bool),
                name="repaired",
            )
        else:
            repaired = _readonly(
                self.repaired, dtype=np.dtype(bool), name="repaired"
            )
        if repaired.size != latitude.size:
            raise ValueError("repaired must have the same length as latitude")
        if np.any(repaired & ~valid):
            raise ValueError("repaired samples must belong to the valid domain")

        depth = float(self.depth)
        if not np.isfinite(depth) or depth <= 0:
            raise ValueError("depth must be a positive finite value")

        provenance = MappingProxyType(
            {str(key): str(value) for key, value in self.provenance.items()}
        )

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "raw_longitude", raw_longitude)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "repaired", repaired)
        object.__setattr__(self, "provenance", provenance)

    @property
    def southern_latitude(self) -> float:
        """Southern end of the valid trace domain in degrees north."""

        return float(self.latitude[self.valid][0])

    @property
    def northern_latitude(self) -> float:
        """Northern end of the valid trace domain in degrees north."""

        return float(self.latitude[self.valid][-1])

    def covers(self, southern_latitude: float, northern_latitude: float) -> bool:
        """Return whether the continuous valid domain covers an interval."""

        return (
            southern_latitude <= northern_latitude
            and southern_latitude >= self.southern_latitude
            and northern_latitude <= self.northern_latitude
        )

    def has_latitude(self, latitude: float, *, atol: float = 1e-10) -> bool:
        """Return whether a valid sample exists at an exact model latitude."""

        matches = np.isclose(self.latitude, latitude, rtol=0.0, atol=atol)
        return bool(np.any(matches & self.valid))

    def longitude_at(self, latitude: float) -> float:
        """Interpolate continuous longitude at a latitude in the valid domain."""

        if not self.covers(latitude, latitude):
            raise ValueError(
                f"latitude {latitude} is outside the valid domain of {self.key!r}"
            )
        return float(
            np.interp(latitude, self.latitude[self.valid], self.longitude[self.valid])
        )

    def common_northern_limit(
        self, other: BoundaryTrace, *, cap: float
    ) -> float:
        """Return the northernmost common valid sample at or south of ``cap``."""

        common = np.intersect1d(
            self.latitude[self.valid], other.latitude[other.valid]
        )
        common = common[common <= cap]
        if common.size == 0:
            raise ValueError(
                f"{self.key!r} and {other.key!r} have no common latitude at "
                f"or south of {cap}"
            )
        return float(common[-1])
