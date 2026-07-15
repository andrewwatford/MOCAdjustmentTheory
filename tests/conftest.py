from __future__ import annotations

import numpy as np
import pytest

from moc_adjustment_theory import Basin, BoundaryTrace


@pytest.fixture
def boundary_traces() -> dict[str, BoundaryTrace]:
    latitude = np.arange(-56.0, 56.0)

    def trace(
        key: str,
        side: str,
        longitude: float,
        south: float,
        north: float,
    ) -> BoundaryTrace:
        valid = (latitude >= south) & (latitude <= north)
        values = np.full(latitude.size, np.nan)
        values[valid] = longitude
        return BoundaryTrace(
            key=key,
            side=side,
            latitude=latitude,
            longitude=values,
            valid=valid,
            provenance={"fixture": "synthetic"},
        )

    return {
        "atlantic_west": trace(
            "atlantic_west", "west", -70.0, -56.0, 55.0
        ),
        "atlantic_east": trace(
            "atlantic_east", "east", 20.0, -35.0, 55.0
        ),
        "indian_west": trace("indian_west", "west", 20.0, -35.0, 50.0),
        "indian_east": trace("indian_east", "east", 120.0, -44.0, 50.0),
        "pacific_west": trace(
            "pacific_west", "west", 120.0, -44.0, 52.0
        ),
        "pacific_east": trace(
            "pacific_east", "east", 290.0, -56.0, 52.0
        ),
    }


@pytest.fixture
def non_itf_basins(boundary_traces: dict[str, BoundaryTrace]) -> tuple[Basin, ...]:
    t = boundary_traces
    return (
        Basin(
            key="atlantic_north",
            western_boundary=t["atlantic_west"],
            eastern_boundary=t["atlantic_east"],
            southern_boundary=-35.0,
            northern_boundary=55.0,
            inflows=("T_A",),
            outflows=("T_N",),
        ),
        Basin(
            key="indian_north",
            western_boundary=t["indian_west"],
            eastern_boundary=t["indian_east"],
            southern_boundary=-35.0,
            northern_boundary=50.0,
            inflows=("T_I",),
        ),
        Basin(
            key="pacific_north",
            western_boundary=t["pacific_west"],
            eastern_boundary=t["pacific_east"],
            southern_boundary=-44.0,
            northern_boundary=52.0,
            inflows=("T_P",),
        ),
        Basin(
            key="atlantic_indian_transition",
            western_boundary=t["atlantic_west"],
            eastern_boundary=t["indian_east"],
            southern_boundary=-44.0,
            northern_boundary=-35.0,
            inflows=("T_T",),
            outflows=("T_I", "T_A"),
        ),
        Basin(
            key="atlantic_pacific_transition",
            western_boundary=t["atlantic_west"],
            eastern_boundary=t["pacific_east"],
            southern_boundary=-56.0,
            northern_boundary=-44.0,
            inflows=("T_S",),
            outflows=("T_P", "T_T"),
        ),
    )

