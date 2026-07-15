from __future__ import annotations

import numpy as np
import pytest

from moc_adjustment_theory import BoundaryTrace


def test_boundary_trace_copies_and_freezes_arrays() -> None:
    latitude = np.array([-1.0, 0.0, 1.0])
    longitude = np.array([170.0, 180.0, 190.0])
    trace = BoundaryTrace("pacific_east", "east", latitude, longitude)

    latitude[0] = -99.0
    longitude[-1] = -99.0

    assert trace.latitude.tolist() == [-1.0, 0.0, 1.0]
    assert trace.longitude.tolist() == [170.0, 180.0, 190.0]
    assert not trace.latitude.flags.writeable
    assert not trace.longitude.flags.writeable


def test_longitude_interpolation_preserves_continuous_pacific_branch() -> None:
    trace = BoundaryTrace(
        "pacific_east",
        "east",
        latitude=[-1.0, 0.0, 1.0],
        longitude=[170.0, 180.0, 190.0],
    )

    assert trace.longitude_at(0.5) == pytest.approx(185.0)


def test_trace_domain_and_common_northern_limit() -> None:
    latitude = np.arange(-2.0, 4.0)
    west = BoundaryTrace(
        "indian_west",
        "west",
        latitude,
        [np.nan, 20.0, 20.0, 20.0, 20.0, np.nan],
    )
    east = BoundaryTrace(
        "indian_east",
        "east",
        latitude,
        [np.nan, 120.0, 120.0, 120.0, np.nan, np.nan],
    )

    assert west.southern_latitude == -1.0
    assert west.northern_latitude == 2.0
    assert west.covers(-1.0, 2.0)
    assert not west.covers(-2.0, 2.0)
    assert west.common_northern_limit(east) == 1.0
    assert west.common_northern_limit(east, cap=0.0) == 0.0
    assert west.common_northern_limit(east, cap=3.0) == 1.0


@pytest.mark.parametrize(
    ("latitude", "longitude", "match"),
    [
        ([0.0], [1.0], "at least two"),
        ([0.0, 0.0], [1.0, 2.0], "strictly increasing"),
        ([0.0, 1.0], [1.0], "same length"),
    ],
)
def test_trace_rejects_invalid_coordinates(
    latitude: list[float], longitude: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        BoundaryTrace("trace", "west", latitude, longitude)


def test_trace_rejects_internal_invalid_gap() -> None:
    with pytest.raises(ValueError, match="continuous"):
        BoundaryTrace(
            "trace",
            "west",
            latitude=[0.0, 1.0, 2.0],
            longitude=[1.0, np.nan, 3.0],
        )


def test_trace_rejects_repair_outside_valid_domain() -> None:
    with pytest.raises(ValueError, match="valid domain"):
        BoundaryTrace(
            "trace",
            "west",
            latitude=[0.0, 1.0, 2.0],
            longitude=[np.nan, 1.0, 2.0],
            repaired=[True, False, False],
        )


def test_trace_provenance_is_immutable() -> None:
    trace = BoundaryTrace(
        "trace",
        "west",
        latitude=[0.0, 1.0],
        longitude=[1.0, 2.0],
        provenance={"source": "synthetic"},
    )

    with pytest.raises(TypeError):
        trace.provenance["source"] = "changed"  # type: ignore[index]
