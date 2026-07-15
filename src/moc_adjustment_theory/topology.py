"""Basin geometry and the fixed non-ITF multi-basin topology."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping

import numpy as np

from .geometry import BoundaryTrace


ATLANTIC_NORTH = "atlantic_north"
INDIAN_NORTH = "indian_north"
PACIFIC_NORTH = "pacific_north"
ATLANTIC_INDIAN_TRANSITION = "atlantic_indian_transition"
ATLANTIC_PACIFIC_TRANSITION = "atlantic_pacific_transition"

NON_ITF_BASIN_KEYS = (
    ATLANTIC_NORTH,
    INDIAN_NORTH,
    PACIFIC_NORTH,
    ATLANTIC_INDIAN_TRANSITION,
    ATLANTIC_PACIFIC_TRANSITION,
)

_EXPECTED_CONNECTIONS = MappingProxyType(
    {
        "T_S": (None, ATLANTIC_PACIFIC_TRANSITION),
        "T_P": (ATLANTIC_PACIFIC_TRANSITION, PACIFIC_NORTH),
        "T_T": (
            ATLANTIC_PACIFIC_TRANSITION,
            ATLANTIC_INDIAN_TRANSITION,
        ),
        "T_I": (ATLANTIC_INDIAN_TRANSITION, INDIAN_NORTH),
        "T_A": (ATLANTIC_INDIAN_TRANSITION, ATLANTIC_NORTH),
        "T_N": (ATLANTIC_NORTH, None),
    }
)


@dataclass(frozen=True, slots=True, eq=False)
class Basin:
    """One latitude-bounded dynamical sector.

    A basin combines geometry with named inflow and outflow membership. The
    connection names are joined to other basins by :class:`MultiBasinTopology`;
    no separate port abstraction is used.

    Parameters
    ----------
    key
        Stable basin name.
    western_boundary, eastern_boundary
        Shared isobath traces defining ``x_b`` and ``x_e``.
    southern_boundary, northern_boundary
        Exact model latitudes in degrees north.
    inflows, outflows
        Names of northward transport connections. Outflow order is the
        east-to-west child order at a branch.
    """

    key: str
    western_boundary: BoundaryTrace
    eastern_boundary: BoundaryTrace
    southern_boundary: float
    northern_boundary: float
    inflows: tuple[str, ...] = ()
    outflows: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        key = self.key.strip()
        if not key:
            raise ValueError("basin key must not be empty")
        if self.western_boundary.side != "west":
            raise ValueError("western_boundary must be a west trace")
        if self.eastern_boundary.side != "east":
            raise ValueError("eastern_boundary must be an east trace")
        if not np.isclose(
            self.western_boundary.depth, self.eastern_boundary.depth
        ):
            raise ValueError("western and eastern traces must use the same depth")

        south = float(self.southern_boundary)
        north = float(self.northern_boundary)
        if not np.isfinite(south) or not np.isfinite(north) or south >= north:
            raise ValueError("basin boundaries must be finite with south < north")
        if not self.western_boundary.covers(south, north):
            raise ValueError("western boundary does not cover the basin")
        if not self.eastern_boundary.covers(south, north):
            raise ValueError("eastern boundary does not cover the basin")
        for trace in (self.western_boundary, self.eastern_boundary):
            if not trace.has_latitude(south) or not trace.has_latitude(north):
                raise ValueError("basin endpoint latitudes must be sampled exactly")

        west_mask = (
            self.western_boundary.valid
            & (self.western_boundary.latitude >= south)
            & (self.western_boundary.latitude <= north)
        )
        east_mask = (
            self.eastern_boundary.valid
            & (self.eastern_boundary.latitude >= south)
            & (self.eastern_boundary.latitude <= north)
        )
        latitudes = self.western_boundary.latitude[west_mask]
        if not np.array_equal(
            latitudes, self.eastern_boundary.latitude[east_mask]
        ):
            raise ValueError("basin boundaries must share one sampled latitude grid")
        widths = (
            self.eastern_boundary.longitude[east_mask]
            - self.western_boundary.longitude[west_mask]
        )
        if np.any(widths <= 0):
            raise ValueError("eastern boundary must lie east of western boundary")

        inflows = tuple(str(name).strip() for name in self.inflows)
        outflows = tuple(str(name).strip() for name in self.outflows)
        if any(not name for name in (*inflows, *outflows)):
            raise ValueError("connection names must not be empty")
        if len(set(inflows)) != len(inflows) or len(set(outflows)) != len(outflows):
            raise ValueError("connection names must not be duplicated")
        if set(inflows) & set(outflows):
            raise ValueError("a connection cannot be both an inflow and an outflow")

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "southern_boundary", south)
        object.__setattr__(self, "northern_boundary", north)
        object.__setattr__(self, "inflows", inflows)
        object.__setattr__(self, "outflows", outflows)

    @property
    def depth(self) -> float:
        """Shared isobath and active-layer depth in metres."""

        return self.western_boundary.depth

    @property
    def latitude(self) -> np.ndarray:
        """Sampled latitude coordinates within the basin."""

        mask = (
            self.western_boundary.valid
            & (self.western_boundary.latitude >= self.southern_boundary)
            & (self.western_boundary.latitude <= self.northern_boundary)
        )
        return self.western_boundary.latitude[mask]

    def x_b(self, latitude: float) -> float:
        """Return the western isobath longitude at ``latitude``."""

        if not self.southern_boundary <= latitude <= self.northern_boundary:
            raise ValueError("latitude is outside the basin")
        return self.western_boundary.longitude_at(latitude)

    def x_e(self, latitude: float) -> float:
        """Return the eastern isobath longitude at ``latitude``."""

        if not self.southern_boundary <= latitude <= self.northern_boundary:
            raise ValueError("latitude is outside the basin")
        return self.eastern_boundary.longitude_at(latitude)


class MultiBasinTopology:
    """The fixed five-basin topology for the non-ITF global model.

    Constructing this object stitches five separately initialized
    :class:`Basin` objects together through their named transports. It validates
    the graph, east-to-west child order, gateway latitudes, and shared boundary
    identity specified by the non-ITF theory.
    """

    def __init__(self, basins: Iterable[Basin]) -> None:
        basin_list = tuple(basins)
        if len(basin_list) != len(NON_ITF_BASIN_KEYS):
            raise ValueError("the non-ITF topology requires exactly five basins")
        by_key = {basin.key: basin for basin in basin_list}
        if len(by_key) != len(basin_list):
            raise ValueError("basin keys must be unique")
        if set(by_key) != set(NON_ITF_BASIN_KEYS):
            raise ValueError(
                "basin keys must match the fixed non-ITF five-basin model"
            )

        connections = self._assemble_connections(basin_list)
        if connections != dict(_EXPECTED_CONNECTIONS):
            raise ValueError("basin inflows and outflows do not match the non-ITF graph")

        children = {
            basin.key: tuple(
                connections[name][1]
                for name in basin.outflows
                if connections[name][1] is not None
            )
            for basin in basin_list
        }
        expected_children = {
            ATLANTIC_NORTH: (),
            INDIAN_NORTH: (),
            PACIFIC_NORTH: (),
            ATLANTIC_INDIAN_TRANSITION: (INDIAN_NORTH, ATLANTIC_NORTH),
            ATLANTIC_PACIFIC_TRANSITION: (
                PACIFIC_NORTH,
                ATLANTIC_INDIAN_TRANSITION,
            ),
        }
        if children != expected_children:
            raise ValueError("basin outflows do not preserve east-to-west child order")

        self._validate_geometry(by_key)
        parents = {
            key: tuple(
                source
                for source, target in connections.values()
                if target == key and source is not None
            )
            for key in NON_ITF_BASIN_KEYS
        }
        depth = basin_list[0].depth
        if not all(np.isclose(basin.depth, depth) for basin in basin_list):
            raise ValueError("all basins must use one isobath/active-layer depth")

        self._basins = MappingProxyType(
            {key: by_key[key] for key in NON_ITF_BASIN_KEYS}
        )
        self._connections = MappingProxyType(connections)
        self._children = MappingProxyType(children)
        self._parents = MappingProxyType(parents)
        self._depth = float(depth)

    @staticmethod
    def _assemble_connections(
        basins: tuple[Basin, ...]
    ) -> dict[str, tuple[str | None, str | None]]:
        sources: dict[str, str] = {}
        targets: dict[str, str] = {}
        for basin in basins:
            for name in basin.outflows:
                if name in sources:
                    raise ValueError(f"connection {name!r} has multiple sources")
                sources[name] = basin.key
            for name in basin.inflows:
                if name in targets:
                    raise ValueError(f"connection {name!r} has multiple targets")
                targets[name] = basin.key
        names = (*sources, *(name for name in targets if name not in sources))
        return {
            name: (sources.get(name), targets.get(name))
            for name in names
        }

    @staticmethod
    def _validate_geometry(basins: Mapping[str, Basin]) -> None:
        r1 = basins[ATLANTIC_NORTH]
        r2 = basins[INDIAN_NORTH]
        r3 = basins[PACIFIC_NORTH]
        r4 = basins[ATLANTIC_INDIAN_TRANSITION]
        r5 = basins[ATLANTIC_PACIFIC_TRANSITION]

        if not (
            r5.southern_boundary
            < r5.northern_boundary
            < r4.northern_boundary
            < r1.northern_boundary
        ):
            raise ValueError("junctions must satisfy y_S < y_P < y_I < y_N")
        if not (
            r5.northern_boundary
            == r4.southern_boundary
            == r3.southern_boundary
        ):
            raise ValueError("Pacific gateway latitude is inconsistent")
        if not (
            r4.northern_boundary
            == r2.southern_boundary
            == r1.southern_boundary
        ):
            raise ValueError("Indian gateway latitude is inconsistent")
        indian_limit = r2.western_boundary.common_northern_limit(
            r2.eastern_boundary, cap=r1.northern_boundary
        )
        if not np.isclose(r2.northern_boundary, indian_limit):
            raise ValueError(
                "Indian northern closure must be the northernmost common "
                "boundary sample at or south of y_N"
            )
        pacific_limit = r3.western_boundary.common_northern_limit(
            r3.eastern_boundary, cap=r1.northern_boundary
        )
        if not np.isclose(r3.northern_boundary, pacific_limit):
            raise ValueError(
                "Pacific northern closure must be the northernmost common "
                "boundary sample at or south of y_N"
            )

        expected_traces = {
            ATLANTIC_NORTH: ("atlantic_west", "atlantic_east"),
            INDIAN_NORTH: ("indian_west", "indian_east"),
            PACIFIC_NORTH: ("pacific_west", "pacific_east"),
            ATLANTIC_INDIAN_TRANSITION: (
                "atlantic_west",
                "indian_east",
            ),
            ATLANTIC_PACIFIC_TRANSITION: (
                "atlantic_west",
                "pacific_east",
            ),
        }
        for key, (west, east) in expected_traces.items():
            basin = basins[key]
            if (
                basin.western_boundary.key != west
                or basin.eastern_boundary.key != east
            ):
                raise ValueError(f"{key!r} uses the wrong boundary traces")

        if not (
            r1.western_boundary is r4.western_boundary
            and r1.western_boundary is r5.western_boundary
        ):
            raise ValueError("Atlantic western boundary must be shared by identity")
        if r2.eastern_boundary is not r4.eastern_boundary:
            raise ValueError("Indian eastern boundary must be shared by identity")
        if r3.eastern_boundary is not r5.eastern_boundary:
            raise ValueError("Pacific eastern boundary must be shared by identity")

    @property
    def basin_keys(self) -> tuple[str, ...]:
        """Basin keys in the region order used by the theory."""

        return NON_ITF_BASIN_KEYS

    @property
    def connections(self) -> Mapping[str, tuple[str | None, str | None]]:
        """Transport name to ``(source, target)`` basin endpoints."""

        return self._connections

    @property
    def depth(self) -> float:
        """Shared isobath and active-layer depth in metres."""

        return self._depth

    @property
    def atlantic_path(self) -> tuple[str, ...]:
        """South-to-north basin path defining the Atlantic view."""

        return (
            ATLANTIC_PACIFIC_TRANSITION,
            ATLANTIC_INDIAN_TRANSITION,
            ATLANTIC_NORTH,
        )

    @property
    def eastern_boundary_groups(self) -> Mapping[str, tuple[str, ...]]:
        """Basins sharing each independent eastern-boundary thickness."""

        groups: dict[str, list[str]] = {}
        for key, basin in self._basins.items():
            groups.setdefault(basin.eastern_boundary.key, []).append(key)
        return MappingProxyType(
            {trace: tuple(keys) for trace, keys in groups.items()}
        )

    def basin(self, key: str) -> Basin:
        """Return a basin by key."""

        try:
            return self._basins[key]
        except KeyError as error:
            raise KeyError(f"unknown basin {key!r}") from error

    def children(self, key: str) -> tuple[str, ...]:
        """Return children in east-to-west order."""

        self.basin(key)
        return self._children[key]

    def parents(self, key: str) -> tuple[str, ...]:
        """Return upstream parent basins."""

        self.basin(key)
        return self._parents[key]

    def __iter__(self) -> Iterator[Basin]:
        return iter(self._basins.values())

    def __len__(self) -> int:
        return len(self._basins)
