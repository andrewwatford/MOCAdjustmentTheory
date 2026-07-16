"""Labelled output returned by the global adjustment model."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True, slots=True)
class GlobalAdjustmentOutput:
    """Complete time- and frequency-domain model result."""

    dataset: xr.Dataset
    spectral: xr.Dataset

    def transport_at(self, region: str, latitude: float) -> xr.DataArray:
        """Total transport at any supported latitude in one model region."""

        selected = self.dataset.transport.sel(region=region).dropna(
            "latitude", how="all"
        )
        south = float(selected.latitude[0])
        north = float(selected.latitude[-1])
        if not south <= latitude <= north:
            raise ValueError(
                f"latitude {latitude:g} is outside {region!r} support "
                f"[{south:g}, {north:g}]"
            )
        return selected.interp(latitude=float(latitude))
