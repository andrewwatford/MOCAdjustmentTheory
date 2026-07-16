"""Labelled output returned by the global adjustment model."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True, slots=True)
class GlobalAdjustmentOutput:
    """Complete time- and frequency-domain model result."""

    _dataset: xr.Dataset
    _spectral: xr.Dataset

    @property
    def dataset(self) -> xr.Dataset:
        """Time-domain fields on the original forcing record."""

        return self._dataset

    @property
    def spectral(self) -> xr.Dataset:
        """Frequency-domain solution and compact model diagnostics."""

        return self._spectral

    @property
    def h_e(self) -> xr.DataArray:
        return self._dataset.h_e

    @property
    def h_b(self) -> xr.DataArray:
        return self._dataset.h_b

    @property
    def h_w(self) -> xr.DataArray:
        return self._dataset.h_w

    @property
    def transport(self) -> xr.DataArray:
        return self._dataset.transport

    @property
    def transport_ekman(self) -> xr.DataArray:
        return self._dataset.transport_ekman

    @property
    def transport_geostrophic(self) -> xr.DataArray:
        return self._dataset.transport_geostrophic

    def transport_at(self, region: str, latitude: float) -> xr.DataArray:
        """Total transport at any supported latitude in one model region."""

        selected = self._dataset.transport.sel(region=region).dropna(
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
