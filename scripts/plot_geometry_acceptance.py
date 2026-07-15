"""Build the scientific review figure for a serialized GEBCO topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


TRACE_COLORS = {
    "atlantic_west": "#0072B2",
    "atlantic_east": "#E69F00",
    "indian_west": "#009E73",
    "indian_east": "#CC79A7",
    "pacific_west": "#56B4E9",
    "pacific_east": "#D55E00",
}

BASIN_COLORS = {
    "atlantic_north": "#0072B2",
    "indian_north": "#009E73",
    "pacific_north": "#D55E00",
    "atlantic_indian_transition": "#CC79A7",
    "atlantic_pacific_transition": "#56B4E9",
}


def _split_wrapped(longitude: np.ndarray) -> np.ndarray:
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    discontinuity = np.flatnonzero(np.abs(np.diff(wrapped)) > 180.0) + 1
    wrapped = wrapped.copy()
    wrapped[discontinuity] = np.nan
    return wrapped


def _load_bathymetry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path, engine="h5netcdf") as source:
        elevation = source.elevation.sel(lat=slice(-56.5, 66.5)).isel(
            lat=slice(None, None, 120), lon=slice(None, None, 120)
        ).load()
    depth = np.where(elevation.values < 0.0, -elevation.values, np.nan)
    return elevation.lon.values, elevation.lat.values, depth


def build_figure(
    geometry_path: Path, bathymetry_path: Path, output_path: Path
) -> None:
    geometry = xr.open_dataset(geometry_path, engine="h5netcdf").load()
    longitude, latitude, depth = _load_bathymetry(bathymetry_path)

    figure = plt.figure(figsize=(15.0, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.25, 1.25, 1.0))
    map_axis = figure.add_subplot(grid[:, :2])
    width_axis = figure.add_subplot(grid[0, 2])
    step_axis = figure.add_subplot(grid[1, 2], sharey=width_axis)

    image = map_axis.pcolormesh(
        longitude,
        latitude,
        np.clip(depth, 0.0, 5_000.0),
        shading="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=5_000.0,
        rasterized=True,
    )
    map_axis.set_facecolor("0.88")
    colorbar = figure.colorbar(image, ax=map_axis, orientation="horizontal", pad=0.055)
    colorbar.set_label("Ocean depth (m; clipped at 5000 m)")

    for key in map(str, geometry.trace.values):
        selected = geometry.sel(trace=key)
        color = TRACE_COLORS[key]
        raw = np.asarray(selected.longitude_raw.values, dtype=float)
        final = np.asarray(selected.longitude.values, dtype=float)
        valid = np.asarray(selected.valid.values, dtype=bool)
        repaired = np.asarray(selected.repaired.values, dtype=bool)
        y = np.asarray(geometry.latitude.values, dtype=float)

        map_axis.plot(
            _split_wrapped(raw),
            y,
            color=color,
            linewidth=0.55,
            alpha=0.28,
            linestyle=":",
        )
        map_axis.plot(
            _split_wrapped(np.where(valid, final, np.nan)),
            y,
            color=color,
            linewidth=1.35,
            label=key.replace("_", " "),
        )
        marked = repaired & valid
        map_axis.scatter(
            _split_wrapped(np.where(marked, final, np.nan)),
            y,
            s=3.0,
            color=color,
            edgecolors="none",
            zorder=4,
        )

    provenance = json.loads(str(geometry.trace_provenance.isel(trace=0).item()))
    closures = json.loads(provenance["closure_definitions"])
    for name, definition in closures.items():
        points = np.asarray(definition["points"], dtype=float)
        map_axis.plot(
            _split_wrapped(points[:, 0]),
            points[:, 1],
            color="black",
            linewidth=1.0,
            linestyle="--",
        )
        middle = points[len(points) // 2]
        map_axis.annotate(
            name.replace("_", " "),
            xy=(((middle[0] + 180.0) % 360.0) - 180.0, middle[1]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    for value, label in ((-56.0, "yS"), (-44.0, "yP"), (-35.0, "yI"), (55.0, "yN")):
        map_axis.axhline(value, color="0.2", linewidth=0.65, alpha=0.7)
        map_axis.text(-178.0, value + 0.8, label, fontsize=8, va="bottom")

    y = np.asarray(geometry.latitude.values, dtype=float)
    trace_names = list(map(str, geometry.trace.values))
    for basin in map(str, geometry.basin.values):
        west_name = str(geometry.basin_west_trace.sel(basin=basin).item())
        east_name = str(geometry.basin_east_trace.sel(basin=basin).item())
        west_index = trace_names.index(west_name)
        east_index = trace_names.index(east_name)
        west = geometry.longitude.isel(trace=west_index).values
        east = geometry.longitude.isel(trace=east_index).values
        valid = (
            geometry.valid.isel(trace=west_index).values
            & geometry.valid.isel(trace=east_index).values
            & (y >= float(geometry.basin_southern_latitude.sel(basin=basin)))
            & (y <= float(geometry.basin_northern_latitude.sel(basin=basin)))
        )
        width = (east - west) * 111.2 * np.cos(np.deg2rad(y)) / 1_000.0
        width_axis.plot(
            np.where(valid, width, np.nan),
            y,
            color=BASIN_COLORS[basin],
            linewidth=1.3,
            label=basin.replace("_", " "),
        )
        if basin == "indian_north":
            north = float(geometry.basin_northern_latitude.sel(basin=basin))
            west_at_north = float(np.interp(north, y[valid], west[valid]))
            east_at_north = float(np.interp(north, y[valid], east[valid]))
            map_axis.plot(
                [west_at_north, east_at_north],
                [north, north],
                color=BASIN_COLORS[basin],
                linewidth=1.35,
            )
            map_axis.annotate(
                "closed Indian north",
                xy=((west_at_north + east_at_north) / 2.0, north),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=7,
                ha="center",
            )

    for key in map(str, geometry.trace.values):
        selected = geometry.sel(trace=key)
        values = np.asarray(selected.longitude.values, dtype=float)
        valid = np.asarray(selected.valid.values, dtype=bool)
        indices = np.flatnonzero(valid)
        step = (
            np.abs(np.diff(values[indices]))
            * 111.2
            * np.cos(np.deg2rad(y[indices[1:]]))
        )
        step_axis.plot(
            step,
            y[indices[1:]],
            color=TRACE_COLORS[key],
            linewidth=0.9,
            label=key.replace("_", " "),
        )
    step_axis.axvline(120.0, color="black", linestyle="--", linewidth=0.8)

    map_axis.set(xlim=(-180.0, 180.0), ylim=(-56.0, 66.0), xlabel="Longitude", ylabel="Latitude")
    map_axis.set_title("a  GEBCO bathymetry, raw edges, final traces, and closures", loc="left")
    map_axis.legend(ncol=2, fontsize=7, loc="lower left")
    width_axis.set(xlabel="Basin width (10³ km)", ylabel="Latitude")
    width_axis.set_title("b  Model-region widths", loc="left")
    width_axis.legend(fontsize=6.6, loc="best")
    step_axis.set(xlabel="Adjacent-row step (km)", ylabel="Latitude", xlim=(0.0, 125.0))
    step_axis.set_title("c  Final-trace continuity", loc="left")
    step_axis.legend(fontsize=6.4, ncol=2, loc="best")
    for axis in (map_axis, width_axis, step_axis):
        axis.grid(alpha=0.18, linewidth=0.5)

    figure.suptitle("GEBCO 2026 1000 m multi-basin geometry acceptance")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("bathymetry", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    build_figure(arguments.geometry, arguments.bathymetry, arguments.output)


if __name__ == "__main__":
    main()
