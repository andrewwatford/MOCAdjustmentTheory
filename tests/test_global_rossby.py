"""Tests for the global Rossby-wave adjustment model."""

import numpy as np
import pytest
import xarray as xr
from dask.array import Array
from dask.callbacks import Callback

import moc_adjustment_theory.global_rossby as global_rossby
from moc_adjustment_theory import (
    GlobalRossbyModel,
    forward_transform,
    inverse_transform,
)


def geometry():
    latitude = np.arange(-60.0, 71.0, 10.0)

    def boundary(value, lower, upper):
        return xr.DataArray(
            np.where((latitude >= lower) & (latitude <= upper), value, np.nan),
            dims="latitude",
        )

    return xr.Dataset(
        {
            "x_wA": boundary(-70.0, -50.0, 70.0),
            "x_eA": boundary(10.0, -30.0, 70.0),
            "x_wI": boundary(30.0, -30.0, 30.0),
            "x_eI": boundary(100.0, -40.0, 30.0),
            "x_wP": boundary(120.0, -40.0, 60.0),
            "x_eP": boundary(280.0, -50.0, 60.0),
        },
        coords={"latitude": latitude},
        attrs={"isobath_depth_m": 1000.0},
    )


def forcing(t_n=(0.0, 1.0e6), wind=False, t_n_latitude=60.0):
    omega = np.array([0.0, 1.0e-6])
    latitude = np.arange(-50.0, 71.0, 10.0)
    longitude = np.arange(-80.0, 291.0, 10.0)
    zeros = np.zeros((omega.size, latitude.size, longitude.size), dtype=complex)
    dataset = xr.Dataset(
        {
            "M_Ek_x": (("omega", "latitude", "longitude"), zeros),
            "M_Ek_y": (("omega", "latitude", "longitude"), zeros),
            "T_N": ("omega", np.asarray(t_n, dtype=complex)),
        },
        coords={
            "omega": omega,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    if wind:
        profile = 1.0 + (latitude[:, None] + 50.0) / 120.0
        dataset["M_Ek_y"][1] = (2.0 + 0.5j) * profile
    dataset.T_N.attrs["latitude_degrees_north"] = t_n_latitude
    return dataset


def temporal_forcing():
    """Return zero-mean monthly forcing on the synthetic global grid."""
    time = np.array(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
            "2000-04-01",
        ],
        dtype="datetime64[D]",
    )
    latitude = np.arange(-50.0, 71.0, 10.0)
    longitude = np.arange(-80.0, 291.0, 10.0)
    zeros = np.zeros((time.size, latitude.size, longitude.size))
    dataset = xr.Dataset(
        {
            "M_Ek_x": (("time", "latitude", "longitude"), zeros),
            "M_Ek_y": (("time", "latitude", "longitude"), zeros),
            "T_N": ("time", [0.0, 1.0e6, 0.0, -1.0e6]),
        },
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    )
    dataset.T_N.attrs["latitude_degrees_north"] = 60.0
    return dataset


def nan_masked_temporal_forcing():
    """Return temporal forcing with NaNs outside the modelled ocean union."""
    dataset = temporal_forcing()
    model = GlobalRossbyModel(geometry(), 0.02)
    latitude, longitude = model._forcing_coordinates(dataset)
    regional_geometry = model._geometry(latitude, 60.0)
    quadrature = model._zonal_quadrature(
        longitude, latitude, regional_geometry
    )
    ocean_mask = model._active_ocean_mask(
        latitude.size, longitude.size, quadrature
    )
    for name in ("M_Ek_x", "M_Ek_y"):
        values = dataset[name].values.copy()
        values[:, ~ocean_mask] = np.nan
        dataset[name] = (dataset[name].dims, values)
    return dataset, ocean_mask


def test_unforced_wind_solution_closes_regional_budgets():
    result = GlobalRossbyModel(geometry(), 0.02)._solve_frequency(forcing())

    assert set(result.data_vars) == {"h_e", "h_b", "h_w", "h", "T", "T_g", "T_Ek"}
    assert result.h_e.dims == ("omega", "region")
    assert result.T.dims == ("omega", "region", "latitude")
    assert np.all(result.isel(omega=0).fillna(0) == 0)
    np.testing.assert_allclose(result.T_Ek.fillna(0), 0)
    np.testing.assert_allclose(result.T_g, result.T, equal_nan=True)

    # Closed Indian and Pacific northern boundaries and the supplied Atlantic one.
    np.testing.assert_allclose(result.T.sel(region="north_atlantic").isel(omega=1).dropna("latitude")[-1], 1.0e6)
    np.testing.assert_allclose(result.T.sel(region="north_indian").isel(omega=1).dropna("latitude")[-1], 0.0, atol=1e-8)
    np.testing.assert_allclose(result.T.sel(region="north_pacific").isel(omega=1).dropna("latitude")[-1], 0.0, atol=1e-8)

    # The diagnostic boundary thickness obeys free Rossby-wave propagation.
    h_b = result.h_b.sel(region="north_atlantic").isel(omega=1).dropna("latitude")
    assert np.all(np.isfinite(h_b))
    np.testing.assert_allclose(np.abs(h_b), abs(result.h_e.sel(region="north_atlantic").isel(omega=1)))

    h = result.h.sel(region="north_atlantic").isel(omega=1)
    assert h.dims == ("latitude", "longitude")
    for latitude in h.dropna("latitude", how="all").latitude.values:
        row = h.sel(latitude=latitude).dropna("longitude")
        assert row.size
        np.testing.assert_allclose(
            row[-1], result.h_e.sel(region="north_atlantic").isel(omega=1)
            * np.exp(
                1j * forcing().omega[1].item()
                * global_rossby.EARTH_RADIUS_M
                * np.cos(np.deg2rad(latitude))
                * np.deg2rad(
                    row.longitude[-1].item()
                    - geometry().x_eA.sel(latitude=latitude).item()
                )
                / global_rossby.rossby_speed(
                    np.asarray([latitude]), 0.02, 1000.0
                )[0]
            ),
        )


def test_transition_topology_conserves_transport():
    result = GlobalRossbyModel(geometry(), 0.02)._solve_frequency(forcing())
    t = result.T.isel(omega=1)

    north_4 = t.sel(region="atlantic_indian").dropna("latitude")[-1]
    south_1 = t.sel(region="north_atlantic").dropna("latitude")[0]
    south_2 = t.sel(region="north_indian").dropna("latitude")[0]
    np.testing.assert_allclose(north_4, south_1 + south_2)

    north_5 = t.sel(region="atlantic_pacific").dropna("latitude")[-1]
    south_3 = t.sel(region="north_pacific").dropna("latitude")[0]
    south_4 = t.sel(region="atlantic_indian").dropna("latitude")[0]
    np.testing.assert_allclose(north_5, south_3 + south_4)


def test_northern_forcing_latitude_truncates_only_the_atlantic():
    result = GlobalRossbyModel(geometry(), 0.02)._solve_frequency(
        forcing(t_n_latitude=55.0)
    )

    atlantic = result.T.sel(region="north_atlantic").dropna("latitude")
    pacific = result.T.sel(region="north_pacific").dropna("latitude")
    assert float(atlantic.latitude[-1]) == 50.0
    assert float(pacific.latitude[-1]) == 60.0
    np.testing.assert_allclose(atlantic.isel(omega=1, latitude=-1), 1.0e6)


def test_northern_forcing_latitude_must_be_valid_and_covered():
    model = GlobalRossbyModel(geometry(), 0.02)

    with pytest.raises(ValueError, match="south of the equator"):
        model._solve_frequency(forcing(t_n_latitude=-1.0))
    with pytest.raises(ValueError, match="forcing latitude does not reach"):
        model._solve_frequency(forcing(t_n_latitude=80.0))

    missing = forcing()
    missing.T_N.attrs.clear()
    with pytest.raises(ValueError, match="finite numeric"):
        model._solve_frequency(missing)

    short_geometry = geometry().where(geometry().latitude <= 60.0)
    with pytest.raises(ValueError, match="Atlantic geometry does not reach"):
        GlobalRossbyModel(short_geometry, 0.02)._solve_frequency(
            forcing(t_n_latitude=70.0)
        )


def test_nonzero_ekman_forcing_produces_consistent_diagnostics():
    result = GlobalRossbyModel(geometry(), 0.02)._solve_frequency(
        forcing(t_n=(0.0, 0.0), wind=True)
    )

    assert np.any(np.abs(result.T_Ek.isel(omega=1).fillna(0)) > 0)
    np.testing.assert_allclose(result.T, result.T_g + result.T_Ek, equal_nan=True)
    f = 2.0 * 7.292115e-5 * np.sin(np.deg2rad(result.latitude))
    expected = result.h_e - f * result.T_g / (0.02 * 1000.0)
    np.testing.assert_allclose(result.h_w, expected, equal_nan=True)
    for region in result.region.values:
        h_b = result.h_b.sel(region=region).isel(omega=1).dropna("latitude")
        assert h_b.size and np.all(np.isfinite(h_b))

        field = result.h.sel(region=region).isel(omega=1)
        for latitude in field.dropna("latitude", how="all").latitude.values:
            row = field.sel(latitude=latitude).dropna("longitude")
            np.testing.assert_allclose(
                row[0], h_b.sel(latitude=latitude), rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                row[-1],
                result.h_e.sel(region=region).isel(omega=1),
                rtol=1e-12,
                atol=1e-12,
            )


def test_height_field_uses_grid_points_between_off_grid_boundaries():
    shifted = geometry().copy()
    shifted["x_wA"] = shifted.x_wA + 5.0
    shifted["x_eA"] = shifted.x_eA + 5.0

    result = GlobalRossbyModel(shifted, 0.02)._solve_frequency(forcing())
    field = result.h.sel(region="north_atlantic").isel(omega=1)

    for latitude in field.dropna("latitude", how="all").latitude.values:
        row = field.sel(latitude=latitude).dropna("longitude")
        assert row.longitude[0] == -60.0
        assert row.longitude[-1] == 10.0
        speed = global_rossby.rossby_speed(
            np.asarray([latitude]), 0.02, 1000.0
        )[0]
        expected = result.h_e.sel(region="north_atlantic").isel(
            omega=1
        ) * np.exp(
            1j
            * forcing().omega[1].item()
            * global_rossby.EARTH_RADIUS_M
            * np.cos(np.deg2rad(latitude))
            * np.deg2rad(row.longitude - 15.0)
            / speed
        )
        np.testing.assert_allclose(row, expected)


def test_boundary_solution_satisfies_three_basin_system():
    omega = np.array([1.0e-6])
    f_term = np.array([[1 + 1j, 2 - 1j, 3 + 2j, 4 - 2j, 5 + 0.5j]])
    r = np.array([[2 + 3j, 3 + 4j, 4 + 5j, 5 + 6j, 6 + 7j]])
    t_n, t_i, t_p, t_s = (np.array([value + 0j]) for value in (7, 8, 9, 10))
    k_i, k_p = -20.0, -30.0
    h = GlobalRossbyModel._solve_boundaries(
        omega, f_term, r, t_n, t_i, t_p, t_s, k_i, k_p
    )[0]
    matrix = np.array(
        [
            [r[0, 0] + k_i, r[0, 3] + k_p - k_i, r[0, 4] - k_p],
            [-k_i, r[0, 1] + k_i, 0],
            [0, -k_p, r[0, 2] + k_p],
        ]
    )
    rhs = np.array(
        [
            f_term[0, [0, 3, 4]].sum() + t_n[0] + t_i[0] + t_p[0] - t_s[0],
            f_term[0, 1] - t_i[0],
            f_term[0, 2] - t_p[0],
        ]
    )
    np.testing.assert_allclose(matrix @ h, rhs)


def test_nonzero_dc_forcing_is_rejected():
    with pytest.raises(ValueError, match="zero-frequency forcing"):
        GlobalRossbyModel(geometry(), 0.02)._solve_frequency(
            forcing(t_n=(1.0, 0.0))
        )


def test_solve_transforms_monthly_forcing_and_restores_time() -> None:
    """The temporal solve preserves the monthly time coordinate."""
    input_forcing = temporal_forcing()

    result = GlobalRossbyModel(geometry(), 0.02).solve(input_forcing)

    assert set(result) == {"h_e", "h_b", "h_w", "h", "T", "T_g", "T_Ek"}
    np.testing.assert_array_equal(result.time, input_forcing.time)
    assert result.h_e.dims == ("time", "region")
    assert result.T.dims == ("time", "region", "latitude")
    assert result.h.dims == (
        "time",
        "region",
        "latitude",
        "longitude",
    )
    subset = result.h.isel(
        time=0,
        region=0,
        latitude=slice(0, 2),
        longitude=slice(0, 2),
    )
    assert np.prod(subset.data.numblocks) < np.prod(result.h.data.numblocks)
    assert float(
        result.T.sel(region="north_atlantic").dropna("latitude").latitude[-1]
    ) == 60.0
    np.testing.assert_allclose(result.T, result.T_g + result.T_Ek, equal_nan=True)
    for name in result.data_vars:
        assert np.all(np.isfinite(result[name].fillna(0)))


def test_solve_accepts_nans_only_outside_model_ocean() -> None:
    """Exterior NaNs produce the same zero-wind solution as finite zeros."""
    masked_forcing, _ = nan_masked_temporal_forcing()
    model = GlobalRossbyModel(geometry(), 0.02)

    expected = model.solve(temporal_forcing(), pad_length=0).compute()
    actual = model.solve(masked_forcing, pad_length=0).compute()

    xr.testing.assert_allclose(actual, expected)


def test_solve_rejects_nan_inside_model_ocean() -> None:
    """An active-ocean NaN remains an input error with a specific message."""
    input_forcing, ocean_mask = nan_masked_temporal_forcing()
    latitude_index, longitude_index = np.argwhere(ocean_mask)[0]
    input_forcing["M_Ek_x"].values[
        :, latitude_index, longitude_index
    ] = np.nan
    result = GlobalRossbyModel(geometry(), 0.02).solve(
        input_forcing,
        pad_length=0,
    )

    with pytest.raises(ValueError, match="M_Ek_x.*active model ocean"):
        result.h_e.compute()


def test_exterior_nan_extension_copies_nearest_ocean_values() -> None:
    """The one-cell stencil halo uses a constant nearest-ocean extension."""
    ocean_mask = np.zeros((5, 5), dtype=bool)
    ocean_mask[2, 2] = True
    values = np.full((2, 5, 5), np.nan)
    values[:, 2, 2] = (3.0, -2.0)

    extended = global_rossby._extend_exterior_wind(values, ocean_mask)

    np.testing.assert_allclose(extended[0, 1:4, 1:4], 3.0)
    np.testing.assert_allclose(extended[1, 1:4, 1:4], -2.0)
    assert np.all(extended[:, (0, 4), :] == 0.0)
    assert np.all(extended[:, :, (0, 4)] == 0.0)


def test_solve_requires_first_day_month_labels() -> None:
    input_forcing = temporal_forcing().assign_coords(
        time=temporal_forcing().time + np.timedelta64(14, "D")
    )

    with pytest.raises(ValueError, match="first day of each month"):
        GlobalRossbyModel(geometry(), 0.02).solve(
            input_forcing,
            pad_length=0,
        )


def test_solve_requires_contiguous_months() -> None:
    input_forcing = temporal_forcing().assign_coords(
        time=np.array(
            ["2000-01-01", "2000-02-01", "2000-04-01", "2000-05-01"],
            dtype="datetime64[D]",
        )
    )

    with pytest.raises(ValueError, match="months must be contiguous"):
        GlobalRossbyModel(geometry(), 0.02).solve(input_forcing, pad_length=0)


def test_solve_builds_all_outputs_lazily() -> None:
    """Constructing the model result must not execute graph tasks."""
    executed = []
    with Callback(pretask=lambda key, graph, state: executed.append(key)):
        result = GlobalRossbyModel(geometry(), 0.02).solve(
            temporal_forcing(),
            pad_length=0,
        )

    assert executed == []
    assert all(
        isinstance(array.data, Array)
        for array in result.data_vars.values()
    )
    assert result.h.dims == (
        "time",
        "region",
        "latitude",
        "longitude",
    )


def test_nan_masked_solve_builds_all_outputs_lazily() -> None:
    """Geometry-aware NaN validation and extension must remain deferred."""
    input_forcing, _ = nan_masked_temporal_forcing()
    executed = []
    with Callback(pretask=lambda key, graph, state: executed.append(key)):
        result = GlobalRossbyModel(geometry(), 0.02).solve(
            input_forcing,
            pad_length=0,
        )

    assert executed == []
    assert all(
        isinstance(array.data, Array)
        for array in result.data_vars.values()
    )


def test_height_field_retains_float32_spatial_precision() -> None:
    """The native float32 forcing does not create a float64 full field."""
    input_forcing = temporal_forcing()
    input_forcing["M_Ek_x"] = input_forcing.M_Ek_x.astype(np.float32)
    input_forcing["M_Ek_y"] = input_forcing.M_Ek_y.astype(np.float32)

    result = GlobalRossbyModel(geometry(), 0.02).solve(
        input_forcing,
        pad_length=0,
    )

    assert result.h.dtype == np.float32


def test_solve_transforms_derived_ekman_fields_not_both_winds(
    monkeypatch,
) -> None:
    """The temporal path performs one gridded FFT after linear preprocessing."""
    transformed = []
    actual_transform = global_rossby.forward_transform

    def recording_transform(data, **kwargs):
        transformed.append((data.name, data.dims))
        return actual_transform(data, **kwargs)

    monkeypatch.setattr(
        global_rossby, "forward_transform", recording_transform
    )

    GlobalRossbyModel(geometry(), 0.02).solve(
        temporal_forcing(),
        pad_length=0,
    )

    assert transformed == [
        ("T_N", ("time",)),
        ("w_Ek", ("time", "latitude", "longitude")),
        ("T_Ek", ("time", "region", "latitude")),
    ]


def test_temporal_preprocessing_matches_internal_frequency_kernel() -> None:
    """Spatial linear operations commute with the temporal transform."""
    input_forcing = temporal_forcing()
    phase = np.array([0.0, 1.0, 0.0, -1.0])[:, None, None]
    latitude = input_forcing.latitude.values[None, :, None]
    longitude = input_forcing.longitude.values[None, None, :]
    input_forcing["M_Ek_x"] = (
        ("time", "latitude", "longitude"),
        phase * (1.0 + longitude / 400.0 + 0.0 * latitude),
    )
    input_forcing["M_Ek_y"] = (
        ("time", "latitude", "longitude"),
        phase * (2.0 + latitude / 100.0 + 0.0 * longitude),
    )
    spacing = global_rossby._MODEL_MONTH_SECONDS
    model = GlobalRossbyModel(geometry(), 0.02)

    temporal = model.solve(input_forcing, pad_length=0)
    forcing_hat = xr.Dataset(
        {
            name: forward_transform(
                input_forcing[name],
                pad_length=0,
                sample_spacing_seconds=spacing,
            )
            for name in ("M_Ek_x", "M_Ek_y", "T_N")
        }
    )
    spectral = model._solve_frequency(forcing_hat)
    metadata = forcing_hat.T_N.attrs["_moc_adjustment_fourier"]
    restored = {}
    for name, array in spectral.data_vars.items():
        array = array.copy(deep=False)
        array.attrs = {
            **array.attrs,
            "_moc_adjustment_fourier": metadata,
        }
        masked = array.isnull().all("omega")
        restored[name] = inverse_transform(array.fillna(0.0)).where(~masked)

    restored = xr.Dataset(restored)
    for name in ("h_e", "h_b", "h_w", "h"):
        xr.testing.assert_allclose(
            temporal[name], restored[name], rtol=0.0, atol=1e-10
        )
    for name in ("T", "T_g", "T_Ek"):
        xr.testing.assert_allclose(
            temporal[name], restored[name], rtol=0.0, atol=1e-7
        )


def test_frequency_kernel_is_not_public() -> None:
    assert not hasattr(GlobalRossbyModel(geometry(), 0.02), "solve_frequency")


def test_default_padding_covers_longest_crossing_time() -> None:
    """Automatic padding matches an explicit crossing-time sample count."""
    input_forcing = temporal_forcing()
    spacing = global_rossby._MODEL_MONTH_SECONDS
    model = GlobalRossbyModel(geometry(), 0.02)
    expected_pad_length = int(
        np.ceil(model.longest_crossing_time_seconds / spacing)
    )

    automatic = model.solve(input_forcing)
    explicit = model.solve(
        input_forcing,
        pad_length=expected_pad_length,
    )

    assert model.longest_crossing_time_seconds > 0.0
    xr.testing.assert_allclose(automatic, explicit)
