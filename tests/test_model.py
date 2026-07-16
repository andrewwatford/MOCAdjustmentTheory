import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import GlobalAdjustmentModel


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


def forcing(t_n=(0.0, 1.0e6), wind=False):
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
    return xr.Dataset(
        {
            "M_Ek_x": (("time", "latitude", "longitude"), zeros),
            "M_Ek_y": (("time", "latitude", "longitude"), zeros),
            "T_N": ("time", [0.0, 1.0e6, 0.0, -1.0e6]),
        },
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    )


def test_unforced_wind_solution_closes_regional_budgets():
    result = GlobalAdjustmentModel(geometry(), 0.02).solve_frequency(forcing())

    assert set(result.data_vars) == {"h_e", "h_b", "h_w", "T", "T_g", "T_Ek"}
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


def test_transition_topology_conserves_transport():
    result = GlobalAdjustmentModel(geometry(), 0.02).solve_frequency(forcing())
    t = result.T.isel(omega=1)

    north_4 = t.sel(region="atlantic_indian").dropna("latitude")[-1]
    south_1 = t.sel(region="north_atlantic").dropna("latitude")[0]
    south_2 = t.sel(region="north_indian").dropna("latitude")[0]
    np.testing.assert_allclose(north_4, south_1 + south_2)

    north_5 = t.sel(region="atlantic_pacific").dropna("latitude")[-1]
    south_3 = t.sel(region="north_pacific").dropna("latitude")[0]
    south_4 = t.sel(region="atlantic_indian").dropna("latitude")[0]
    np.testing.assert_allclose(north_5, south_3 + south_4)


def test_nonzero_ekman_forcing_produces_consistent_diagnostics():
    result = GlobalAdjustmentModel(geometry(), 0.02).solve_frequency(
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


def test_boundary_solution_satisfies_three_basin_system():
    omega = np.array([1.0e-6])
    f_term = np.array([[1 + 1j, 2 - 1j, 3 + 2j, 4 - 2j, 5 + 0.5j]])
    r = np.array([[2 + 3j, 3 + 4j, 4 + 5j, 5 + 6j, 6 + 7j]])
    t_n, t_i, t_p, t_s = (np.array([value + 0j]) for value in (7, 8, 9, 10))
    k_i, k_p = -20.0, -30.0
    h = GlobalAdjustmentModel._solve_boundaries(
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
        GlobalAdjustmentModel(geometry(), 0.02).solve_frequency(forcing(t_n=(1.0, 0.0)))


def test_solve_transforms_monthly_forcing_and_restores_time() -> None:
    """The temporal solve applies one FFT contract in both directions."""
    input_forcing = temporal_forcing()
    spacing = 365.25 / 12 * 24 * 60 * 60

    result = GlobalAdjustmentModel(geometry(), 0.02).solve(
        input_forcing,
        sample_spacing_seconds=spacing,
    )

    assert set(result) == {"h_e", "h_b", "h_w", "T", "T_g", "T_Ek"}
    np.testing.assert_array_equal(result.time, input_forcing.time)
    assert result.h_e.dims == ("time", "region")
    assert result.T.dims == ("time", "region", "latitude")
    np.testing.assert_allclose(result.T, result.T_g + result.T_Ek, equal_nan=True)
    for name in result.data_vars:
        assert np.all(np.isfinite(result[name].fillna(0)))


def test_solve_infers_daily_spacing() -> None:
    """A uniform daily grid needs no explicit physical interval."""
    input_forcing = temporal_forcing().assign_coords(
        time=np.datetime64("2000-01-01")
        + np.arange(4) * np.timedelta64(1, "D")
    )

    result = GlobalAdjustmentModel(geometry(), 0.02).solve(
        input_forcing,
        omega_dim="angular_frequency",
    )

    np.testing.assert_array_equal(result.time, input_forcing.time)
    assert "angular_frequency" not in result.dims
