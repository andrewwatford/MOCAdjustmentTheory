from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from moc_adjustment_theory import (
    GlobalAdjustmentModel,
    GlobalForcing,
    MultiBasinGeometry,
)


REGIONS = {
    "atlantic_north": {
        "west": "a_west",
        "east": "a_east",
        "south": -35.0,
        "north": 55.0,
    },
    "indian_north": {
        "west": "i_west",
        "east": "i_east",
        "south": -35.0,
        "north": 25.0,
    },
    "pacific_north": {
        "west": "p_west",
        "east": "p_east",
        "south": -44.0,
        "north": 60.0,
    },
    "atlantic_indian_transition": {
        "west": "a_west",
        "east": "i_east",
        "south": -44.0,
        "north": -35.0,
    },
    "atlantic_pacific_transition": {
        "west": "a_west",
        "east": "p_east",
        "south": -56.0,
        "north": -44.0,
    },
}

TRACE_VARIABLES = {
    "a_west": "x_wA",
    "a_east": "x_eA",
    "i_west": "x_wI",
    "i_east": "x_eI",
    "p_west": "x_wP",
    "p_east": "x_eP",
}


def model_geometry() -> MultiBasinGeometry:
    latitude = np.arange(-60.0, 66.0)

    def trace(value: float, south: float, north: float) -> tuple[str, np.ndarray]:
        data = np.full(latitude.size, np.nan)
        data[(latitude >= south) & (latitude <= north)] = value
        return "latitude", data

    dataset = xr.Dataset(
        {
            "x_wA": trace(-60.0, -56.0, 65.0),
            "x_eA": trace(20.0, -35.0, 65.0),
            "x_wI": trace(30.0, -35.0, 30.0),
            "x_eI": trace(90.0, -44.0, 30.0),
            "x_wP": trace(110.0, -44.0, 65.0),
            "x_eP": trace(280.0, -56.0, 65.0),
        },
        coords={"latitude": latitude},
        attrs={"isobath_depth_m": 1000.0},
    )
    return MultiBasinGeometry.from_isobath_dataset(
        dataset,
        trace_variables=TRACE_VARIABLES,
        region_definitions=REGIONS,
    )


def model_forcing(
    *,
    ekman_amplitude: float = 0.0,
    northern_amplitude: float = 1.0,
    southern_amplitude: float = 0.0,
    grid_spacing: float = 5.0,
    zonal_fraction: float = 0.2,
) -> GlobalForcing:
    time = np.arange("2001-01-01", "2001-01-13", dtype="datetime64[D]")
    latitude = np.arange(-60.0, 65.0 + 0.5 * grid_spacing, grid_spacing)
    longitude = np.arange(0.0, 360.0, grid_spacing)
    phase = 2.0 * np.pi * np.arange(time.size) / time.size
    spatial = (
        np.cos(np.deg2rad(latitude))[None, :, None]
        * (1.0 + 0.2 * np.cos(np.deg2rad(longitude)))[None, None, :]
    )
    temporal = np.sin(phase)[:, None, None]
    M_ek_y = xr.DataArray(
        ekman_amplitude * temporal * spatial,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        attrs={"units": "m2 s-1"},
    )
    zonal_structure = np.sin(np.deg2rad(longitude))[None, None, :]
    M_ek_x = xr.DataArray(
        zonal_fraction
        * ekman_amplitude
        * temporal
        * np.cos(np.deg2rad(latitude))[None, :, None]
        * zonal_structure,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        attrs={"units": "m2 s-1"},
    )
    northern = xr.DataArray(
        northern_amplitude * np.cos(phase),
        dims="time",
        coords={"time": time},
        attrs={"units": "Sv"},
    )
    southern = xr.DataArray(
        southern_amplitude * np.cos(phase),
        dims="time",
        coords={"time": time},
    )
    southern.attrs["units"] = "Sv"
    return GlobalForcing.from_time_series(
        M_ek_x=M_ek_x,
        M_ek_y=M_ek_y,
        northern_transport=northern,
        southern_transport=southern,
        padding_samples=time.size - 1,
        n_fft=64,
    )


def test_zero_ekman_solution_closes_budgets_and_returns_every_field() -> None:
    forcing = model_forcing()
    output = GlobalAdjustmentModel(model_geometry(), forcing).solve()

    assert set(output.dataset.data_vars) >= {
        "h_e",
        "h_b",
        "h_w",
        "transport",
        "transport_ekman",
        "transport_geostrophic",
        "compatibility_residual",
    }
    xr.testing.assert_allclose(
        output.transport,
        output.transport_ekman + output.transport_geostrophic,
    )
    assert float(abs(output.spectral.southern_budget_residual).max()) < 1e-6
    assert float(abs(output.transport_ekman).max()) < 1e-12
    assert float(abs(output.spectral.compatibility_residual).max()) < 1e-12
    assert np.isfinite(output.spectral.condition_number.isel(omega=slice(1, None))).all()
    assert output.spectral.attrs["n_fft"] == 64
    assert output.dataset.attrs["padding_mode"] == "reflect"
    assert output.dataset.attrs["time_mean_removed"] is True
    np.testing.assert_allclose(
        output.h_e.sel(region="indian_north"),
        output.h_e.sel(region="atlantic_indian_transition"),
    )
    np.testing.assert_allclose(
        output.h_e.sel(region="pacific_north"),
        output.h_e.sel(region="atlantic_pacific_transition"),
    )

    spectral = output.spectral
    indian_south = spectral.transport.sel(region="indian_north").dropna(
        "latitude", how="all"
    ).isel(latitude=0)
    pacific_south = spectral.transport.sel(region="pacific_north").dropna(
        "latitude", how="all"
    ).isel(latitude=0)
    kappa_I = 0.02 * 1000.0 / (
        2.0 * 7.292115e-5 * np.sin(np.deg2rad(-35.0))
    )
    kappa_P = 0.02 * 1000.0 / (
        2.0 * 7.292115e-5 * np.sin(np.deg2rad(-44.0))
    )
    np.testing.assert_allclose(
        indian_south,
        kappa_I
        * (
            spectral.h_e.sel(region="indian_north")
            - spectral.h_e.sel(region="atlantic_north")
        ),
        rtol=1e-10,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pacific_south,
        kappa_P
        * (
            spectral.h_e.sel(region="pacific_north")
            - spectral.h_e.sel(region="indian_north")
        ),
        rtol=1e-10,
        atol=1e-6,
    )


def test_vector_transport_is_derived_into_ekman_and_geostrophic_components() -> None:
    output = GlobalAdjustmentModel(
        model_geometry(), model_forcing(ekman_amplitude=0.1)
    ).solve()

    assert float(abs(output.transport_ekman).max()) > 0.0
    assert np.isfinite(output.spectral.compatibility_residual).all()
    relation = output.h_e - (
        2.0
        * 7.292115e-5
        * np.sin(np.deg2rad(output.h_w.latitude))
        * output.transport_geostrophic
        / (0.02 * 1000.0)
    )
    xr.testing.assert_allclose(output.h_w, relation)


def test_compatibility_residual_converges_with_grid_refinement() -> None:
    geometry = model_geometry()
    coarse = GlobalAdjustmentModel(
        geometry,
        model_forcing(
            ekman_amplitude=0.1,
            grid_spacing=5.0,
            zonal_fraction=0.0,
        ),
    ).solve()
    fine = GlobalAdjustmentModel(
        geometry,
        model_forcing(
            ekman_amplitude=0.1,
            grid_spacing=2.5,
            zonal_fraction=0.0,
        ),
    ).solve()

    coarse_error = float(abs(coarse.spectral.compatibility_residual).max())
    fine_error = float(abs(fine.spectral.compatibility_residual).max())
    assert fine_error < coarse_error


def test_vectorized_solution_matches_one_direct_frequency_solve() -> None:
    forcing = model_forcing(ekman_amplitude=0.1)
    output = GlobalAdjustmentModel(model_geometry(), forcing).solve()
    spectral = output.spectral
    index = 4
    F = spectral.F.isel(omega=index)
    r = spectral.r.isel(omega=index)
    kappa_I = 0.02 * 1000.0 / (
        2.0 * 7.292115e-5 * np.sin(np.deg2rad(-35.0))
    )
    kappa_P = 0.02 * 1000.0 / (
        2.0 * 7.292115e-5 * np.sin(np.deg2rad(-44.0))
    )
    transport_I_ekman = spectral.transport_ekman.sel(
        region="indian_north"
    ).dropna("latitude", how="all").isel(omega=index, latitude=0)
    transport_P_ekman = spectral.transport_ekman.sel(
        region="pacific_north"
    ).dropna("latitude", how="all").isel(omega=index, latitude=0)
    matrix = np.array(
        [
            [
                r.sel(region="atlantic_north") + kappa_I,
                r.sel(region="atlantic_indian_transition")
                + kappa_P
                - kappa_I,
                r.sel(region="atlantic_pacific_transition") - kappa_P,
            ],
            [
                -kappa_I,
                r.sel(region="indian_north") + kappa_I,
                0.0,
            ],
            [
                0.0,
                -kappa_P,
                r.sel(region="pacific_north") + kappa_P,
            ],
        ],
        dtype=complex,
    )
    rhs = np.array(
        [
            F.sel(region="atlantic_north")
            + F.sel(region="atlantic_indian_transition")
            + F.sel(region="atlantic_pacific_transition")
            + forcing.spectral.northern_transport.isel(omega=index)
            + transport_I_ekman
            + transport_P_ekman
            - forcing.spectral.southern_transport.isel(omega=index),
            F.sel(region="indian_north") - transport_I_ekman,
            F.sel(region="pacific_north") - transport_P_ekman,
        ],
        dtype=complex,
    )
    expected = np.linalg.solve(matrix, rhs)
    actual = np.array(
        [
            spectral.h_e.sel(region="atlantic_north").isel(omega=index),
            spectral.h_e.sel(region="indian_north").isel(omega=index),
            spectral.h_e.sel(region="pacific_north").isel(omega=index),
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_transport_at_interpolates_only_inside_region_support() -> None:
    output = GlobalAdjustmentModel(model_geometry(), model_forcing()).solve()

    transport = output.transport_at("atlantic_north", 12.5)
    assert transport.dims == ("time",)
    with pytest.raises(ValueError, match="outside"):
        output.transport_at("atlantic_north", -40.0)


def test_external_transports_are_positive_northward() -> None:
    northern = GlobalAdjustmentModel(
        model_geometry(),
        model_forcing(northern_amplitude=1.0, southern_amplitude=0.0),
    ).solve()
    southern = GlobalAdjustmentModel(
        model_geometry(),
        model_forcing(northern_amplitude=0.0, southern_amplitude=1.0),
    ).solve()

    np.testing.assert_allclose(southern.spectral.h_e, -northern.spectral.h_e)


def test_rectangular_constant_pumping_anchors_P_F_and_r_signs() -> None:
    earth_radius = 6.371e6
    rotation_rate = 7.292115e-5
    time = np.arange("2001-01-01", "2001-01-13", dtype="datetime64[D]")
    latitude = np.arange(-60.0, 66.0, 5.0)
    longitude = np.arange(-179.75, 180.0, 0.5)
    phase = 2.0 * np.pi * np.arange(time.size) / time.size
    pumping = xr.DataArray(
        1e-7 * np.sin(phase),
        dims="time",
        coords={"time": time},
    )
    M_ek_x = xr.DataArray(
        np.asarray(pumping)[:, None, None]
        * earth_radius
        * np.cos(np.deg2rad(latitude))[None, :, None]
        * np.deg2rad(longitude)[None, None, :],
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        attrs={"units": "m2 s-1"},
    )
    M_ek_y = xr.zeros_like(M_ek_x)
    M_ek_y.attrs["units"] = "m2 s-1"
    zero = xr.DataArray(
        np.zeros(time.size),
        dims="time",
        coords={"time": time},
        attrs={"units": "Sv"},
    )
    forcing = GlobalForcing.from_time_series(
        M_ek_x=M_ek_x,
        M_ek_y=M_ek_y,
        northern_transport=zero,
        southern_transport=zero,
        padding_samples=time.size - 1,
        n_fft=64,
    )
    term = GlobalAdjustmentModel(model_geometry(), forcing)._region_terms(
        "atlantic_north"
    ).compute()

    index = 3
    omega = float(forcing.omega[index])
    regional_latitude = np.asarray(term.latitude)
    f = 2.0 * rotation_rate * np.sin(np.deg2rad(regional_latitude))
    beta = (
        2.0
        * rotation_rate
        * np.cos(np.deg2rad(regional_latitude))
        / earth_radius
    )
    uncapped = np.full_like(f, np.inf)
    np.divide(beta * 0.02 * 1000.0, f**2, out=uncapped, where=f != 0.0)
    c = np.minimum(uncapped, np.sqrt(0.02 * 1000.0) / 3.0)
    width = (
        earth_radius
        * np.cos(np.deg2rad(regional_latitude))
        * np.deg2rad(80.0)
    )
    eastern_phase = np.exp(-1j * omega * width / c)
    y = earth_radius * np.deg2rad(regional_latitude)
    expected_r = np.trapezoid(c * (eastern_phase - 1.0), x=y)
    pumping_hat = complex(forcing.transform(pumping).isel(omega=index))
    expected_F = np.trapezoid(
        pumping_hat
        * (c / (1j * omega) * (1.0 - eastern_phase) - width),
        x=y,
    )

    assert complex(term.r.isel(omega=index)) == pytest.approx(
        expected_r, rel=1e-12, abs=1e-8
    )
    actual_F = complex(term.F.isel(omega=index))
    assert abs(actual_F - expected_F) / abs(expected_F) < 2e-3
