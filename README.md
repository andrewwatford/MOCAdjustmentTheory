# MOC Adjustment Theory

`moc-adjustment-theory` implements a global, linear reduced-gravity model of
meridional-overturning-circulation adjustment in connected ocean basins.

## Install from source

```bash
python -m pip install .
```

## Install for development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs build --strict
```

## API

```python
import xarray as xr

from moc_adjustment_theory import GlobalRossbyModel

isobath = xr.open_dataset("data/tracked/isobath/global_isobath_GEBCO_1000m.nc")
forcing = xr.open_dataset("data/untracked/forcing/global_ERA5_SCOTIA_forcing_v3.nc")

model = GlobalRossbyModel(isobath, g_prime=0.02)
solution = model.solve(forcing)

# Every output remains lazy until a subset or complete result is requested.
atlantic_snapshot = solution.h.sel(region="north_atlantic").isel(time=0).compute()
```

Model forcing always consists of calendar-month means. This is represented by
a contiguous time coordinate on the first day of every month; no additional
temporal metadata is required. The v3 construction notebook normalises both
ERA5 and SCOTIA to that convention, and `GlobalRossbyModel.solve()` validates
the coordinate. The solver applies its linear spectral response directly to
the monthly-mean sequence; no boxcar deconvolution is needed because monthly
averaging commutes with the linear, time-invariant model.

See the [deployed documentation](https://andrewwatford.github.io/MOCAdjustmentTheory/)
for the theory, input schema, and API reference.
