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
forcing = xr.open_dataset("data/untracked/forcing/global_ERA5_SCOTIA_forcing.nc")

model = GlobalRossbyModel(isobath, g_prime=0.02)
solution = model.solve(
    forcing,
    sample_spacing_seconds=365.25 / 12 * 24 * 60 * 60,
)
```

See the [deployed documentation](https://andrewwatford.github.io/MOCAdjustmentTheory/)
for the theory, input schema, and API reference.
