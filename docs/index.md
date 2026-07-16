# MOC Adjustment Theory

The package solves one model: the five-region `GlobalAdjustmentModel`.

```python
import xarray as xr
from moc_adjustment_theory import (
    FFTConvention,
    GlobalAdjustmentModel,
    MultiBasinGeometry,
)

geometry = MultiBasinGeometry.from_isobath_dataset(isobaths)
forcing = xr.Dataset(
    {"M_Ek_x": M_Ek_x, "M_Ek_y": M_Ek_y, "T_N": northern_transport}
)
fft = FFTConvention(
    sample_interval_seconds=365.25 * 86_400 / 12,
    n_fft=2_048,
)
output = GlobalAdjustmentModel(geometry, forcing, fft=fft).solve()
```

The model derives Ekman pumping and every section transport from the supplied
vector Ekman transport. The geometry product supplies the integration domain;
the user does not provide a second region schema or basin mask.

Start with [Geometry](geometry.md), then see the [Core API](core_api.md),
[model specification](specifications/model_architecture.md), and the single
[ERA5 + SCOTIA worked example](global_era5_scotia.md).

## Local development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs serve
```
