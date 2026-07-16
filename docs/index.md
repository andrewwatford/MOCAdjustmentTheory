# MOC Adjustment Theory

`moc-adjustment-theory` implements the fixed five-region reduced-gravity model
of global overturning adjustment. Its public workflow has four objects and one
solve. The northern Indian and Pacific basins are closed; the model has no
Indonesian Throughflow connection or transport input.

```python
from moc_adjustment_theory import (
    GlobalAdjustmentModel,
    GlobalForcing,
    MultiBasinGeometry,
)

geometry = MultiBasinGeometry.from_isobath_dataset(
    isobaths,
    trace_variables=trace_variables,
    region_definitions=region_definitions,
)

forcing = GlobalForcing.from_time_series(
    M_ek_x=M_ek.x,
    M_ek_y=M_ek.y,
    northern_transport=northern_transport,
    southern_transport=southern_transport,
)

output = GlobalAdjustmentModel(geometry, forcing, g_prime=0.02).solve()
```

The supplied Ekman vector transport is the scientific package boundary. The
package derives Ekman pumping, all section Ekman transports, the regional
forcing terms, and the complete time-dependent solution. Conversion from wind
stress—including reference density, equatorial regularization, and coastal
tapering—remains an upstream user choice.

Start with [Geometry](geometry.md) for the compact isobath interface, then see
the [Core API](core_api.md) and focused
[model specification](specifications/model_architecture.md).

## Local development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs serve
```
