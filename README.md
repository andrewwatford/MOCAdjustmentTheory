# MOC Adjustment Theory

`moc-adjustment-theory` implements the five-region reduced-gravity model of
global meridional-overturning-circulation adjustment.

The package is organized around four public objects:

- `MultiBasinGeometry` for bathymetry-derived boundary traces and `H`;
- `GlobalForcing` for aligned wind and boundary-transport anomalies;
- `GlobalAdjustmentModel` for the frequency-domain solve; and
- `GlobalAdjustmentOutput` for labelled time-domain diagnostics.

## Development installation

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs build --strict
```

Geometry is loaded from a compact six-trace isobath dataset with explicit
variable mappings and region bounds. Bathymetry extraction remains an
auxiliary notebook workflow rather than package functionality. See the
[documentation](https://andrewwatford.github.io/MOCAdjustmentTheory/) for the
architecture and geometry configuration contract.
