# MOC Adjustment Theory

`moc-adjustment-theory` implements the five-region reduced-gravity model of
global meridional-overturning-circulation adjustment. The framework uses the
closed-Indian/closed-Pacific topology and does not include an Indonesian
Throughflow connection.

The package is organized around four public objects:

- `MultiBasinGeometry` for bathymetry-derived boundary traces and `H`;
- `GlobalForcing` for aligned Ekman and boundary-transport anomalies;
- `GlobalAdjustmentModel` for the frequency-domain solve; and
- `GlobalAdjustmentOutput` for labelled time-domain diagnostics.

## Development installation

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs build --strict
```

To run the worked notebooks, install the example dependencies and point them
at the local scientific-data checkout:

```bash
python -m pip install -e '.[examples]'
export MOC_EXAMPLE_DATA_ROOT=/path/to/data/untracked
```

Geometry is loaded from a compact six-trace isobath dataset with explicit
variable mappings and region bounds. Bathymetry extraction remains an
auxiliary notebook workflow rather than package functionality. See the
[documentation](https://andrewwatford.github.io/MOCAdjustmentTheory/) for the
architecture and geometry configuration contract.
