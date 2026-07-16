# MOC Adjustment Theory

`moc-adjustment-theory` implements the five-region, reduced-gravity model of
global meridional-overturning-circulation adjustment.

The public workflow is deliberately small:

1. load a canonical `MultiBasinGeometry` isobath product;
2. put `M_Ek_x`, `M_Ek_y`, and northern total transport `T_N` in one
   `xarray.Dataset`;
3. choose an `FFTConvention`; and
4. solve with `GlobalAdjustmentModel`.

The result contains `h_e`, `h_b`, `h_w`, and total, Ekman, and geostrophic
transport for all five regions.

## Development installation

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs build --strict
```

To run the worked example, install the example dependencies and point it at
the local ERA5 and SCOTIA data checkout:

```bash
python -m pip install -e '.[examples]'
export MOC_EXAMPLE_DATA_ROOT=/path/to/data/untracked
```

See the [documentation](https://andrewwatford.github.io/MOCAdjustmentTheory/)
for the geometry convention, API, equations, and ERA5 + SCOTIA example.
