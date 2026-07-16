# Geometry

`MultiBasinGeometry` is a compact xarray-backed description of the six
physical boundary traces used by the five-region model. It carries the shared
active-layer/isobath depth `H`, but no wind-grid masks, forcing, or solver
state.

## Loading an isobath product

The package boundary is a compact isobath dataset, not raw bathymetry. The
variable convention and every model-region limit remain explicit:

```python
geometry = MultiBasinGeometry.from_isobath_dataset(
    isobaths,
    trace_variables={
        "atlantic_west": "x_wA",
        "atlantic_east": "x_eA",
        "indian_west": "x_wI",
        "indian_east": "x_eI",
        "pacific_west": "x_wP",
        "pacific_east": "x_eP",
    },
    region_definitions={
        "atlantic_north": {
            "west": "atlantic_west",
            "east": "atlantic_east",
            "south": y_I,
            "north": y_N,
        },
        # The other four required regions use the same schema.
    },
)
```

Every `south` and `north` value must be finite and explicit. The constructor
validates the fixed five-region stitching, trace coverage, boundary ordering,
and consistency between `H` and the dataset's `isobath_depth_m` attribute.
Explicitly requested endpoints are interpolated only when bracketed by finite
source samples; missing gateways are never extrapolated or silently filled.

The canonical dataset exposes `longitude(trace, latitude)`, `valid`, the
region trace mappings and bounds, and provenance. `geometry.x_b` and
`geometry.x_e` provide the derived `(region, latitude)` boundary views.

## Producing the product

[`extract_isobath_gebco.ipynb`](https://github.com/andrewwatford/MOCAdjustmentTheory/blob/main/notebooks/extract_isobath_gebco.ipynb)
is the auxiliary, inspectable workflow for turning GEBCO bathymetry into the
six traces. Its geographic decisions—windows, closures, and feature cleanup—
do not need a second implementation inside the package.
