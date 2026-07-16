# Geometry

`MultiBasinGeometry` holds the six physical boundary traces, five region
mappings and latitude bounds, and shared active-layer/isobath depth `H`.
It contains no forcing-grid mask or solver state.

## Canonical isobath product

```python
geometry = MultiBasinGeometry.from_isobath_dataset(isobaths)
```

The file convention is authoritative. It contains these six
`(latitude,)` variables:

```text
atlantic_west   atlantic_east
indian_west     indian_east
pacific_west    pacific_east
```

It also contains the `region` coordinate and `region_west_trace`,
`region_east_trace`, `region_south`, and `region_north`. The user does not
repeat those names or bounds in a potentially conflicting schema.

The loader validates the fixed five-region stitching, trace coverage,
boundary ordering, and consistency between `H` and `isobath_depth_m`. Exact
region endpoints are interpolated only inside finite trace support; missing
gateways are not extrapolated.

`geometry.x_b` and `geometry.x_e` are the resulting `(region, latitude)`
boundary views. Here `x_b` is outside the western boundary-current region,
not at the coastline.

## Producing the product

[`extract_isobath_gebco.ipynb`](https://github.com/andrewwatford/MOCAdjustmentTheory/blob/main/notebooks/extract_isobath_gebco.ipynb)
is the auxiliary workflow that writes the canonical variables and region
metadata. Raw bathymetry processing is intentionally outside the package.
