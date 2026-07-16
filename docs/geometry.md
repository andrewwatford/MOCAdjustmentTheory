# Geometry

`MultiBasinGeometry` is a compact xarray-backed description of the six
physical boundary traces used by the five-region model. It carries the shared
active-layer/isobath depth `H`, but no wind-grid masks, forcing, or solver
state.

## Extracting from bathymetry

Geography is configuration, not library code. `from_bathymetry` therefore
requires the basin search windows and seeds, every regional latitude bound,
the region-to-trace mappings, and any artificial closure or ignored-feature
geometry:

```python
geometry = MultiBasinGeometry.from_bathymetry(
    elevation,
    H=1_000.0,
    basin_definitions={
        "atlantic": {
            "longitude_bounds": (-100.0, 40.0),
            "seed": (-30.0, 0.0),
        },
        # Indian and Pacific definitions omitted here for brevity.
    },
    region_definitions={
        "atlantic_north": {
            "west": "atlantic_west",
            "east": "atlantic_east",
            "south": -35.0,
            "north": 55.0,
        },
        # The other four required regions follow the same schema.
    },
    closures=closure_paths,
    ignored_features=ignored_features,
    extraction_options={
        "positive": "up",
        "coarsen_factor": 4,
        "maximum_gap_degrees": 0.25,
        "smoothing_sigma_degrees": 0.35,
    },
)
```

The numbers above illustrate configuration used by the GEBCO extraction
notebook; they are not package defaults. Every `south` and `north` value must
be finite and explicit. In particular, the extractor does not infer a northern
closure from the last connected bathymetry row.

An empty `closures` argument applies no implicit Indonesian, Caribbean,
Bering, or other geographic closure. A closure supplies its affected basin
names, polyline points, and width. An ignored feature similarly supplies basin
names and a rectangular bound. Unknown basin names and extraction options are
errors, so misspelled scientific configuration cannot be silently ignored.

The bathymetry must be a global `xarray.DataArray` with one-dimensional
`latitude` and `longitude` coordinates and metre units. Its vertical sign must
be explicit in CF metadata or in `extraction_options["positive"]`.
Extraction remains lazy until each configured basin latitude/longitude window
has been selected, allowing a chunked global GEBCO array to be used without
materializing the global grid.

## Loading the tracked isobath product

The notebook output can be loaded without rerunning the 7 GB source-grid
extraction. Variable and region conventions remain explicit:

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
    region_definitions=region_definitions,
)
```

The loader interpolates explicitly requested endpoints only when they are
bracketed by finite trace samples. It never extrapolates a missing gateway.

The canonical dataset exposes `longitude(trace, latitude)`, `valid`, the
region trace mappings and bounds, and provenance. `geometry.x_b` and
`geometry.x_e` provide the derived `(region, latitude)` boundary views.

The canonical five region roles are fixed by the global model algebra. Their
coordinates and physical trace names are still supplied configuration; they
are not embedded geographic constants.
