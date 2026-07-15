# GEBCO-to-MultiBasinGeometry specification

Status: proposed geometry contract for review. This document fixes the
scientific geometry, minimum extraction behaviour, output, and acceptance
tests. Numerical implementation details may remain private.

## 1. Purpose and fixed conventions

The geometry pipeline converts the local GEBCO 2026 sub-ice bathymetry into
the six shared boundary traces used by the fixed non-Indonesian-Throughflow
(non-ITF) model. Those traces define five `Basin` geometries and are assembled
into a `MultiBasinGeometry` for construction of `MultiBasinTopology`.

The first implementation uses these conventions:

| Quantity | Convention |
|---|---|
| Isobath and active-layer depth | one shared `H = 1000 m` |
| Southern model boundary | `y_S = -56` degrees north |
| Pacific junction | `y_P = -44` degrees north |
| Indian junction | `y_I = -35` degrees north |
| Atlantic northern boundary | `y_N = 55` degrees north |
| Western boundary coordinate | `x_b`, the western 1000 m isobath |
| Eastern boundary coordinate | `x_e`, the eastern 1000 m isobath |
| Global topology | fixed non-ITF graph |

The extraction depth and active-layer depth are the same parameter. The
pipeline does not maintain a separate western-wall coordinate or WBC-width
prescription: for this theory, the western isobath is `x_b`.

The model domain ends at the southern tip of South America. Bathymetry south
of `y_S` may be used locally to locate the endpoint, but it cannot create a
path through the Southern Ocean or around Antarctica.

## 2. Source data contract

The initial source is:

```text
data/untracked/GEBCO/GEBCO_2026_sub_ice/GEBCO_2026_sub_ice.nc
```

Expected properties are:

| Property | Expected value |
|---|---|
| Product | GEBCO 2026 sub-ice global grid |
| DOI | `10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa` |
| Grid | pixel-centred latitude/longitude |
| CRS | EPSG:4326 |
| Resolution | 15 arc-second |
| Dimensions | `lat=43200`, `lon=86400` |
| Variable | `elevation(lat, lon)` |
| Units/sign | metres, positive upward |
| Initial SHA-256 | `9a338345b7a8b8614718ccd551be4be6be629e24cca50f1bc764bdf3ea6e9c3c` |

Source validation checks the file identity, dimensions, coordinates, cyclic
longitude coverage, variable name, missing values, and elevation sign. A
different GEBCO file is a new input and receives a new provenance identity.

The approximately 7 GB source is processed in chunks. Ordinary unit tests use
small synthetic grids; the full GEBCO file is required only for opt-in
integration tests and production extraction.

## 3. Required traces and basins

The extractor produces exactly six physical traces:

1. `atlantic_west`;
2. `atlantic_east`;
3. `indian_west`;
4. `indian_east`;
5. `pacific_west`;
6. `pacific_east`.

Western traces provide `x_b`; eastern traces provide `x_e`. The five basins
reference these shared traces rather than copying or re-extracting them:

| Basin key | Latitude interval | `x_b` trace | `x_e` trace |
|---|---|---|---|
| `atlantic_north` | `y_I` to `y_N` | `atlantic_west` | `atlantic_east` |
| `indian_north` | `y_I` to `y_NI` | `indian_west` | `indian_east` |
| `pacific_north` | `y_P` to `y_NP` | `pacific_west` | `pacific_east` |
| `atlantic_indian_transition` | `y_P` to `y_I` | `atlantic_west` | `indian_east` |
| `atlantic_pacific_transition` | `y_S` to `y_P` | `atlantic_west` | `pacific_east` |

`y_NI` and `y_NP` use one deterministic convention: each is the northernmost
sampled latitude where that basin's west and east traces are both finite. They
may lie north of `y_N`, which limits only the Atlantic domain. The selected
values are stored in the geometry.

The corresponding topology has the east-to-west child order:

- `atlantic_pacific_transition` ->
  `[pacific_north, atlantic_indian_transition]`;
- `atlantic_indian_transition` ->
  `[indian_north, atlantic_north]`.

The geometry records the shared junctions but does not contain forcing or
model equations.

## 4. Coordinate and trimming conventions

### 4.1 Longitude and latitude

Public coordinates are degrees east and degrees north. Longitudes are kept on
a continuous branch for each ocean during extraction and validation:

| Ocean | Working longitude frame |
|---|---|
| Atlantic | centred near 0 degrees |
| Indian | centred near 75 degrees east |
| Pacific | continuous through 180 degrees |

Wrapped `[-180, 180)` longitudes are display-only. The Pacific trace is never
smoothed or differenced after wrapping across the antimeridian.

GEBCO coordinates are pixel centres. A contour crossing is located between
adjacent cell centres, not at an assumed cell edge. Metric calculations use

$$
dy=R\,d\phi,
\qquad
dx=R\cos\phi\,d\lambda.
$$

### 4.2 Fixed boundaries and grid sampling

`y_S`, `y_P`, `y_I`, and `y_N` are exact geometry coordinates. Extraction
must provide finite traces at those values or fail. A later forcing-grid
adapter may select the nearest in-domain grid row and record that sampled
latitude, but it does not redefine the geometry or move a shared gateway.

### 4.3 Inward trimming

`MultiBasinGeometry.trim(south=None, north=None)` returns a non-mutating view
whose bounds lie inside the extracted interval. It can raise `y_S`, lower
`y_N`, or do both, without re-running the isobath extraction. Basin intervals
and traces are clipped consistently, provenance is retained, and the method
never extrapolates beyond available geometry. Any model consuming a trimmed
view still validates that its required basins and gateways remain present.

## 5. Extraction requirements

The implementation may choose its internal data structures, but it must
perform the following scientific operations.

### 5.1 Build the 1000 m ocean mask

Convert positive-up elevation `z` to positive-down depth `d=-z`. Distinguish
land, ocean shallower than `H`, and ocean deeper than `H`. Truncate the
connectivity mask at `y_S` before identifying the model ocean.

The working grid may be coarsened for search, provided final crossings are
checked against native-resolution GEBCO tiles. The working resolution and
coarsening method are recorded, not exposed as part of the scientific API.

### 5.2 Apply the fixed non-ITF closures

Where the raw 1000 m mask does not express the intended model topology, use
small named, shelf-following closures. The initial set covers the Indonesian
Throughflow, Bering Strait, the Greenland-Iceland-UK/European connection where
needed, and the Caribbean/Bahamas attachment where needed.

Each closure records its name and affected cells. Broad rectangular dams are
not acceptable. Madagascar and New Zealand are excluded from outer-boundary
selection so that their island contours cannot replace a continental basin
boundary.

### 5.3 Select continuous basin boundaries

Identify the connected deep-ocean component within the truncated model domain
and partition its candidate intervals into Atlantic, Indian, and Pacific
search regions. At each latitude, locate plausible west and east 1000 m
crossings, then select a continuous sequence through latitude. Selection must
penalize implausible jumps and branch switches; independent row-wise extrema
are insufficient near islands and marginal seas.

The precise tracking algorithm is an implementation choice. It must be
deterministic for a fixed source and configuration and must flag rows where no
reasonable crossing exists.

### 5.4 Refine and regularize

Refine selected crossings against native-resolution bathymetry. Short isolated
gaps or spikes may be repaired with a conservative, recorded rule. Any
smoothing must remain close to the target isobath and preserve both the raw and
final trace. Long gaps, large jumps, and unresolved branch ambiguity are
failures rather than invitations to draw an arbitrary smooth curve.

### 5.5 Assemble shared geometry

Construct the six traces and five basin views in Section 3. Validate
`x_e(y) > x_b(y)` everywhere, exact trace sharing at composite basins, finite
coverage across every basin interval, and the common gateway latitudes. The
Atlantic single-basin geometry is a view of the same Atlantic traces, not a
second extraction.

## 6. Minimal geometry objects

### `BoundaryTrace`

A trace contains:

- its key and west/east role;
- latitude and continuous-longitude arrays;
- raw and final longitudes;
- valid-domain and repaired-point flags;
- source/configuration provenance.

### `MultiBasinGeometry`

The geometry contains:

- the six shared `BoundaryTrace` objects;
- the five basin-to-trace mappings and latitude intervals;
- `y_S`, `y_P`, `y_I`, `y_N`, `y_NI`, and `y_NP`;
- the target/active-layer depth `H`;
- source and extraction provenance;
- `trim(...)` and an Atlantic-view helper.

This is the geometric input used to construct the `Basin` objects held by
`MultiBasinTopology`. It does not define a separate WBC geometry, generic
physical-ocean hierarchy, polygons, or forcing-grid objects.

## 7. Output and visual review

The primary NetCDF or `xarray.Dataset` contains at least:

```text
coordinates:
    trace
    latitude

variables:
    longitude_raw(trace, latitude)
    longitude(trace, latitude)
    longitude_wrapped(trace, latitude)  # display only
    valid(trace, latitude)
    repaired(trace, latitude)
    basin_west_trace(basin)
    basin_east_trace(basin)
    basin_southern_latitude(basin)
    basin_northern_latitude(basin)
```

Global attributes record the GEBCO DOI/checksum, `H`, algorithm version,
configuration, Git commit, creation time, and longitude conventions.

The required review figure shows:

- the 1000 m bathymetry and all six final traces on a global map truncated at
  `y_S`;
- the fixed closures and any repaired/flagged sections;
- the five basin geometries in distinct colours;
- basin width as a function of latitude.

The result need not carry a large catalogue of optional spatial products. A
clear map and the numerical checks below are sufficient for initial acceptance.

## 8. Acceptance and tests

A production geometry is accepted only when:

1. exactly six traces and five basin mappings match Section 3;
2. all in-domain samples are finite and satisfy `x_e > x_b`;
3. `y_S < y_P < y_I < y_N`, and each shared gateway has one latitude;
4. Pacific longitudes remain continuous across the antimeridian;
5. natural trace points lie on the 1000 m isobath within a declared tolerance;
6. repairs are sparse, labelled, and below a declared displacement limit;
7. the non-ITF closures are named, narrow, and visible in the review figure;
8. island or marginal-sea contours do not replace the intended outer boundary;
9. identical input and configuration reproduce the same numerical traces;
10. the Atlantic view uses the same trace objects as the global geometry;
11. inward trimming clips every affected trace and basin consistently;
12. the domain contains no Antarctic/Southern-Ocean connection south of
    `y_S`.

Unit tests use small cyclic synthetic grids to cover contour interpolation,
the dateline, islands, a closed strait, branch continuity, a short repair,
trace sharing, and inward trimming. Opt-in integration tests validate the
actual GEBCO source and representative extracted sections. One full extraction
produces the review figure and checks all acceptance conditions before the
geometry is adopted.

The historical 500 m and 1000 m products are useful regression comparisons,
but the new default and production target is 1000 m. Visual agreement with a
legacy trace does not replace the acceptance checks above.
