# GEBCO-to-MultiBasinGeometry specification

Status: proposed data-processing and geometry contract for review. This
document specifies the algorithm, provenance, output schema, and acceptance
tests. It does not implement the extraction pipeline.

## 1. Purpose and scope

The geometry pipeline converts the local GEBCO 2026 sub-ice bathymetry into:

1. six shared, single-valued source bathymetric shelf/isobath traces;
2. five non-Indonesian-Throughflow dynamical region geometries assembled from
   those traces;
3. the directed-region interfaces and physical-ocean views required by the
   global model;
4. a lossless Atlantic-only view using the same Atlantic traces;
5. machine-readable provenance and a QA report sufficient to accept or reject
   a result without relying on visual plausibility alone.

The pipeline is designed for configurable isobath depths such as 500 m or
1000 m. The extraction depth, active-layer depth `H`, and reduced gravity
`g_prime` are separate scientific choices and must never be inferred from one
another.

The output is essentially `MultiBasinGeometry`, but bathymetry alone cannot
determine every physical quantity it contains. In particular, GEBCO determines
the western shelf trace `x_w`; it does not determine the offshore edge `x_b`
of the unresolved western boundary current (WBC) region. Section 10 specifies
how that distinction is preserved.

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
| Grid | pixel-centred geographic latitude/longitude |
| CRS | EPSG:4326 |
| Resolution | 15 arc-second |
| Dimensions | `lat=43200`, `lon=86400` |
| Longitude | approximately `-179.9979` to `179.9979` degrees east |
| Latitude | approximately `-89.9979` to `89.9979` degrees north |
| Variable | `elevation(lat, lon)` |
| Storage/units | signed integer metres, positive upward |
| Initial SHA-256 | `9a338345b7a8b8614718ccd551be4be6be629e24cca50f1bc764bdf3ea6e9c3c` |

The sibling GEBCO documentation and terms PDFs remain with the local source and
are recorded in the repository's tracked data manifest.

Before processing, `GEBCOSource.validate()` must verify the DOI/version,
checksum, dimensions, coordinate monotonicity, cyclic longitude coverage,
pixel-centre registration, CRS, variable name, fill values, and positive-up
elevation convention. A new GEBCO release or changed file is a new input and
must generate a new provenance hash; it must not pass silently because its
variable happens to have the same name.

The 7 GB grid is always processed out of core. The implementation opens it in
latitude-oriented chunks, never calls an unrestricted `.load()`, and records
the chunk/reduction plan. A representative small fixture and synthetic grids
provide unit-test coverage without requiring GEBCO in ordinary CI.

## 3. Required physical traces and regions

The global non-ITF geometry uses six unique source bathymetric traces:

1. `atlantic_west`;
2. `atlantic_east`;
3. `indian_west`;
4. `indian_east`;
5. `pacific_west`;
6. `pacific_east`.

They have independent valid latitude domains. Approximate initial southern
limits from the source work are:

- Atlantic west: `y_S = -56` degrees north;
- Atlantic east: `y_I = -35` degrees north;
- Indian west: `y_I = -35` degrees north;
- Indian east: `y_P = -44` degrees north;
- Pacific west: `y_P = -44` degrees north;
- Pacific east: `y_S = -56` degrees north.

Northern trace limits follow the configured physical basin closures. The
global model's Atlantic northern boundary is initially `y_N=55` degrees north;
the Indian and Pacific limits `y_NI` and `y_NP` are explicit and may differ.

The five region views are assembled by reference, not by re-extracting a new
contour:

| Region | Latitude interval | West trace | East trace |
|---|---|---|---|
| `atlantic_north` | `y_I` to `y_N` | `atlantic_west` | `atlantic_east` |
| `indian_north` | `y_I` to `y_NI` | `indian_west` | `indian_east` |
| `pacific_north` | `y_P` to `y_NP` | `pacific_west` | `pacific_east` |
| `indo_atlantic` | `y_P` to `y_I` | `atlantic_west` | `indian_east` |
| `atlantic_pacific` | `y_S` to `y_P` | `atlantic_west` | `pacific_east` |

Junction latitudes are reviewed configuration values, not values inferred from
the first finite sample in a noisy trace. The contract distinguishes three
latitude layers:

1. **requested physical latitude**: the reviewed scientific boundary;
2. **geometry-effective latitude**: an outer closure clipped to the continuous
   finite overlap of its defining bathymetric west/east traces;
3. **sampled latitude**: the first or last row actually used after mapping the
   geometry to a particular wind/model grid.

For example, the geometry-effective southern Atlantic-Pacific limit cannot
precede `max(first_valid(atlantic_west), first_valid(pacific_east))`. Requested
and geometry-effective limits are preserved, and a shift beyond a reviewed
tolerance is an error. Internal gateways `y_P` and `y_I` remain at their
requested physical latitudes; they are never inferred from trace coverage.

## 4. Coordinate conventions

### 4.1 Pixel centres

GEBCO coordinates locate cell centres. Algorithms must not treat them as cell
edges. The crossing between a shallow and deep pixel is interpolated between
their centre coordinates and elevations.

### 4.2 Cyclic longitude

The source is cyclic but contains no duplicated `-180/180` endpoint. Connected
component labelling and neighbourhood operations must explicitly connect the
first and last longitude columns.

Calculations use continuous longitude branches selected for each physical
ocean:

| Ocean | Reference centre | Initial search window |
|---|---:|---:|
| Atlantic | 0 degrees | -100 to 40 degrees east |
| Indian | 75 degrees | 20 to 150 degrees east |
| Pacific | 180 degrees | 105 to 290 degrees east |

The exact windows are configuration, not constants buried in extraction code.
Pacific longitudes remain unwrapped through candidate tracking, interpolation,
regularization, polygon assembly, and validation. Wrapped longitudes are
display-only auxiliary coordinates. Smoothing a Pacific trace after wrapping
it to `[-180,180)` is prohibited.

### 4.3 Metric calculations

Latitude and longitude are exposed in degrees, but distances and areas use a
spherical or declared geodesic metric:

$$dy=R\,d\phi,\qquad dx=R\cos\phi\,d\lambda.$$

Component-area thresholds use square kilometres, not a fixed number of cells
whose physical area varies with latitude.

### 4.4 Physical and sampled boundaries

`MultiBasinGeometry` stores requested and geometry-effective continuous
latitudes. Geometry-effective coverage is defined from the source bathymetric
west/east traces. A resolved `x_b` must cover that whole interval; a shorter
`x_b` is a configuration error rather than a reason to clip it again.

A later `GeometrySampler` maps the effective geometry to a particular
wind/model grid. It uses the first/last included row for a sampled outer
boundary or closed-basin taper and records the source-grid identity; it does
not assume an exact coordinate match. Sampled latitudes belong to that adapter
result, not to the source `MultiBasinGeometry`.

An internal shared gateway remains one physical interface for its parent and
children. Sampling either maps all participants to the same row or fails with
the coordinate mismatch. This is deliberately a small adapter rule, not a
second geometry-extraction algorithm.

## 5. Versioned extraction configuration

Every run consumes a serializable, versioned configuration containing at
least:

- GEBCO identity and expected checksum;
- target isobath depth;
- working-grid coarsening rule;
- ocean longitude frames and search windows;
- six trace domains and all six requested latitudes `y_S`, `y_P`, `y_I`,
  `y_N`, `y_NI`, and `y_NP`;
- deep-ocean anchor points and basin-specific partition/search corridors;
- minimum shallow-barrier and deep-component physical areas;
- named topology overrides, each with an explicit `force_barrier`,
  `force_deep`, or `exclude_barrier_component` operation;
- named ignored-island components or boxes, initially including Madagascar
  and New Zealand;
- interval-tracking weights and hard constraints;
- crossing-refinement window;
- gap, outlier, displacement, and smoothing tolerances;
- requested-to-effective boundary-shift tolerance;
- `x_b` policy or explicit WBC-width prescription;
- output and QA thresholds.

The normalized configuration has a cryptographic hash stored in every output.
Manual changes to a bridge, anchor, domain, or tolerance therefore produce a
different geometry identity.

## 6. Extraction algorithm

### Stage 1: validate and subset the source

Validate the source contract before reading bathymetry. Select only the
latitude band required by all traces plus a declared halo for connectivity;
the prototype uses approximately `-60.5` to `72.5` degrees north. Record the
actual source slices.

The extraction halo south of `y_S` is available only for local interpolation
and endpoint QA. Before connected-component analysis, mask everything south of
the requested physical `y_S`: the theoretical ocean ends at the southern tip
of South America and treats that section as an external boundary with
prescribed inflow. No path around Antarctica may connect the model geometry.

Convert positive-up elevation `z` to positive-down ocean depth

$$d=-z.$$

Land remains non-ocean and is not confused with shallow ocean simply because
both satisfy a numerical threshold.

### Stage 2: build a deterministic working grid

Block-average the native 15 arc-second grid by a configured integer factor;
the current prototype uses `4x4`, producing a one-arc-minute working grid.
Record whether partial coastal blocks are included and how land/ocean mixtures
are reduced. The reduction tree, chunking, numerical dtype, and dependency
versions are fixed in provenance. Canonical scientific arrays must be exactly
reproducible in the pinned reference environment; other supported environments
must meet declared numerical tolerances. NetCDF file bytes are not required to
match across compression libraries or metadata serializers.

Native-resolution tiles remain available for Stage 8 refinement and QA. The
working grid is a search accelerator, not the final authority for an isobath
crossing.

### Stage 3: create the unmodified shallow/deep diagnostic

For target depth `D`, derive explicit masks for land, shallow ocean, and deep
ocean. A conceptual barrier is land plus ocean shallower than `D`; the selected
basin interior is connected deep ocean.

Preserve this raw classification as a diagnostic before any manual topology
operation. The QA report must make it possible to distinguish a natural GEBCO
barrier from one created by configuration.

### Stage 4: apply named topology overrides

The non-ITF ocean topology requires a small number of scientific choices where
a depth contour alone does not express the intended basin closure. Each choice
is an explicit named operation with a polyline or component rule, width,
rationale, affected mask, and citation/provenance. Every override declares its
mask semantics: `force_barrier`, `force_deep`, or
`exclude_barrier_component`. An implementation may not infer the operation
from the override's name.

The initial configuration is expected to include:

- `greenland_iceland_uk_europe_bridge`, forcing a narrow shallow barrier that
  joins the intended shelf/continental components where required;
- `southeast_asia_australia_closure`, closing Indonesian Throughflow for the
  non-ITF topology with a narrow shelf-following route;
- `bering_strait_closure`;
- `caribbean_bahamas_attachment`, when the Caribbean arc is intended to join
  the American shelf.

Before interval selection, label shallow/barrier components and remove named
ignored-island components from the selection mask while retaining them in the
raw diagnostic. At minimum, Madagascar and New Zealand are explicit initial
exclusions. This is the production counterpart of the prototype's
`_large_barrier_mask` and `EXCLUDED_ISLAND_BOXES`; merely stating that islands
are ignored is not sufficient.

Overrides must be narrow and shelf-following. Large rectangular dams are
prohibited. The report gives the area and length changed by every operation and
plots raw and modified masks separately.

### Stage 5: identify the global interior and basin partitions

Label deep water using cyclic eight-connectivity after truncating at `y_S` and
applying the non-ITF overrides and ignored-island filtering. Select the
intended model-domain deep-ocean component using multiple stable anchors;
every anchor must be deep at the target isobath and belong to that component.

This is not a model of the ocean south of South America. Within the retained
domain, however, sectors 5 (Atlantic-Pacific) and 4 (Indo-Atlantic) are
intentionally composite connected regions. The three named physical oceans
therefore are views/partitions of the truncated model domain, not three
independent global connected-component labels.

Within the global component, explicit basin partition masks, search corridors,
and gateway cross-sections define where Atlantic, Indian, and Pacific boundary
candidates may be sought. Multiple anchors per partition guard against a
marginal sea or a wrong contour branch. Save the global component and all
partition/corridor masks for QA.

### Stage 6: enumerate candidate intervals by latitude

At every working-grid latitude and within each basin's unwrapped partition and
search corridor, enumerate contiguous intervals belonging to the selected
global deep component. Ignored barrier components cannot split an interval.
For each candidate record:

- western/eastern cell indices and unwrapped longitudes;
- width and physical area contribution;
- overlap with intervals on adjacent rows;
- distance to basin anchors and expected boundary corridor;
- whether either endpoint touches a search-window edge;
- nearby branches or islands that could cause switching.

No longitude-window endpoint is silently accepted as a shelf crossing.

### Stage 7: choose one continuous interval sequence

Choose one interval per latitude with a deterministic continuity-aware
tracker, not independent row-wise `min/max` selection. A dynamic-programming
or equivalent method may minimize a declared cost built from:

- geodesic displacement of west and east endpoints;
- loss of overlap with the previous interval;
- implausible width changes;
- departure from configured boundary corridors;
- search-window-edge contact;
- branch switches not supported by topology configuration.

Hard constraints enforce partition/corridor membership, anchor containment,
and trace domains. The algorithm reports the selected path and low-confidence
rows. Runner-up costs and reverse-order sensitivity are recommended QA, not
part of the minimal `MultiBasinGeometry` contract.

### Stage 8: locate and refine true isobath crossings

For each selected interval edge, interpolate the target-depth crossing between
the adjacent shallow and deep working-grid cell centres. Then reopen a small
native 15 arc-second GEBCO window around that estimate and refine the crossing
on the native grid. Native refinement has a configured maximum search distance;
exceeding it is a branch-identity failure, not permission to select a remote
crossing.

At an exact requested gateway latitude that is not a native pixel-centre row,
interpolate native bathymetry in latitude first and then solve for the zonal
target-depth crossing. Do not obtain a gateway by merely interpolating between
two already extracted trace longitudes, which need not remain on the target
isobath.

Every result receives a status:

- `native_crossing`;
- `working_grid_crossing` if native refinement is impossible but the coarse
  crossing is valid;
- `configured_endpoint` for an explicitly reviewed domain boundary;
- `topology_override`, carrying the override ID and source geometry for a
  boundary segment created by a reviewed bridge or closure;
- `short_gap_fill` for a later permitted repair;
- `failure`.

An unconfigured search-window edge is always `failure`. Raw longitude, final
longitude, adjacent source depths, independently evaluated native-grid depth,
override ID, and status are retained.

### Stage 9: enforce independent trace domains

Apply the reviewed domains of all six traces independently. Junctions such as
`y_S`, `y_P`, and `y_I` are represented exactly on the geometry output
latitude grid using the bathymetric interpolation-and-recrossing rule above.
Region geometry-effective outer limits are then obtained from the finite
overlap of the bathymetric traces that define that region, with requested and
effective values named separately.

An explicit `in_domain` mask distinguishes designed out-of-domain missing
values from extraction failure. Every sample inside a final effective region
must be finite. A permitted short gap is filled and labelled `short_gap_fill`;
an unfilled in-domain `NaN` rejects the geometry regardless of gap length.

### Stage 10: conservative regularization

Raw crossings are the starting point for any repair:

1. detect endpoint jumps using geodesic distance and robust local statistics;
2. fill only internal gaps shorter than the configured maximum;
3. optionally apply a robust median/Gaussian or spline smoother on the
   continuous longitude branch;
4. cap pointwise displacement from raw crossings;
5. snap every smoothed natural-contour point back to the nearest valid
   target-depth crossing within the configured native search radius.

Regularization must not make a visually smooth curve cease to be the requested
isobath. A `topology_override` segment remains on its reviewed override
geometry and is exempt only from isobath snap-back; its length, displacement,
and affected area are gated separately. Long gaps, persistent branch
ambiguity, or excessive displacement are failures requiring configuration
review, not invitations to interpolate.

Both raw and final traces remain in the output.

### Stage 11: construct shared `x_b` traces explicitly

Bathymetry produces `x_w`, the western isobath/shelf trace. Model dynamics use
`x_b`, the offshore edge of the unresolved WBC region. `MultiBasinGeometry`
therefore requires one of:

1. an explicit `x_b(y)` trace with provenance;
2. a WBC-width prescription converted to longitude using local spherical
   geometry;
3. a named `thin_wbc` approximation setting `x_b=x_w`.

Option 3 is acceptable only when metadata states the approximation. APIs,
variables, and documentation retain both names; they never redefine `x_w` as
`x_b`.

There is one shared `x_b` per physical western boundary, not one independent
copy per region:

- Atlantic `x_b` is reused by regions 1, 4, and 5 and the Atlantic-only view;
- Indian `x_b` is used by region 2;
- Pacific `x_b` is used by region 3.

In each continuous longitude frame, validation requires

$$x_w\le x_b < x_e,$$

with equality `x_b=x_w` allowed only for named `thin_wbc`. A non-thin
prescription also has reviewed minimum and maximum WBC-width bounds and must
cover the full geometry-effective latitude domain. Smoothness checks on `x_b`
are separate from isobath-adherence checks on `x_w`.

### Stage 12: assemble region views and optional polygons

Construct the five normative region views from references to the six final
bathymetric traces and the shared `x_b` traces. Validate positive local width,
the exact mappings in Section 3, and intended interface adjacency at `y_P` and
`y_I`.

Polygon generation is recommended for spatial QA but is not required by the
numerical `MultiBasinGeometry` interface. If produced, northern and southern
edges are explicit sections at configured boundaries; polygons use continuous
longitudes and must be closed, correctly oriented, non-self-intersecting, and
free of unintended overlap. The Atlantic, Indian, and Pacific views reference
the same regions and interfaces rather than creating divergent boundaries.

### Stage 13: create the Atlantic-only view

`MultiBasinGeometry.atlantic_only()` returns the Atlantic region from `y_I` to
`y_N` by reference to the same Atlantic trace objects and `x_b` policy. It is
not a second extraction run. Source identifiers, raw/final crossings,
configuration hash, and provenance remain identical to the global geometry.

## 7. Proposed geometry objects

### `BoundaryTrace`

A validated, immutable trace contains:

- key and physical-ocean role;
- latitude and continuous-longitude arrays;
- raw and final values;
- target isobath, crossing status, and topology-override ID where applicable;
- source-depth residuals and confidence flags;
- `in_domain` and `valid` masks plus configured endpoints;
- coordinate frame and display wrapping rule;
- source/configuration/provenance hashes.

### `RegionGeometry`

A region contains references to west, shared WBC-edge, and east traces;
requested and geometry-effective north/south latitudes; derived metric
widths/areas; optional polygon geometry; and the IDs of adjacent interfaces.
It owns no forcing or physics parameters.

### `MultiBasinGeometry`

The global object contains:

- the six unique source bathymetric traces;
- derived dynamical `x_b` traces when `thin_wbc` is not used;
- the five region views;
- requested and geometry-effective junction/closure latitudes;
- shared-boundary and region-to-trace mappings;
- directed-interface geometry needed by the fixed topology;
- Atlantic/Indian/Pacific composite views;
- source and configuration provenance.

It rejects duplicate copies of a supposedly shared trace. Shared geometry is
represented by object identity or an immutable trace key, making accidental
divergence structurally difficult.

### `GeometrySampler`

The sampler returns a grid-specific view containing sampled boundary
latitudes, the source grid identity, and mappings back to requested and
geometry-effective latitudes. It does not mutate or serialize those sampled
values as if they were properties of the source geometry.

## 8. Output contract

### 8.1 Primary NetCDF

The primary CF-oriented NetCDF contains at least:

```text
coordinates:
    trace
    wbc_trace
    region
    latitude

variables:
    longitude_raw(trace, latitude)
    longitude(trace, latitude)
    longitude_wrapped(trace, latitude)       # display only
    crossing_status(trace, latitude)
    topology_override_id(trace, latitude)
    crossing_confidence(trace, latitude)
    depth_residual(trace, latitude)
    raw_to_final_displacement(trace, latitude)
    in_domain(trace, latitude)
    valid(trace, latitude)
    region_west_trace(region)
    region_east_trace(region)
    region_requested_southern_latitude(region)
    region_requested_northern_latitude(region)
    region_effective_southern_latitude(region)
    region_effective_northern_latitude(region)
    region_x_b_trace(region)
    x_b_longitude(wbc_trace, latitude)         # when a policy is resolved
```

A separate sampled-geometry record contains
`region_sampled_southern_latitude`, `region_sampled_northern_latitude`, the
wind/model latitude-grid identity, and the mapping to effective boundaries.

String/status encodings and missing-value conventions are documented. Global
attributes include GEBCO DOI/checksum, target depth, algorithm and schema
versions, Git commit, dependency versions, normalized configuration and hash,
creation time, longitude frames, and `x_b` policy.

### 8.2 Recommended spatial review products

An implementation may emit GeoJSON or GeoPackage layers for:

- raw and final traces;
- topology-override polylines and affected areas;
- selected deep-ocean component masks or outlines;
- five region polygons;
- anchors, failures, short-gap fills, and low-confidence points.

These products are useful for inspection but are not required by the numerical
geometry interface. The NetCDF/configuration pair is the scientific source of
truth.

### 8.3 QA report

The required machine-readable report includes:

- input and output identities;
- counts by crossing status;
- independently evaluated native-depth adherence for natural-crossing points,
  reported separately from topology overrides;
- raw-to-final RMS, p95, and maximum displacement;
- internal gap lengths and repairs;
- area/length changed by each topology override;
- region width extrema and mapping validity;
- canonical scientific-array hashes in the pinned reference environment;
- comparison with previous 500 m and 1000 m products as regression evidence.

The required visual review shows bathymetry, raw/final traces, ignored islands,
topology overrides, status flags, the five region views, and width versus
latitude. Polygon layers, Hausdorff distance, curvature/total-variation
distributions, resolution-sensitivity panels, and legacy-difference panels are
recommended extended QA rather than hard interface requirements.

## 9. Hard acceptance gates

A production geometry is rejected unless all hard gates pass:

1. exactly six source bathymetric traces and five correctly mapped region
   views, plus derived `x_b` traces where required;
2. explicit domains, `y_S < y_P < y_I < y_N`, `y_NI >= y_I`, and
   `y_NP >= y_P`, with finite coverage over every effective interval;
3. no unflagged search-window-edge fallback;
4. every in-domain sample is finite; every repaired gap is labelled and an
   unfilled gap of any length is rejected;
5. continuous Pacific longitudes with no antimeridian jump;
6. `x_w <= x_b < x_e` everywhere, with equality only under `thin_wbc`, and
   every non-thin WBC width lies within reviewed bounds;
7. exact region-to-trace mappings, positive region widths, and one consistent
   physical latitude for every shared gateway;
8. all anchors lie inside the selected global deep component and their named
   basin partition/search corridor;
9. ignored Madagascar/New Zealand fixtures cannot become final outer
   boundaries or cause interval switching;
10. every natural regularized point maps to a target-depth crossing within the
    maximum native search distance, while every non-crossing point carries a
    permitted explicit status and provenance;
11. native-depth adherence meets reviewed tolerances but is not used as proof
    of branch identity; partition, corridor, continuity, and search-distance
    gates provide that proof;
12. raw-to-final pointwise displacement stays below the reviewed limit;
13. every manual topology override declares its mask operation, is named and
    quantified, and all affected trace samples carry its ID;
14. canonical scientific arrays repeat exactly for identical input/config in
    the pinned reference environment; other supported environments pass
    declared numerical tolerances;
15. Atlantic-only traces and shared Atlantic `x_b` are identical to their
    global source objects;
16. requested, geometry-effective, and grid-sampled latitudes are named and
    recorded separately; outer shifts meet tolerance and internal gateways do
    not move;
17. if optional polygons are emitted, they are valid, non-self-intersecting,
    and have only intended overlap/shared boundaries.

Numerical thresholds begin as proposed configuration and require scientific
review against representative regions. A threshold is not loosened merely to
make one run pass; the failure and rationale are reviewed.

## 10. Test strategy

### Unit tests with synthetic grids

Small cyclic bathymetry fixtures test:

- elevation sign and pixel-centre interpolation;
- a contour crossing the dateline;
- multiple islands and marginal-sea branches;
- explicit Madagascar/New Zealand exclusion from outer-boundary selection;
- an open/closed strait changed by a named narrow bridge;
- `force_barrier`, `force_deep`, and `exclude_barrier_component` semantics;
- override-derived trace samples retaining their override ID;
- interval tracking through a temporary split;
- exact gateway recrossing at a latitude between source rows;
- configured endpoints versus illegal window fallbacks;
- short and long internal gaps;
- smoothing and snap-back displacement limits;
- requested versus geometry-effective versus grid-sampled boundaries;
- shared `x_b` identity, `x_w <= x_b < x_e`, WBC-width limits, and thin-WBC
  metadata;
- optional region polygon orientation, overlap, and shared boundaries.

### Integration tests with repository data

Opt-in tests requiring `data/untracked/GEBCO` verify source identity, run
representative latitude bands, and compare canonical array hashes in the
pinned reference environment plus numerical metrics elsewhere. A full global
extraction is a slow pipeline test, not part of every unit-test run.

### Regression is not acceptance

The existing GEBCO 500 m and 1000 m products are valuable comparison baselines,
but agreement with them cannot replace isobath adherence, topology, polygon,
and provenance gates. A legacy product may contain the same branch-switch or
smoothing error the new pipeline is intended to detect.

## 11. Known failure modes and required response

| Failure | Required response |
|---|---|
| Wrong elevation sign or coordinate registration | Stop at source validation |
| Lost cyclic adjacency | Fail dateline/component tests |
| Wrapped Pacific smoothing | Prohibit by data model |
| Anchor on shelf/land | Configuration error with local diagnostic |
| Open ITF/Bering/Caribbean topology | Review named override; do not add a broad dam |
| Artificial bridge too wide | Fail changed-area/width gate |
| Ignored island or marginal sea becomes an outer boundary | Hard failure; revise exclusion, partition, or corridor configuration |
| Coarsening opens or closes a strait | Native-resolution sensitivity failure |
| Search-window edge used as boundary | Hard failure unless configured endpoint |
| Smoothing leaves requested isobath | Snap back or reject |
| Negative/near-zero region width | Hard geometry failure |
| Optional polygon self-intersection or unintended overlap | Reject the spatial QA product; do not publish it |
| Long missing trace section | Require scientific/configuration review |
| Changed GEBCO/config without provenance change | Hash/provenance test failure |
| Bathymetric `x_w` presented as physical `x_b` | Schema/metadata validation failure |

## 12. Proposed implementation sequence after approval

1. Define configuration and output schemas plus synthetic fixtures.
2. Implement source validation, cyclic coordinates, and out-of-core working
   grid generation.
3. Implement raw topology masks, ignored-island filtering, named override
   semantics, truncated model-domain component selection, and basin
   partitions/corridors.
4. Implement global interval tracking and native crossing refinement.
5. Implement conservative regularization with snap-back and trace QA.
6. Implement shared `x_b`, region assembly, boundary sampling, and the Atlantic
   view; add optional polygons after the core mappings pass.
7. Implement NetCDF/core-QA writers and full integration tests, followed by
   optional spatial review products.

This is a technical dependency order, not a predetermined branch or PR plan.
The first full GEBCO extraction is not accepted until the hard gates and visual
report have both been reviewed.

## 13. Decisions requested in review

Before implementation, review should settle:

1. initial target isobath depth or depths;
2. exact `y_S`, `y_P`, `y_I`, `y_N`, `y_NI`, and `y_NP` values;
3. approved ocean frames, anchors, and named topology-override polylines;
4. working-grid reduction rule near mixed land/ocean blocks;
5. interval-tracking weights and confidence threshold;
6. maximum short-gap length and regularization displacement;
7. requested-to-effective boundary-shift tolerance, maximum native crossing
   search distance, and depth-adherence gate;
8. initial `x_b` policy and any WBC-width prescription;
9. whether 500 m and 1000 m products are both required from the first
   production implementation;
10. which extended spatial and resolution-sensitivity QA products are required
    for the first accepted extraction.

## 14. Source basis

This specification is based on:

- `atlantic_adjustment/notebooks/extract_isobath_gebco.ipynb`;
- the development history in Codex task
  `019efe23-b1a6-7bc2-a24b-189aee627e09`;
- the changed-isobath robustness findings in Codex task
  `019f6120-a741-7972-80b3-96d2376b09e3`;
- the resulting 500 m and 1000 m GEBCO trace products;
- `atlantic_adjustment/notebooks/global_ocean.ipynb`;
- `atlantic_adjustment/notes/two_basins/build/main.pdf` and its TeX source;
- the GEBCO 2026 grid documentation and terms supplied beside the source data.

The historical notebook is treated as a prototype. Its useful elements—small
shelf-following bridges, separate trace domains, side-sea filtering, robust
outlier handling, and smoothing—are retained only where they satisfy the
stronger crossing, topology, determinism, and provenance contracts above.

The no-ITF derivation and closing graph discussion in the multibasin writeup
are normative for the domain: `y_S` is the external southern boundary at South
America, while sectors 5 and 4 are intentionally composite regions north of
that boundary. GEBCO cells south of `y_S` must not introduce an Antarctic
connectivity path that is absent from the theory.
