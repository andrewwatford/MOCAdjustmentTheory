# Model architecture specification

Status: proposed architecture for review. This document specifies interfaces,
equations, invariants, and acceptance tests; it does not authorize scientific
implementation yet.

## 1. Scope and design stance

The first scientific release will support two deliberate model facades:

1. `GlobalNoITFModel`, representing three physical oceans as the five
   dynamical regions in the non-Indonesian-Throughflow derivation.
2. `AtlanticOnlyModel`, representing the single Atlantic region from Cape
   Agulhas to the northern observing line.

They will share geometry, wind, Fourier, response-kernel, solving, and
diagnostic components. They will not initially expose an unrestricted graph
compiler. The global graph is scientifically fixed enough that accepting
arbitrary topologies would add validation burden before there is a tested use
case. Internal graph objects will nevertheless make the five-region structure
explicit, preserve child order, and leave a path to later generalization.

The central separation is:

```text
source data -> validated geometry and forcings -> frequency operators
            -> model-specific linear system -> labelled solution diagnostics
```

Geometry contains no forcing. Forcing contains no solved state. Model facades
own the model-specific equations and compile them in response to typed forcing
prescriptions. Diagnostics derive from one solved state and the same operators
used in the solve.

## 2. Terminology and sign conventions

These terms are normative.

- A **physical ocean** is the Atlantic, Indian, or Pacific.
- A **region** is one of the five latitude-bounded dynamical domains used by
  the global equations. The global model has three physical oceans but five
  regions.
- `x_w(y)` and `x_e(y)` are bathymetric western and eastern shelf/isobath
  traces.
- `x_b(y)` is the offshore, interior edge of the unresolved western boundary
  current (WBC) region. It is physically distinct from `x_w`, even when a thin
  WBC approximation uses `x_b approximately x_w`.
- `h_e` is the eastern-boundary active-layer thickness anomaly.
- `h_b = h(x_b, y, t)` is the interior thickness immediately outside the WBC
  region, obtained from the Rossby-characteristic solution. It is not the
  thickness at the western wall.
- `h_w` is the thickness at the western boundary inferred from geostrophic
  transport. It is not a "westward-propagating thickness."
- `T` is total northward transport and `T_tilde = T - T_Ek` is non-Ekman
  northward transport.
- Graph arrows and positive connection transports point northward.
- Longitude and latitude are degrees east and degrees north at public
  boundaries. Metric integrals use radians and spherical scale factors.
- The Fourier convention is NumPy's forward convention,
  `a_hat(omega) = integral a(t) exp(-i omega t) dt`; a delay `tau` therefore
  contributes `exp(-i omega tau)`.

The relations

$$
T = \widetilde T + T_{Ek},
\qquad
\widetilde T = \frac{g'H}{f}(h_e-h_w),
\qquad
h_w = h_e-\frac{f\widetilde T}{g'H}
$$

must hold under the documented sign convention.

## 3. The non-ITF five-region model

Let the junction latitudes satisfy

$$y_S < y_P < y_I < y_N.$$

The initial values used in the source work are approximately `y_S=-56`,
`y_P=-44`, `y_I=-35`, and `y_N=55` degrees north, but these are explicit
configuration values rather than hard-coded constants.

| ID | Public key | Latitude domain | Western trace | Eastern trace |
|---:|---|---|---|---|
| 1 | `atlantic_north` | `y_I` to `y_N` | Atlantic west | Atlantic east |
| 2 | `indian_north` | `y_I` to `y_NI` | Indian west | Indian east |
| 3 | `pacific_north` | `y_P` to `y_NP` | Pacific west | Pacific east |
| 4 | `indo_atlantic` | `y_P` to `y_I` | Atlantic west | Indian east |
| 5 | `atlantic_pacific` | `y_S` to `y_P` | Atlantic west | Pacific east |

Regions are views assembled from six shared physical traces, not five
independently extracted contours. `y_NI` and `y_NP` are the northern limits of
the closed Indian and Pacific regions and may differ from `y_N`.

### 3.1 Directed topology

```mermaid
flowchart BT
    S["External south"] --> R5["5: Atlantic-Pacific"]
    R5 -->|"Pacific branch T_P"| R3["3: Pacific north"]
    R5 -->|"Transition T_T"| R4["4: Indo-Atlantic"]
    R4 -->|"Indian branch T_I"| R2["2: Indian north"]
    R4 -->|"Atlantic continuation T_A"| R1["1: Atlantic north"]
    R1 --> N["External north"]
```

The required east-to-west child orders are:

- region 5: `[pacific_north, indo_atlantic]`;
- region 4: `[indian_north, atlantic_north]`.

The first child is the eastern child and shares its eastern-boundary thickness
with its parent. Consequently there are three independent eastern-boundary
unknowns:

- `h_A` for region 1;
- `h_I` shared by regions 2 and 4;
- `h_P` shared by regions 3 and 5.

The decomposed branch transports are

$$
T_I=T_{I,Ek}+\kappa_I(h_I-h_A),
\qquad
T_P=T_{P,Ek}+\kappa_P(h_P-h_I),
$$

where

$$
\kappa_I=\frac{g'H}{f(y_I)},
\qquad
\kappa_P=\frac{g'H}{f(y_P)}.
$$

The signed Coriolis parameter is required; replacing it with `abs(f)` is an
error. `T_A` and `T_T` are auxiliary continuation transports determined by
regional volume budgets.

Region 1 has the prescribed northern transport. Regions 2 and 3 have solid
northern boundaries. The global southern transport is wind-driven in the
total-transport formulation and zero after consistent conversion to the
non-Ekman formulation.

### 3.2 Physical-ocean views

A physical ocean is not always a graph node. The global Atlantic diagnostic is
the ordered path

```text
atlantic_pacific -> indo_atlantic -> atlantic_north
```

with branch jumps at `y_P` and `y_I`. This should be represented by an
`OceanPath` or `CompositeOceanView` that references regions and connections.
It must not duplicate state or interpolate blindly across a gateway. Indian
and Pacific views are defined similarly from the graph.

## 4. Atlantic-only model

The Atlantic-only facade uses one region, normally from `-35` to `55` degrees
north, with Atlantic `x_b(y)` and `x_e(y)` throughout. It has no step changes
and no Indian or Pacific branch transports.

It reuses the same trace objects, physical parameters, wind operator, Fourier
plan, and diagnostics as the global model. It has a distinct southern closure
and therefore a distinct model-specific linear system; it should not be
implemented by constructing a malformed one-node instance of the global
five-region equations.

The public configuration switch should be explicit:

```python
global_model = GlobalNoITFModel(geometry=global_geometry, physics=physics)
atlantic_model = AtlanticOnlyModel(
    geometry=global_geometry.atlantic_only(),
    physics=physics,
)
```

## 5. Public forcing contract

### 5.1 Wind input is wind-stress anomaly

The public wind forcing is the pair of wind-stress anomalies
`tau_x_prime(time, latitude, longitude)` and
`tau_y_prime(time, latitude, longitude)`, not `w_Ek`.

A `WindStressAnomaly` validates:

- a common, strictly increasing time coordinate;
- one-dimensional latitude and cyclic longitude coordinates;
- dimensions and finite values;
- dynamic (`N m-2`) or kinematic (`m2 s-2`) stress units;
- whether and how the time mean was removed;
- provenance, source record, interpolation, and taper configuration.

The normal preparation path computes the anomaly over the exact common model
record before any curl or transport integral. For a SCOTIA/ERA5 experiment,
this means removing the time mean over their common record while retaining the
seasonal cycle. Precomputed `w_Ek` may be accepted only as an internal cache or
advanced low-level operator input; it is not the primary user interface.

The wind operator derives, with one shared regularization,

$$
I_\gamma(f)=\frac{f}{f^2+\gamma^2},
$$

$$
\mathbf M'_{Ek}
=\left(
\frac{\tau'_y I_\gamma}{\rho_0},
-\frac{\tau'_x I_\gamma}{\rho_0}
\right),
\qquad
w'_{Ek}=\nabla\cdot\mathbf M'_{Ek},
$$

and

$$
T'_{Ek}(y)
=-\int_{x_b}^{x_e}\frac{\tau'_x I_\gamma}{\rho_0}\,dx.
$$

Stress is tapered before taking the curl. It goes to zero on solid zonal
boundaries and on the closed northern boundaries of the Indian and Pacific.
The three step-shaped Atlantic regions are differentiated separately so that
topological taper discontinuities do not create artificial wind-curl sheets.

### 5.2 Independent regional forcing

`RegionForcingSet` contains one replaceable wind-stress prescription for each
of the five global regions. A region may:

- use the default geometry-masked view of one global stress field;
- use a separate stress dataset;
- be scaled for a sensitivity experiment;
- be explicitly set to zero.

The absence of a region prescription is an error unless a named default policy
fills it. This prevents an omitted forcing from being silently treated as
zero. Region masks and tapers belong to the wind operator configuration, not
to the raw stress data.

External transport prescriptions are typed as `total` or `non_ekman`. A total
transport is converted using the local Ekman transport derived from the same
wind-stress anomaly field. The conversion and sign are recorded in output
metadata. This is essential for the SCOTIA northern-boundary ambiguity and for
equivalence between the two equation forms below.

## 6. Time and Fourier plan

A `FourierPlan` owns all temporal preprocessing so every forcing uses exactly
the same convention:

1. intersect source records and resample onto one uniform time axis;
2. form anomalies on that common interval;
3. reflect `n-1` samples at each end of an `n`-sample record;
4. choose `n_fft >= 3n-2`, with optional additional zero padding;
5. use `omega = 2*pi*rfftfreq(n_fft, d=dt)`;
6. solve nonnegative frequencies;
7. set the zero-frequency anomaly solution to exactly zero;
8. reconstruct with `irfft`, crop the original central interval, and remove
   only floating-point residual means.

Reflection and cropping prevent the periodic FFT assumption from connecting
the first and last observations directly. The plan records original and
padded indices, cadence, reflection length, `n_fft`, and frequency units.

The Nyquist policy must be explicit. The preferred first implementation uses
an odd padded length, avoiding a standalone Nyquist coefficient. If an even
length is supported, tests must prove that its real-valued coefficient and
operator projection preserve a real inverse transform; it must not be silently
dropped.

## 7. Region response operators

For region `j`, let

$$
L_j(y)=x_e^{(j)}(y)-x_b^{(j)}(y)>0,
$$

and let `c(y)>0` denote the magnitude of westward long-Rossby-wave speed. Define

$$
P_j(\omega,x,y)
=1-e^{-i\omega(x-x_b^{(j)})/c},
$$

$$
F_j=-\iint_{B_j}P_j\widehat w_{Ek,j}\,dA,
\qquad
r_j=-\int_{y_{Sj}}^{y_{Nj}}
cP_j(\omega,x_e^{(j)},y)\,dy.
$$

Each regional budget is

$$
T_{out}^{(j)}-T_{in}^{(j)}=-F_j+r_jh_e^{(j)}.
$$

Units are part of the operator contract:

- `F`: `m3 s-1`;
- `r` and `kappa`: `m2 s-1`;
- `h_e`: `m`;
- `r*h_e` and `kappa*h_e`: `m3 s-1`.

Partial north-of-latitude integrals are computed by the same operator and
cached for transport diagnostics. They are not independently reimplemented in
plotting code.

## 8. Global equation systems

### 8.1 Full F/r form: derivation and acceptance oracle

The seven unknowns are

$$
(h_A,h_I,h_P,T_A,T_T,T_I,T_P).
$$

Five regional budgets and the two branch decompositions give seven equations.
After eliminating transports, the three-thickness system is

$$
\begin{pmatrix}
r_1+\kappa_I&r_4+\kappa_P-\kappa_I&r_5-\kappa_P\\
-\kappa_I&r_2+\kappa_I&0\\
0&-\kappa_P&r_3+\kappa_P
\end{pmatrix}
\begin{pmatrix}h_A\\h_I\\h_P\end{pmatrix}
=
\begin{pmatrix}
F_1+F_4+F_5+T_N+T_{I,Ek}+T_{P,Ek}-T_S\\
F_2-T_{I,Ek}\\
F_3-T_{P,Ek}
\end{pmatrix}.
$$

The implementation must reproduce this matrix up to a declared permutation.
It is the independent acceptance oracle even if the production solver uses the
more compact form below.

### 8.2 Q/K form: recommended production system

Define

$$
K_j(y,\omega)=c(y)\left[1-e^{-i\omega L_j(y)/c(y)}\right],
$$

$$
p_j(y,\omega)=\int\widehat w_{Ek,j}\,dx,
\qquad
q_j(y,\omega)=\int\widehat w_{Ek,j}
e^{-i\omega(x-x_b)/c}\,dx.
$$

When local Ekman transports and pumping derive from the same stress anomaly,
the explicit `p` terms cancel in the non-Ekman transport system. In unknown
order `(h_P, h_I, h_A)`, the production matrix is

$$
\begin{pmatrix}
\kappa_P-\int_3K_Pdy&-\kappa_P&0\\
0&\kappa_I-\int_2K_Idy&-\kappa_I\\
-\kappa_P-\int_5K_Ady&
\kappa_P-\kappa_I-\int_4K_Ady&
\kappa_I-\int_1K_Ady
\end{pmatrix}
\begin{pmatrix}h_P\\h_I\\h_A\end{pmatrix}
=
\begin{pmatrix}
Q_P\\Q_I\\\widetilde T_N+Q_A
\end{pmatrix},
$$

where

$$
Q_P=\int_3q_Pdy,
\quad
Q_I=\int_2q_Idy,
\quad
Q_A=\int_{1\cup4\cup5}q_Ady.
$$

The first implementation should solve this explicit three-by-three system per
frequency, report rank and condition number, and reconstruct all auxiliary
graph transports afterward. A generic equation compiler is unnecessary for
the initial fixed global model.

### 8.3 Atlantic-only system

For total northern transport,

$$
\left[
\frac{g'H}{f_S}-\int_{y_S}^{y_N}K\,dy
\right]\widehat h_e
=\widehat T_N-\widehat T_{Ek,S}
-\int_{y_S}^{y_N}(\widehat p-\widehat q)\,dy.
$$

Using non-Ekman northern transport gives the equivalent form

$$
\left[
\frac{g'H}{f_S}-\int K\,dy
\right]\widehat h_e
=\widehat{\widetilde T}_N+\int\widehat q\,dy.
$$

The Atlantic facade uses this scalar operator, rather than routing through the
global three-thickness matrix.

## 9. Diagnostics and solution contract

The interior characteristic solution is

$$
\widehat h(x,y)
=\widehat h_e e^{i\omega(x-x_e)/c}
+\int_{x_e}^{x}\frac{\widehat w_{Ek}(x')}{c}
e^{i\omega(x-x')/c}\,dx'.
$$

At the interior edge of the WBC region,

$$
\widehat h_b
=\widehat h_e e^{-i\omega L/c}-\frac{\widehat q}{c}.
$$

For the global Atlantic path, the applicable eastern thickness is

$$
h_e^*(y)=
\begin{cases}
h_A,&y\ge y_I,\\
h_I,&y_P\le y<y_I,\\
h_P,&y<y_P.
\end{cases}
$$

The path's non-Ekman transport is

$$
\widetilde T_A(y)
=\widetilde T_N
+\int_y^{y_N}\left[q_A+K_Ah_e^*\right]dy'
+\mathbf 1_{y<y_I}T_{I,g}
+\mathbf 1_{y<y_P}T_{P,g},
$$

where

$$
T_{I,g}=\kappa_I(h_I-h_A),
\qquad
T_{P,g}=\kappa_P(h_P-h_I).
$$

`AdjustmentSolution` returns labelled `xarray` objects containing at least:

- `h_e(time, independent_boundary)`;
- `edge_transport(time, connection)`;
- `h_b(time, region, latitude)`;
- `h_w(time, ocean_view, latitude)`;
- `transport_non_ekman(time, ocean_view, latitude)`;
- `transport_ekman(time, ocean_view, latitude)`;
- `transport_total(time, ocean_view, latitude)`;
- frequency, rank, condition number, and residual diagnostics;
- named forcing contributions when a decomposition solve is requested.

Results record physical parameters, geometry/configuration hashes, source
provenance, anomaly period, Fourier plan, regularization, tapering, units, and
sign conventions. A forcing-decomposition result must close exactly: the sum
of its named linear contributions equals the full solution to numerical
tolerance.

## 10. Proposed components

| Component | Responsibility |
|---|---|
| `PhysicalParameters` | `g_prime`, `H`, `rho0`, Earth/rotation constants, Rossby cap, Ekman regularization |
| `TimeAxis` | Common uniform analysis interval and units |
| `FourierPlan` | Anomalies, reflection, padding, frequencies, crop, DC/Nyquist policy |
| `BoundaryTrace` | One validated unwrapped longitude function and provenance |
| `RegionGeometry` | Latitude domain plus references to `x_w`, `x_b`, and `x_e` |
| `MultiBasinGeometry` | Six traces, five region views, interfaces, polygons, physical-ocean views |
| `DirectedRegionTopology` | Fixed non-ITF graph, edge directions, degrees, child order, boundary-sharing groups |
| `WindStressAnomaly` | Typed `tau_x_prime` and `tau_y_prime` source field |
| `RegionForcingSet` | Independent regional stresses plus typed external transports |
| `WindResponseOperator` | Taper, regularized Ekman transport, curl, `w_Ek`, `p`, `q`, `F` |
| `RegionFrequencyOperator` | `K`, `r`, partial integrals, characteristic `h_b` coefficients |
| `GlobalNoITFModel` | Assemble and solve the fixed global system, reconstruct graph state |
| `AtlanticOnlyModel` | Assemble and solve the one-region southern closure |
| `CompositeOceanView` | Stitch region diagnostics across junctions with branch jumps |
| `AdjustmentSolution` | Labelled state, residuals, transport/thickness diagnostics, provenance |

All configuration objects are immutable after validation. Arrays are either
owned read-only NumPy values or labelled xarray objects with explicit copy/view
semantics. Model instances contain configuration and reusable operators, not
mutable experiment results.

## 11. Error and validation policy

Configuration errors fail before any expensive transform. They include:

- missing or duplicate region keys/connections;
- a graph other than the exact non-ITF preset passed to the global facade;
- incomplete or non-unique child order;
- geometry/topology interface disagreement;
- missing regional wind forcing without an explicit default;
- mixing total and non-Ekman transport without a conversion field;
- inconsistent time axes, units, anomaly intervals, or stress conventions;
- use of `x_w` as `x_b` without explicit approximation metadata;
- non-finite inputs or non-positive region width.

Frequency failures report the frequency/period, matrix rank, condition number,
equation labels, and scaled residual. The solver never silently fills an
underdetermined bin, drops a Nyquist bin, or changes a forcing component.

## 12. Acceptance tests required before scientific release

### Geometry and topology

- Exactly six physical traces and five non-ITF regions.
- `y_S < y_P < y_I < y_N`; all region domains are finite.
- Exact graph degrees, acyclic order, child order, and boundary-sharing groups.
- Unwrapped `x_e > x_b` throughout every region.
- `x_b` remains semantically distinct from `x_w`.

### Wind and forcing

- Both stress-anomaly components are required.
- Explicit anomaly mean is zero over the common record at every grid point.
- Tapering precedes curl; closed-boundary and region-seam tests detect leakage.
- Pumping and local Ekman transport use the same regularization and source
  stress.
- Independent zero/replace/scale tests for all five regional forcings.
- Total-to-non-Ekman conversion closes at every prescribed section.

### Frequency operators and solvers

- FFT/reflection/crop round trip and no circular end-to-start arrival.
- Zero forcing gives an exactly zero anomaly solution.
- Analytic sinusoidal-delay phase test.
- Global assembled matrix matches the `F/r` matrix above up to a declared
  permutation.
- Production `Q/K` and oracle `F/r` formulations agree when driven by the same
  wind stress.
- All nonzero bins have expected rank; conditioning and residuals are exposed.
- Inverse transforms are real to tolerance and DC is exactly zero.
- Linearity and superposition hold for every named forcing contribution.

### Diagnostics

- Every regional volume budget closes in frequency and time domains.
- Internal connection transports are conservative under graph orientation.
- `T = T_tilde + T_Ek` and
  `T_tilde = g_prime*H/f*(h_e-h_w)` away from the regularized equatorial
  singularity.
- `h_b` equals the characteristic solution evaluated at `x_b`.
- Global Atlantic path has the correct branch jumps at `y_I` and `y_P`.
- Atlantic-only northern reconstruction and southern closure match their
  prescribed transports.

## 13. Implementation sequence after specification approval

1. Implement value objects and validation only: parameters, time, traces,
   regions, fixed topology, and typed forcing schemas.
2. Implement wind-stress anomaly preparation and unit-tested spherical Ekman
   operators.
3. Implement `FourierPlan` and isolated region response kernels.
4. Implement the Atlantic-only scalar solver and analytic benchmark tests.
5. Implement the explicit non-ITF global matrix and `F/r` oracle.
6. Implement labelled diagnostics and physical-ocean views.
7. Add real-data integration workflows and worked examples only after all
   synthetic acceptance tests pass.

Each stage should be its own reviewed PR. Real-data examples must not become
the primary proof of equation correctness.

## 14. Decisions requested in review

The architecture can proceed with defaults, but the following scientific
choices should be explicitly approved before implementation:

1. Whether `x_b approximately x_w` is accepted initially, and how WBC width is
   later prescribed.
2. Default `H`, `g_prime`, isobath depth, and whether global and Atlantic-only
   defaults differ. Existing notebooks use both 500 m and 1000 m choices.
3. Whether public northern transport defaults to `total` or `non_ekman` for
   SCOTIA experiments. Both remain supported and explicitly typed.
4. The default equatorial regularization scale and stress-taper widths.
5. Exact junction and closed-basin northern latitudes.
6. Whether the first global implementation should expose only the fixed
   non-ITF preset, as recommended here, or also an experimental arbitrary-graph
   API.

## 15. Source basis

This specification reconciles:

- `atlantic_adjustment/notebooks/global_ocean.ipynb`;
- `atlantic_adjustment/notebooks/frequency_space.ipynb`;
- `atlantic_adjustment/notes/two_basins/build/main.pdf` and its TeX source;
- the later global/Atlantic transport and comparison notebooks;
- the existing single-basin source as historical behavior, not as a required
  API.

Where sources conflict, this document uses the non-ITF graph, wind-stress
anomalies as the public forcing, `h_b` at the interior edge of the WBC region,
and `h_w` as the geostrophic western-boundary diagnostic. The later ITF graph
in the two-basin note and interface mockup is explicitly outside the initial
scope.
