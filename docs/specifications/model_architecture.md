# Global adjustment model specification

## 1. Scope

This package implements one model: `GlobalAdjustmentModel`. Its purpose is to
solve the linear, five-region adjustment problem described in the multi-basin
theory and return the complete time-dependent solution.

The user workflow is deliberately short:

1. construct a `MultiBasinGeometry`;
2. construct one `GlobalForcing` from domain-wide wind-stress anomalies and
   northern and southern boundary transports;
3. give both objects to `GlobalAdjustmentModel` and solve the
   \(3\times3\times n_\omega\) frequency-domain system; and
4. inspect a `GlobalAdjustmentOutput` containing \(h_e\), \(h_b\), \(h_w\), and
   transport for every region.

There is no separate single-basin model, Atlantic model, equation compiler,
port abstraction, or public Fourier-plan object. Atlantic-only diagnostics are
selections from the global output, not a second solver.

## 2. Public objects

The intended public surface is:

```python
MultiBasinGeometry
GlobalForcing
GlobalAdjustmentModel
GlobalAdjustmentOutput
```

`Basin` may be retained as an internal or convenience value object if it makes
geometry construction clearer. Users do not need to instantiate one in order
to run the model.

All labelled arrays should use `xarray`. Large wind fields and geometry masks
may be backed by `dask`; the final \(3\times3\) solves are small dense linear
algebra and do not need distributed machinery.

## 3. Multi-basin geometry

`MultiBasinGeometry` is the single description of the domain. It contains both
the geometric information required for integrations and the directed,
ordered topology required for the budgets.

The five regions are:

| Label | Region | Meridional range | Western / eastern boundary |
|---|---|---|---|
| 1 | North Atlantic | \(y_I\) to \(y_N\) | Atlantic / Atlantic |
| 2 | North Indian | \(y_I\) to \(y_{N,I}\) | Indian / Indian |
| 3 | North Pacific | \(y_P\) to \(y_{N,P}\) | Pacific / Pacific |
| 4 | Atlantic–Indian transition | \(y_P\) to \(y_I\) | Atlantic / Indian |
| 5 | Atlantic–Pacific transition | \(y_S\) to \(y_P\) | Atlantic / Pacific |

The theory stops at \(y_S\), near the southern tip of South America. It does
not model the Southern Ocean as an additional connected basin.

The ordered children are fixed by the derivation:

```text
region 5 -> [region 3, region 4]
region 4 -> [region 2, region 1]
```

Changing child order changes the algebra and must therefore be explicit and
validated. The geometry also records boundary curves, latitude coordinates,
region masks, integration weights, and the locations \(x_e(y)\) and \(x_b(y)\).
Here \(x_b\) is just outside the western boundary-current region; it is not the
western coastline.

The eastern-boundary thickness has only three independent values:

```text
h_A : region 1
h_I : regions 2 and 4
h_P : regions 3 and 5
```

The geometry owns that sharing map so downstream results can still be exposed
on all five regions.

## 4. One forcing object

`GlobalForcing` accepts three domain-wide inputs on a common time record:

- wind-stress anomalies \(\tau'_x(x,y,t)\) and \(\tau'_y(x,y,t)\);
- northern total transport \(T_N(t)\); and
- southern total transport \(T_S(t)\).

Only the total transports at the external northern and southern closures are
supplied. Ekman transports at internal sections, including \(T_{I,\mathrm{Ek}}\)
and \(T_{P,\mathrm{Ek}}\), are never independent inputs. They are derived from
the same wind stress used to calculate Ekman pumping. This prevents a regional
transport prescription from contradicting the area-integrated wind forcing.

The forcing constructor is responsible for:

1. aligning the time axes and checking units;
2. forming or validating anomalies;
3. applying any declared detrending, gap filling, or tapering;
4. padding the common record;
5. applying `rfft` to every input with one convention; and
6. retaining everything needed to apply the matching `irfft` and crop.

The original time-domain inputs and preprocessing metadata remain available
for provenance.

### 4.1 Ekman quantities

The wind is converted to kinematic stress with one density \(\rho_0\). Near the
equator the inverse Coriolis parameter is regularized consistently, for
example with

\[
I_\gamma(f)=\frac{f}{f^2+\gamma^2}.
\]

The Ekman transport and pumping are then

\[
\mathbf M_{\mathrm{Ek}}
=
\left(
\frac{\tau'_y}{\rho_0}I_\gamma,
-\frac{\tau'_x}{\rho_0}I_\gamma
\right),
\qquad
w_{\mathrm{Ek}}=\nabla\!\cdot\mathbf M_{\mathrm{Ek}}.
\]

The model samples this one derived field against the geometry to obtain every
regional pumping integral and every Ekman section transport. Dask-backed
differentiation and reductions may be used, but the resulting regional arrays
should be small and eagerly validated before the solve.

### 4.2 Cheap compatibility diagnostic

For each region the implementation may expose

\[
\epsilon_j(t)=
\iint_{R_j}w_{\mathrm{Ek}}\,dA
-\left(\sum_{\rm out}T_{\mathrm{Ek}}
-\sum_{\rm in}T_{\mathrm{Ek}}\right).
\]

This is a diagnostic, not another forcing constraint. It should reuse the
already-computed pumping and section transports, so calculating it requires
only regional reductions and no additional derivatives, transforms, or
solves. Small residuals can arise from masks, discretization, and the
regularized equatorial band. No separate lateral shelf flux is prescribed.

## 5. Fourier contract

`GlobalForcing` owns the transform convention. A model must use the supplied
frequency coordinate and must return through the forcing object's inverse
transform; it must not construct a competing frequency grid.

The default convention is:

- a common, uniformly sampled time coordinate;
- NumPy's real-input `rfft` convention,
  \(\widehat a(\omega)=\int a(t)e^{-i\omega t}\,dt\), so a delay contributes
  \(e^{-i\omega\tau}\);
- non-negative angular frequency
  \(\omega=2\pi\,\mathrm{rfftfreq}(n_{\rm fft},\Delta t)\);
- zero padding sufficient to isolate the retained interval from circular
  wraparound (normally at least \(n-1\) samples on each side);
- a zero anomaly at the zero-frequency bin; and
- matching `irfft`, crop, coordinates, and units on output.

Padding length, crop slices, normalization, treatment of the Nyquist bin, and
the anomaly reference are immutable metadata on `GlobalForcing`. If a model
needs a stricter convention it may reject the forcing with a clear validation
error; it may not silently reinterpret it.

## 6. GlobalAdjustmentModel

`GlobalAdjustmentModel` ingests one geometry, one forcing, and the physical
parameters. It derives all geometry-dependent forcing terms, assembles the
fixed global system at every frequency, solves it, reconstructs the complete
diagnostics, and applies the forcing-owned inverse transform.

### 6.1 Propagation and regional terms

Within a region the eastern-boundary signal obeys

\[
\partial_t h-c(y)\,\partial_x h=-w_{\mathrm{Ek}},
\qquad
c(y)=\frac{\beta g'H}{f^2},
\]

with any low-latitude cap on \(c\) recorded as model metadata. Define

\[
P_j(\omega,x,y)
=\exp\!\left[
\frac{i\omega\,[x_b^{(j)}(y)-x]}{c(y)}
\right]-1.
\]

This \(P_j\) is the regional budget kernel, with its sign chosen to absorb
the leading signs in the regional forcing and storage coefficient:

\[
F_j(\omega)=
\int_{y_{S,j}}^{y_{N,j}}
\int_{x_b^{(j)}(y)}^{x_e^{(j)}(y)}
P_j(\omega,x,y)\,
\widehat w_{\mathrm{Ek},j}(x,y,\omega)\,dx\,dy,
\]

\[
r_j(\omega)=
\int_{y_{S,j}}^{y_{N,j}}
c(y)P_j\!\left(\omega,x_e^{(j)}(y),y\right)\,dy.
\]

These definitions give the thin-western-boundary-current regional budget

\[
T_{\mathrm{out}}^{(j)}-T_{\mathrm{in}}^{(j)}
=-F_j+r_jh_e^{(j)}.
\]

These are implementation details of the global model, not user-supplied
regional forcing objects.

### 6.2 Branch transports

The internal branch transports are

\[
T_I=T_{I,\mathrm{Ek}}+\kappa_I(h_I-h_A),
\qquad
T_P=T_{P,\mathrm{Ek}}+\kappa_P(h_P-h_I),
\]

where

\[
\kappa=\frac{g'H}{f}
\]

is evaluated with the signed Coriolis parameter at the relevant section.

### 6.3 The only prognostic solve

At each \(\omega\), solve

\[
\begin{bmatrix}
r_1+\kappa_I & r_4+\kappa_P-\kappa_I & r_5-\kappa_P \\
-\kappa_I & r_2+\kappa_I & 0 \\
0 & -\kappa_P & r_3+\kappa_P
\end{bmatrix}
\begin{bmatrix}h_A\\h_I\\h_P\end{bmatrix}
=
\begin{bmatrix}
F_1+F_4+F_5+T_N+T_{I,\mathrm{Ek}}+T_{P,\mathrm{Ek}}-T_S\\
F_2-T_{I,\mathrm{Ek}}\\
F_3-T_{P,\mathrm{Ek}}
\end{bmatrix}.
\]

The implementation vectorizes this as a stack of \(3\times3\) systems over
`omega`. It records condition numbers and raises or warns according to a
documented threshold. There is no generic equation compiler.

## 7. Complete output

Every call to `solve()` returns a `GlobalAdjustmentOutput`. It always contains
time-domain values for all five regions:

- `h_e(time, region)` — eastern-boundary thickness, with shared values repeated
  according to the geometry map;
- `h_b(time, region, latitude)` — thickness at \(x_b(y)\), outside the western
  boundary-current region;
- `h_w(time, region, latitude)` — western-boundary thickness inferred from the
  geostrophic transport relation; and
- `transport(time, region, latitude)` — total meridional transport.

Transport is also returned as `transport_ekman` and
`transport_geostrophic`. These components are inexpensive once the wind and
thickness solution are known, so they are computed in the same solve rather
than behind a second model run.

The notation is physical: \(h_w\) means western-boundary thickness, not
westward-propagating thickness.

For a latitude \(y\) inside region \(j\), the characteristic solution used for
\(h_b\) is kept distinct from the budget kernel:

\[
\widehat h(x,y,\omega)
=\widehat h_e^{(j)}(\omega)
\exp\!\left[\frac{i\omega[x-x_e(y)]}{c(y)}\right]
+\int_{x_e(y)}^x
\frac{\widehat w_{\mathrm{Ek}}(x',y,\omega)}{c(y)}
\exp\!\left[\frac{i\omega(x-x')}{c(y)}\right]dx',
\]

with \(\widehat h_b(y,\omega)
=\widehat h(x_b(y),y,\omega)\). The symbol \(P_j\) above must not be
substituted for the exponential propagation factor in this solution.

The partial regional budget gives the total transport at any supported
latitude,

\[
\widehat T_j(y,\omega)=
\widehat T_{j,N}
+F_{j,\ge y}(\omega)
-r_{j,\ge y}(\omega)\widehat h_e^{(j)}(\omega).
\]

With \(T=T_{\mathrm{Ek}}+T_g\), the western thickness follows from

\[
T_g=\frac{g'H}{f}(h_e-h_w),
\qquad
h_w=h_e-\frac{fT_g}{g'H}.
\]

The output also carries the frequency-domain \(h_A,h_I,h_P\), transform
metadata, condition numbers, and the optional per-region compatibility
residual. Spectral arrays are retained for reproducibility and decomposition;
they are not a separate public result type.

## 8. Minimal interface

```python
geometry = MultiBasinGeometry.from_bathymetry(bathymetry)

forcing = GlobalForcing.from_time_series(
    wind_stress=wind_stress_anomaly,
    northern_transport=northern_transport,
    southern_transport=southern_transport,
)

model = GlobalAdjustmentModel(
    geometry=geometry,
    forcing=forcing,
    g_prime=0.02,
    H=1_000.0,
    rho0=1_027.0,
)

output = model.solve()
atlantic_transport = output.transport.sel(region="atlantic_north")
```

The exact constructors may evolve, but the ownership boundaries may not:

- Geometry owns boundaries, ordered connectivity, and boundary sharing;
- forcing owns input preprocessing and Fourier conventions;
- the model owns derivation of \(F_j\), \(r_j\), the global solve, and all
  diagnostics; and
- the output owns the complete labelled result for the five regions.

## 9. Validation and acceptance tests

The implementation should fail early when:

- the geometry does not contain the five required regions and child order;
- a boundary curve, mask, integration weight, \(x_e\), or \(x_b\) is missing;
- forcing time axes, units, or anomaly conventions disagree;
- the forcing frequency grid is incompatible with the requested model; or
- a derived array contains non-finite values outside a declared masked band.

The core acceptance tests are intentionally limited:

1. forcing `rfft`/`irfft` round trips recover the retained input interval;
2. zero wind produces zero derived Ekman pumping and section transports;
3. regional compatibility residuals converge with grid refinement;
4. the vectorized solve matches direct solves of each \(3\times3\) system;
5. the solved fields satisfy the original regional volume budgets;
6. \(T=T_{\mathrm{Ek}}+T_g\) and the \(h_w\) relation hold; and
7. every output variable covers all five regions with the forcing-owned time
   coordinate.

Worked examples should use this same path. They may differ in geometry and
forcing data, but they must not introduce example-specific solver classes or
independent internal Ekman transport inputs.
