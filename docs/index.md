# MOC Adjustment Theory

`moc-adjustment-theory` implements a global, linear reduced-gravity adjustment
theory in spectral space.

## Scope

The user workflow is:

1. Construct an `xarray.Dataset` describing the active-layer boundaries,
   typically a prescribed isobath.
2. Construct an `xarray.Dataset` containing an Ekman forcing field and a time
   series of northern-boundary forcing.
3. Pass those datasets and the reduced gravity $g'$ to `GlobalRossbyModel`.
4. Call `solve()` to obtain the results as an `xarray.Dataset`.

Users provide the forcing and isobath datasets, and the model returns another
dataset. The Fourier interface is the stateless `forward_transform` and
`inverse_transform` function pair. `GlobalRossbyModel.solve()` applies that
pair consistently around the frequency-space model. `butterworth_filter`
provides a zero-phase low-pass filter for time-dependent `xarray` data.

## Theory

The ocean is divided into five regions inferred from the isobath dataset:

| Label | Region | Meridional range | Western / eastern boundary |
|---|---|---|---|
| 1 | North Atlantic | $y_I$ to $y_N$ | Atlantic / Atlantic |
| 2 | North Indian | $y_I$ to $y_{N,I}$ | Indian / Indian |
| 3 | North Pacific | $y_P$ to $y_{N,P}$ | Pacific / Pacific |
| 4 | Atlantic–Indian transition | $y_P$ to $y_I$ | Atlantic / Indian |
| 5 | Atlantic–Pacific transition | $y_S$ to $y_P$ | Atlantic / Pacific |

The theory stops at $y_S$, near the southern tip of South America. It does not
model the Southern Ocean as an additional connected basin.

The eastern-boundary thickness anomaly has only three independent values:
$h_e^{(A)}$ for region 1, $h_e^{(I)}$ for regions 2 and 4, and $h_e^{(P)}$
for regions 3 and 5. The index on $h_e^{(j)}$ therefore denotes either a
region or an ocean, according to context.

The western-boundary thickness anomaly, just outside the assumed-small western
boundary current, is computed region by region:

$$
\widehat h_b^{(j)}(y,\omega)
=\widehat h_e^{(j)}(\omega)
\exp\!\left[\frac{i\omega[x_b^{(j)}(y)-x_e^{(j)}(y)]}{c(y)}\right]
-\int_{x_b^{(j)}}^{x_e^{(j)}(y)}
\frac{\widehat w_{\mathrm{Ek}}(x',y,\omega)}{c(y)}
\exp\!\left[\frac{i\omega(x_b^{(j)}-x')}{c(y)}\right]dx',
$$

where

$$
c(y) = \min\{\beta g'H/f^2, \sqrt{g'H}/3\}
$$

is the Rossby wave speed. Define

$$
P_j(\omega,x,y)
=\exp\!\left[\frac{i\omega[x_b^{(j)}(y)-x]}{c(y)}\right]-1,
$$

$$
F_j(\omega)=
\int_{y_{S,j}}^{y_{N,j}}
\int_{x_b^{(j)}(y)}^{x_e^{(j)}(y)}
P_j(\omega,x,y)\,
\widehat w_{\mathrm{Ek},j}(x,y,\omega)\,dx\,dy,
$$

and

$$
r_j(\omega)=
\int_{y_{S,j}}^{y_{N,j}}
c(y)P_j\!\left(\omega,x_e^{(j)}(y),y\right)\,dy.
$$

These definitions give the thin-western-boundary-current regional budget

$$
T_{\mathrm{out}}^{(j)}-T_{\mathrm{in}}^{(j)}
=-F_j+r_jh_e^{(j)}.
$$

Flow between regions 5 and 3, and between regions 4 and 2, is a combination of
geostrophic and Ekman terms:

$$
T_I=T_{I,\mathrm{Ek}}+\kappa_I\left(h_e^{(I)}-h_e^{(A)}\right),
\qquad
T_P=T_{P,\mathrm{Ek}}+\kappa_P\left(h_e^{(P)}-h_e^{(I)}\right),
$$

where

$$
\kappa=\frac{g'H}{f},
\qquad
T_{\mathrm{Ek}} = \int M_{\mathrm{Ek},y}\,dx'.
$$

The frequency-space system is

$$
\begin{bmatrix}
r_1+\kappa_I & r_4+\kappa_P-\kappa_I & r_5-\kappa_P \\
-\kappa_I & r_2+\kappa_I & 0 \\
0 & -\kappa_P & r_3+\kappa_P
\end{bmatrix}
\begin{bmatrix}
\widehat h_e^{(A)}\\
\widehat h_e^{(I)}\\
\widehat h_e^{(P)}
\end{bmatrix}
=
\begin{bmatrix}
F_1+F_4+F_5+T_N+T_{I,\mathrm{Ek}}+T_{P,\mathrm{Ek}}-T_S\\
F_2-T_{I,\mathrm{Ek}}\\
F_3-T_{P,\mathrm{Ek}}
\end{bmatrix}.
$$

At zero frequency, zero-mean forcing leaves one part of this system
undetermined: adding the same constant to $\widehat h_e^{(A)}(0)$,
$\widehat h_e^{(I)}(0)$, and $\widehat h_e^{(P)}(0)$ does not change any
thickness difference or transport. The solver chooses this common offset to be
zero, so the three zero-frequency $h_e$ coefficients vanish. Equivalently,
$h_e$ has zero mean over the full padded FFT interval. This does not require
the returned time series to have zero sample mean after the padded solution is
cropped to the original time coordinate. A nonzero absolute mean thickness
would require an additional volume or initial condition that is not part of
the forcing dataset.

The partial regional budget gives total transport at any supported latitude:

$$
\widehat T_j(y,\omega)=
\widehat T_{j,N}
+F_{j,\ge y}(\omega)
-r_{j,\ge y}(\omega)\widehat h_e^{(j)}(\omega).
$$

With $T=T_{\mathrm{Ek}}+T_g$, the western thickness follows from

$$
T_g^{(j)}=\frac{g'H}{f}\left(h_e^{(j)}-h_w^{(j)}\right),
\qquad
h_w^{(j)}=h_e^{(j)}-\frac{fT_g^{(j)}}{g'H}.
$$

We can also obtain the height field for any point in the domain:

$$
\widehat h^{(j)}(x,y,\omega)
=\widehat h_e^{(j)}(\omega)
\exp\!\left[\frac{i\omega[x-x_e^{(j)}(y)]}{c(y)}\right]
-\int_{x(y)}^{x_e^{(j)}(y)}
\frac{\widehat w_{\mathrm{Ek}}(x',y,\omega)}{c(y)}
\exp\!\left[\frac{i\omega(x-x')}{c(y)}\right]dx'.
$$

## Model interface

The intended interface is:

```python
model = GlobalRossbyModel(
    isobath_ds=isobath_ds,
    g_prime=g_prime,  # m s^-2
)
solution_ds = model.solve(forcing_ds)
```

By default, `solve(pad_length=None)` computes the longest zonal Rossby-wave
crossing time from the supplied geometry, active-layer depth, and reduced
gravity. It converts that duration to forcing time steps and appends at least
that many zero samples. The complete FFT length is made odd to avoid a
self-conjugate Nyquist coefficient when the model applies complex propagation
phases. An integer `pad_length` overrides the physical default and specifies
the minimum number of zero samples to append.

`solve` is Dask-native. It constructs and returns the complete model graph
without loading the full forcing dataset or calculating any output values.
All numerical output variables are Dask-backed, including the dense
`h(time, region, latitude, longitude)` field. The FFT axes are temporarily
rechunked to complete time or frequency series, while the returned data use
12-sample time, single-region, 64-latitude, and 128-longitude chunks. A caller
can compute only a subset or stream the complete result to a chunked store:

```python
atlantic_snapshot = solution_ds.h.sel(
    region="north_atlantic",
).isel(time=0).compute()

solution_ds.to_zarr("solution.zarr")
```

Calling `.compute()` or `.values` on the entire dense height field necessarily
materializes its full logical size in memory. Writing with Dask streams chunks
through memory instead.

The standalone `forward_transform` uses the same right-padding and odd
total-length convention. Its `pad_length` defaults to zero because a stateless
transform has no model geometry from which to infer a crossing time;
`inverse_transform` reads the complete transform contract from the spectrum's
metadata.

For low-pass filtering, `butterworth_filter(data, cutoff_omega, order=4)`
accepts either a `DataArray` or `Dataset`. It designs the Butterworth filter as
second-order sections and applies it forward and backward, producing zero phase
and twice the stated order. Consequently, `cutoff_omega` is the single-pass
$-3$ dB angular frequency in rad s$^{-1}$ and the final amplitude there is
$1/2$. Numeric variables containing the selected time dimension are filtered,
while other dataset variables and complete-series spatial masks are preserved.
Odd reflection is used at both endpoints with SciPy's standard padding length.
Padding reduces but cannot eliminate endpoint transients, so conclusions that
depend on the record ends should be treated cautiously.

### Active-layer dataset

The user-provided isobath dataset contains the following variables as functions
of latitude:

1. `x_wP`: western-boundary longitude of the Pacific basin.
2. `x_wA`: western-boundary longitude of the Atlantic basin.
3. `x_wI`: western-boundary longitude of the Indian basin.
4. `x_eP`: eastern-boundary longitude of the Pacific basin.
5. `x_eA`: eastern-boundary longitude of the Atlantic basin.
6. `x_eI`: eastern-boundary longitude of the Indian basin.

Its `isobath_depth_m` attribute gives the active-layer or isobath depth in
metres.

### Forcing dataset

The forcing dataset contains three required data arrays:

1. `M_Ek_x`: local Ekman transport in the $x$ direction.
2. `M_Ek_y`: local Ekman transport in the $y$ direction.
3. `T_N`: total transport at the northern boundary of the Atlantic basin. Its
   numeric `latitude_degrees_north` attribute gives the latitude at which that
   transport is prescribed.

All three use a common time grid, which is also used for the solve. The Ekman
transports use a latitude–longitude grid that covers the entire active domain.
The model retains North Atlantic rows at or south of the prescribed `T_N`
latitude, without changing the Indian or Pacific limits. The latitude must not
fall south of the equator, and both the forcing grid and Atlantic boundary
geometry must reach it.

The user is responsible for choices concerning Ekman upwelling at the equator,
Ekman transport into solid boundaries, and related ambiguities. Transport at
the southern boundary of the three-basin domain is assumed to be entirely
wind-driven and is obtained by integrating $M_{\mathrm{Ek},y}$ across the
southern latitude. Ekman upwelling is the divergence of the Ekman transport,
$w_{\mathrm{Ek}}=\nabla\mathbin{\cdot}\mathbf{M}_{\mathrm{Ek}}$.

### Output dataset

The output dataset contains:

1. $h_e^{(j)}(t)$, $h_b^{(j)}(y,t), h_w^{(j)}(y,t),$ and $h^{(j)}(x,y,t)$ for each of the five
   regions. The full field has dimensions
   `(time, region, latitude, longitude)` and is lazily masked with NaN outside
   each active region. The $h_e$ values repeat where regions share an ocean
   basin.
2. $T^{(j)}(y,t)$, $T_g^{(j)}(y,t)$, and $T_{\mathrm{Ek}}^{(j)}(y,t)$ for each
   of the five regions.
