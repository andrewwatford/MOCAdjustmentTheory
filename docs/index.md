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
pair consistently around the frequency-space model.

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

```text
h_eA : region 1
h_eI : regions 2 and 4
h_eP : regions 3 and 5
```

The western-boundary thickness anomaly, just outside the assumed-small western
boundary current, is computed region by region:

$$
\widehat h_b^{(j)}(y,\omega)
=\widehat h_e^{(j)}(\omega)
\exp\!\left[\frac{i\omega[x_b^{(j)}(y)-x_e^{(j)}(y)]}{c(y)}\right]
-\int_{x_b^{(j)}(y)}^{x_e^{(j)}(y)}
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
T_I=T_{I,\mathrm{Ek}}+\kappa_I(h_I-h_A),
\qquad
T_P=T_{P,\mathrm{Ek}}+\kappa_P(h_P-h_I),
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
\begin{bmatrix}h_A\\h_I\\h_P\end{bmatrix}
=
\begin{bmatrix}
F_1+F_4+F_5+T_N+T_{I,\mathrm{Ek}}+T_{P,\mathrm{Ek}}-T_S\\
F_2-T_{I,\mathrm{Ek}}\\
F_3-T_{P,\mathrm{Ek}}
\end{bmatrix}.
$$

The partial regional budget gives total transport at any supported latitude:

$$
\widehat T_j(y,\omega)=
\widehat T_{j,N}
+F_{j,\ge y}(\omega)
-r_{j,\ge y}(\omega)\widehat h_e^{(j)}(\omega).
$$

With $T=T_{\mathrm{Ek}}+T_g$, the western thickness follows from

$$
T_g=\frac{g'H}{f}(h_e-h_w),
\qquad
h_w=h_e-\frac{fT_g}{g'H}.
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
3. `T_N`: total transport at the northern boundary of the Atlantic basin.

All three use a common time grid, which is also used for the solve. The Ekman
transports use a latitude–longitude grid that covers the entire active domain.

The user is responsible for choices concerning Ekman upwelling at the equator,
Ekman transport into solid boundaries, and related ambiguities. Transport at
the southern boundary of the three-basin domain is assumed to be entirely
wind-driven and is obtained by integrating $M_{\mathrm{Ek},y}$ across the
southern latitude. Ekman upwelling is the divergence of the Ekman transport,
$w_{\mathrm{Ek}}=\nabla\mathbin{\cdot}\mathbf{M}_{\mathrm{Ek}}$.

### Output dataset

The output dataset contains:

1. $h_e^{(j)}(t)$, $h_b^{(j)}(y,t)$, and $h_w^{(j)}(y,t)$ for each of the five
   regions. The $h_e$ values repeat where regions share an ocean basin.
2. $T^{(j)}(y,t)$, $T_g^{(j)}(y,t)$, and $T_{\mathrm{Ek}}^{(j)}(y,t)$ for each
   of the five regions.
