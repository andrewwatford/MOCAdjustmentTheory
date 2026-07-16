# Global adjustment model specification

## 1. Scope

This package exists to work with a global, linear reduced gravity adjustment theory.
It solves the theory equations in spectral space.
The user workflow is simple:

1. Construct an `xr.Dataset` that includes the boundaries of the active layer, typically a prescribed isobath.
2. Construct an `xr.Dataset` that includes an Ekman forcing field and a timeseries of northern boundary forcing. 
3. Define a `FourierTransformer` object for use in the solve.
4. Pass all three of those, along with the reduced gravity $g'$, to the `GlobalAdjustmentModel` constructor.
5. View your results in an output `xr.Dataset`!

## 2. Public objects

The user should really be constructing their own forcing fields and isobath datasets; the output of the model is also an `xr.Dataset`.
Thus, the only real public objects are the `GlobalAdjustmentModel` and `FourierTransformer` objects.

## 3. The theory itself
The theory itself is quite simple.
We divide the ocean into five regions, inferred from the isobath file provided.

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

The eastern boundary thickness anomaly has only three independent values:

```text
h_eA : region 1
h_eI : regions 2 and 4
h_eP : regions 3 and 5
```

However, the western boundary (just outside the western boundary current, which we assume is small) thickness anomaly is computed region-by-region:

\[
\widehat h_b^{(j)}(y,\omega)
=\widehat h_e^{(j)}(\omega)
\exp\!\left[\frac{i\omega[x_b^{(j)}(y)-x_e^{(j)}(y)]}{c(y)}\right]
-\int_{x_b^{(j)}(y)}^{x_e^{(j)}(y)}
\frac{\widehat w_{\mathrm{Ek}}(x',y,\omega)}{c(y)}
\exp\!\left[\frac{i\omega(x_b^{(j)}-x')}{c(y)}\right]dx',
\]

where

\[
c(y) = \min\{\beta g'H/f^2, \sqrt{g'H}/3\}
\]

is the Rossby wave speed. For notational simplicity we define:

\[
P_j(\omega,x,y)
=\exp\!\left[
\frac{i\omega\,[x_b^{(j)}(y)-x]}{c(y)}
\right]-1.
\]

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

The flow between regions (5) and (3), and (4) and (2), can be expressed as a combination of geostrophic and Ekman terms:

\[
T_I=T_{I,\mathrm{Ek}}+\kappa_I(h_I-h_A),
\qquad
T_P=T_{P,\mathrm{Ek}}+\kappa_P(h_P-h_I),
\]

where

\[
\kappa=\frac{g'H}{f},
\qquad
T_{Ek} = \int M_{Ek,y}\,dx'
\]

Altogether this gives us a system of equations to solve in frequency space:

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

Then, the partial regional budget gives the total transport at any supported
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


## 4. The GlobalAdjustmentModel
As mentioned, the basic use for the `GlobalAdjustmentModel` is as follows:

```python
model = GlobalAdjustmentModel(
  isobath_ds = isobath_ds,
  forcing_ds = forcing_ds,
  fourier_transformer = ft,
  g_prime = g # m/s**2
)
solution_ds = model.solve()
```
Now we will go into a bit more detail about what each of these parts is comprised of.

### 4.1 The active layer / isobath dataset
This is a dataset that the user provides with the following variables as a function of latitude:

1. `x_wP`, western boundary longitude for the Pacific basin;
2. `x_wA`, western boundary longitude for the Atlantic basin;
3. `x_wI`, western boundary longitude for the Indian basin;
4. `x_eP`, eastern boundary longitude for the Pacific basin;
5. `x_eA`, eastern boundary longitude for the Atlantic basin;
6. `x_eI`, eastern boundary longitude for the Indian basin.

It must also have an attribute `isobath_depth_m` which provides the active layer / isobath depth in meters.

### 4.2 The forcing dataset
This dataset contains three crucial `DataArray`s:

1. `M_Ek_x`, the local Ekman transport in the $x$-direction;
2. `M_Ek_y`, the local Ekman transport in the $y$-direction;
3. `T_N`, the total transport at the northern boundary of the Atlantic basin.

These are all defined on a common time grid, which will also be used for the solve.
The Ekman transports are also defined on a latitude-longitude grid that must include the entire active domain.
Because of ambiguities surrounding Ekman upwelling at the equator, Ekman transport into solid boundaries, etc., we elected to have the user make their own decisions on those fronts.
Transport at the southern boundary of the three-basin domain is assumed to be entirely wind-driven, and is derived by integrating $M_{Ek, y}$ across the southern latitude.
Similarly, Ekman upwelling is computed as the divergence of the Ekman transport: $w_{Ek} = \nabla \cdot \mathbf{M}_{Ek, y}$.

### 4.3 The FourierTransformer
To be honest I am not terribly comfortable with the subtleties around solving problems in frequency space. 
This will have to be fleshed out more as we go along.
The idea is that this is something that acts on DataArrays and Fourier / Inverse Fourier Transforms them consistently.

### 4.4 The output dataset
The output dataset consists of:

1. $h_e^{(j)}(t), h_b^{(j)}(y, t), h_w^{(j)}(y, t)$ for each of the five regions. Note that information will be repeated as $h_e$ is identical in a single ocean basin.
2. $T^{(j)}(y, t), T_{g}^{(j)}(y, t), T_{Ek}^{(j)}(y, t)$ for each of the five regions.