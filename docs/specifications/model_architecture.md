# Global adjustment model specification

## 1. Scope and workflow

The package implements one solver, `GlobalAdjustmentModel`, for the fixed
five-region theory. The user:

1. loads a `MultiBasinGeometry`;
2. supplies one `xarray.Dataset` containing `M_Ek_x`, `M_Ek_y`, and `T_N`;
3. supplies an `FFTConvention`; and
4. calls `solve()`.

The result contains \(h_e\), \(h_b\), \(h_w\), and total, Ekman, and
geostrophic transport for every region. There is no separate Atlantic model,
generic topology compiler, port abstraction, or forcing class.

## 2. Geometry

The canonical isobath product contains six named traces and the complete
five-region layout:

| \(j\) | Region | Range | West / east trace |
|---|---|---|---|
| 1 | North Atlantic | \(y_I\) to \(y_N\) | Atlantic / Atlantic |
| 2 | North Indian | \(y_I\) to \(y_{N,I}\) | Indian / Indian |
| 3 | North Pacific | \(y_P\) to \(y_{N,P}\) | Pacific / Pacific |
| 4 | Atlantic–Indian transition | \(y_P\) to \(y_I\) | Atlantic / Indian |
| 5 | Atlantic–Pacific transition | \(y_S\) to \(y_P\) | Atlantic / Pacific |

The theory stops at \(y_S\), near the southern tip of South America. The
Atlantic is not considered north of \(y_N\); the Indian and Pacific closures
may lie farther north.

The file, rather than the caller, specifies the trace-to-region mappings and
bounds. `MultiBasinGeometry.from_isobath_dataset(isobaths)` validates that
convention and produces \(x_b^{(j)}(y)\) and \(x_e^{(j)}(y)\). Here \(x_b\) is
outside the western boundary-current region, not at the coast.

Only three eastern-boundary thicknesses are independent:

```text
h_A : region 1
h_I : regions 2 and 4
h_P : regions 3 and 5
```

## 3. Forcing and transform convention

The forcing dataset contains exactly:

- `M_Ek_x(time, latitude, longitude)` in \(\mathrm{m^2\,s^{-1}}\), positive
  eastward;
- `M_Ek_y(time, latitude, longitude)` in \(\mathrm{m^2\,s^{-1}}\), positive
  northward; and
- `T_N(time)` in Sverdrups or \(\mathrm{m^3\,s^{-1}}\), positive northward.

The package derives

\[
w_{\mathrm{Ek}}=\nabla\!\cdot\mathbf M_{\mathrm{Ek}}
\]

and every Ekman section transport from the same vector field. Wind-stress
conversion, \(\rho_0\), and equatorial regularization are outside the package.
The geometry product alone defines the integration domain; no basin mask is a
second input.

At the external southern closure the model assumes no geostrophic transport:

\[
T_S=T_{S,\mathrm{Ek}}.
\]

Thus \(T_S\) is derived directly from `M_Ek_y` across the southern boundary of
region 5. It is not an independent prescription.

`FFTConvention` declares the sample interval, reflected padding, and optional
`n_fft`. `GlobalAdjustmentModel` removes the forcing time means, applies one
shared NumPy `rfft`, solves on its non-negative angular-frequency coordinate,
and uses the matching `irfft` and crop. The zero-frequency anomaly is zero.

## 4. Regional terms

Within each region,

\[
\partial_t h-c(y)\,\partial_xh=-w_{\mathrm{Ek}},
\qquad
c(y)=\frac{\beta g'H}{f^2},
\]

with the documented low-latitude cap on \(c\). Define the budget kernel

\[
P_j(\omega,x,y)
=\exp\!\left[
\frac{i\omega\,[x_b^{(j)}(y)-x]}{c(y)}
\right]-1.
\]

Then

\[
F_j(\omega)=
\int_{y_{S,j}}^{y_{N,j}}
\int_{x_b^{(j)}(y)}^{x_e^{(j)}(y)}
P_j\,\widehat w_{\mathrm{Ek}}\,dx\,dy,
\]

\[
r_j(\omega)=
\int_{y_{S,j}}^{y_{N,j}}
c(y)P_j\!\left(\omega,x_e^{(j)}(y),y\right)\,dy,
\]

and the regional volume budget is

\[
T_{\mathrm{out}}^{(j)}-T_{\mathrm{in}}^{(j)}
=-F_j+r_jh_e^{(j)}.
\]

The internal branch transports are

\[
T_I=T_{I,\mathrm{Ek}}+\kappa_I(h_I-h_A),
\qquad
T_P=T_{P,\mathrm{Ek}}+\kappa_P(h_P-h_I),
\qquad
\kappa=\frac{g'H}{f}.
\]

## 5. The \(3\times3\) solve

At each \(\omega\), the model solves

\[
\begin{bmatrix}
r_1+\kappa_I & r_4+\kappa_P-\kappa_I & r_5-\kappa_P \\
-\kappa_I & r_2+\kappa_I & 0 \\
0 & -\kappa_P & r_3+\kappa_P
\end{bmatrix}
\begin{bmatrix}h_A\\h_I\\h_P\end{bmatrix}
=
\begin{bmatrix}
F_1+F_4+F_5+T_N+T_{I,\mathrm{Ek}}+T_{P,\mathrm{Ek}}-T_{S,\mathrm{Ek}}\\
F_2-T_{I,\mathrm{Ek}}\\
F_3-T_{P,\mathrm{Ek}}
\end{bmatrix}.
\]

The stack is vectorized over `omega`; condition numbers are retained as a
diagnostic.

## 6. Complete output

Every solve returns:

- `h_e(time, region)`;
- `h_b(time, region, latitude)`, evaluated at \(x_b\);
- `h_w(time, region, latitude)`, the western-boundary thickness;
- `transport(time, region, latitude)`; and
- `transport_ekman` and `transport_geostrophic`.

The characteristic solution used for \(h_b\) is

\[
\widehat h(x,y,\omega)
=\widehat h_e(\omega)
\exp\!\left[\frac{i\omega[x-x_e(y)]}{c(y)}\right]
+\int_{x_e(y)}^x
\frac{\widehat w_{\mathrm{Ek}}(x',y,\omega)}{c(y)}
\exp\!\left[\frac{i\omega(x-x')}{c(y)}\right]dx'.
\]

With \(T=T_{\mathrm{Ek}}+T_g\),

\[
h_w=h_e-\frac{fT_g}{g'H}.
\]

The output also retains compact spectra, \(F_j\), \(r_j\), `T_N`, derived
`T_S`, condition numbers, and the cheap compatibility residual

\[
\epsilon_j=
\iint_{R_j}w_{\mathrm{Ek}}\,dA
-\left(T_{\mathrm{Ek},N}-T_{\mathrm{Ek},S}\right).
\]

This residual is reported, not used to alter the forcing or close the solve.

## 7. Minimal interface

```python
forcing = xr.Dataset(
    {"M_Ek_x": M_Ek_x, "M_Ek_y": M_Ek_y, "T_N": T_N}
)
fft = FFTConvention(sample_interval_seconds=dt, n_fft=2_048)
output = GlobalAdjustmentModel(
    geometry,
    forcing,
    fft=fft,
    g_prime=0.02,
).solve()

atlantic = output.dataset.transport.sel(region="atlantic_north")
```

The implementation fails early for a noncanonical geometry file, missing or
misdimensioned forcing variables, invalid units or coordinates, nonuniform
sampling without an explicit interval, or nonfinite input values.
