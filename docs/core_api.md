# Core API

The model accepts a geometry, one forcing dataset, and an FFT convention:

```python
import xarray as xr
from moc_adjustment_theory import FFTConvention, GlobalAdjustmentModel

forcing = xr.Dataset(
    {
        "M_Ek_x": M_Ek_x,  # (time, latitude, longitude), m2 s-1
        "M_Ek_y": M_Ek_y,  # (time, latitude, longitude), m2 s-1
        "T_N": T_N,        # (time,), Sv or m3 s-1
    }
)
fft = FFTConvention(
    sample_interval_seconds=365.25 * 86_400 / 12,
    padding_samples=T_N.sizes["time"] - 1,
    n_fft=2_048,
)
output = GlobalAdjustmentModel(
    geometry,
    forcing,
    fft=fft,
    g_prime=0.02,
).solve()
```

The model converts `T_N` to SI units, removes the three time means, applies one
shared `rfft`, solves the \(3\times3\times n_\omega\) system, and applies the
matching `irfft` and crop. `M_Ek_x` is positive eastward; `M_Ek_y`, `T_N`, and
all output transports are positive northward.

The southern boundary condition is derived, not supplied:

\[
T_S=T_{S,\mathrm{Ek}},
\]

which states that the geostrophic transport through the external southern
closure is zero. Internal Indian and Pacific Ekman transports are likewise
derived from the same `M_Ek_y` field.

`output.dataset` contains time-domain `h_e`, `h_b`, `h_w`, total transport,
and its Ekman and geostrophic components. `output.spectral` contains the
compact frequency-domain solution, \(F_j\), \(r_j\), `T_N`, derived `T_S`,
condition numbers, and budget diagnostics.

Transport at any supported latitude is a labelled interpolation:

```python
transport_26n = output.transport_at("atlantic_north", 26.5)
```

Wind-stress conversion, reference density, and equatorial regularization are
upstream choices. The package boundary is vector Ekman transport.
