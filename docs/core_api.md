# Core API

The public workflow has four objects and one solve:

```python
geometry = MultiBasinGeometry.from_isobath_dataset(
    isobaths,
    trace_variables=trace_variables,
    region_definitions=region_definitions,
)

forcing = GlobalForcing.from_time_series(
    M_ek_x=ekman_transport.x,
    M_ek_y=ekman_transport.y,
    northern_transport=scotia,
    southern_transport=southern_transport,
    sample_interval_seconds=365.25 * 86_400 / 12,
    n_fft=2_048,
)

output = GlobalAdjustmentModel(
    geometry,
    forcing,
    g_prime=0.02,
).solve()
```

`GlobalForcing` aligns all four inputs, converts transports to SI units,
removes their time means by default, applies the declared reflected padding,
and creates one immutable `rfft` frequency grid. Internal Indian and Pacific
Ekman transports are not accepted as inputs; the model derives them from the
same vector field used for Ekman pumping. Wind-stress conversion, \(\rho_0\),
equatorial regularization, and coastal tapering are upstream user choices.
The component signs are eastward and northward; all boundary and output
transports are positive northward, and positive divergence is upward pumping.

`GlobalAdjustmentModel.solve()` derives the five regional Ekman terms, solves
the stack of \(3\times3\) systems, and returns every standard diagnostic in
one pass. `output.dataset` contains time-domain `h_e`, `h_b`, `h_w`, total
transport, and its Ekman and geostrophic components. `output.spectral` retains
the compact frequency-domain solution, \(F_j\), \(r_j\), matrix condition
numbers, and budget/compatibility residuals.

Transport at an arbitrary supported latitude is a labelled interpolation:

```python
transport_26n = output.transport_at("atlantic_north", 26.5)
```

All transforms back to time use the padding, crop, sampling interval, and
frequency coordinate owned by the forcing object.
