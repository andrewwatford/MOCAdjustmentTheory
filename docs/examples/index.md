# Worked examples

The notebooks use the same definitive topology: the northern Indian and
Pacific basins are closed, and there is no Indonesian Throughflow connection.
They accept vector Ekman transport as model input; any wind-stress conversion
shown here is an explicit upstream notebook choice.

- [Idealized adjustment](01_idealized_non_itf_adjustment.md) isolates
  propagation and damping on a synthetic five-region geometry.
- [Global ERA5 + OSNAP](02_global_non_itf_osnap.md) uses ERA5 wind-stress
  anomalies upstream and OSNAP as the only prescribed northern transport.
- [Atlantic SCOTIA decomposition](03_atlantic_scotia_decomposition.md)
  reproduces the established wind/thermohaline decomposition of $h_b$, $h_w$,
  and no-Ekman transport without adding a new scientific experiment.

The executable `.ipynb` files live in the repository's `notebooks/` directory.
The realistic notebooks read source data from `MOC_EXAMPLE_DATA_ROOT`, which
must contain the `ERA5/`, `OSNAP/`, and `SCOTIA/` directories described in the
repository data README. Source data are not bundled in the package.
