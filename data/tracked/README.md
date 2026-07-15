# Tracked data

This directory is reserved for small data artifacts that are suitable for Git,
including source manifests, compact reference fixtures, and reviewed derived
geometry products.

Large observational and reanalysis source files belong in `../untracked/`.

`source_data_manifest.csv` records the byte size and SHA-256 digest of every
initial local source file. Paths in its final column are relative to the
original `atlantic_adjustment` checkout; `relative_path` is relative to this
repository's `data/untracked/` directory.
