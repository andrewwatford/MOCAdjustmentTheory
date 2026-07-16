# Data layout

Repository data are separated by whether Git is responsible for their
distribution:

- `tracked/` contains small metadata, manifests, and eventually compact
  derived products that are intentionally version controlled.
- `untracked/` contains large source datasets held only on the local
  filesystem. Its contents are ignored by Git apart from `.gitkeep`.

Set `MOC_EXAMPLE_DATA_ROOT` to the `data/untracked/` directory. The worked
examples expect these source directory names:

```text
data/untracked/
├── ERA5/
├── GEBCO/
└── SCOTIA/
```

Do not put required package resources in `untracked/`. Code and tests must
accept explicit paths and must fail clearly when a local source dataset is
absent.
