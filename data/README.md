# Data layout

Repository data are separated by whether Git is responsible for their
distribution:

- `tracked/` contains small metadata, manifests, and eventually compact
  derived products that are intentionally version controlled.
- `untracked/` contains large source datasets held only on the local
  filesystem. Its contents are ignored by Git apart from `.gitkeep`.

The initial local inputs are copied from the sibling `atlantic_adjustment`
checkout and retain their source directory names:

```text
data/untracked/
├── ERA5/
├── GEBCO/
└── SCOTIA/
```

Do not put required package resources in `untracked/`. Code and tests must
accept explicit paths and must fail clearly when a local source dataset is
absent.

The v3 ERA5–SCOTIA forcing file beneath `untracked/forcing/` uses one temporal
convention. Every value is a calendar-month mean and the time coordinate is a
contiguous sequence containing the first day of every month. The source
timestamps are used only to match ERA5 and SCOTIA by calendar month; neither
source is treated as an instantaneous observation.
