# MOC Adjustment Theory

`moc-adjustment-theory` provides modular reduced-gravity adjustment models for
connected ocean basins. Development is proceeding in reviewed scientific
slices.

The current implementation contains the shared boundary traces, separately
initialized basins, and fixed five-basin non-Indonesian-Throughflow topology.
Wind forcing, frequency-space dynamics, and model solutions will follow in
subsequent reviewed changes.

## Install for development

From the repository root:

```console
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs serve
```

## Core objects

Each basin owns its north/south limits and references shared west/east isobath
traces. Named inflows and outflows are stitched together by
`MultiBasinTopology` without a separate port or equation-compiler abstraction.

The [Core API](reference/core.md) is generated directly from the scientific
docstrings. The design specifications remain available while implementation
and validation are in progress.
