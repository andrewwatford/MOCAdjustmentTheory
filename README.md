# MOC Adjustment Theory

`moc-adjustment-theory` is a developing package for reduced-gravity adjustment
in connected ocean basins. It currently provides the boundary, basin, and
fixed non-Indonesian-Throughflow topology objects on which the numerical
models will be built.

## Install from source

```bash
python -m pip install .
```

## Install for development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs build --strict
```

## Current API

```python
from moc_adjustment_theory import Basin, BoundaryTrace, MultiBasinTopology
```

See the [documentation](https://andrewwatford.github.io/MOCAdjustmentTheory/)
for the geometry conventions, fixed topology, and generated API reference.
