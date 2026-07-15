# MOC Adjustment Theory

`moc-adjustment-theory` will provide modular reduced-gravity models of
meridional-overturning-circulation adjustment in connected ocean basins.

The repository is currently in its documentation-first setup phase. The only
public function is a temporary `hello_world` smoke test; scientific model code
will be added only after its abstractions and geometry pipeline have been
reviewed.

## Install from source

```bash
python -m pip install .
python -m moc_adjustment_theory
```

## Install for development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m mkdocs build --strict
```

## Temporary API

```python
from moc_adjustment_theory import hello_world

assert hello_world() == "Hello, world!"
```

See the [documentation](docs/index.md) for the corresponding command-line
smoke test.
