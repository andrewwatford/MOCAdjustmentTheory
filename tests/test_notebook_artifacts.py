from __future__ import annotations

import json
from pathlib import Path

import pytest


NOTEBOOKS = ("global_era5_scotia.ipynb",)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_worked_notebook_is_valid_executed_artifact(name: str) -> None:
    path = Path(__file__).parents[1] / "notebooks" / name
    notebook = json.loads(path.read_text())

    assert notebook["nbformat"] == 4
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
