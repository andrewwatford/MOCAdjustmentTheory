"""Tests for the temporary package-scaffold API."""

import pytest

from moc_adjustment_theory import hello_world


def test_hello_world_default() -> None:
    assert hello_world() == "Hello, world!"


def test_hello_world_normalizes_a_name() -> None:
    assert hello_world("  ocean  ") == "Hello, ocean!"


@pytest.mark.parametrize("invalid", ["", "   "])
def test_hello_world_rejects_empty_names(invalid: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        hello_world(invalid)


def test_hello_world_rejects_non_strings() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        hello_world(42)  # type: ignore[arg-type]
