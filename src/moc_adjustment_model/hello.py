"""Temporary smoke-test API for the package scaffold."""


def hello_world(name: str = "world") -> str:
    """Return a small greeting used to verify an installation.

    This function is intentionally temporary. It gives source, editable, and
    wheel installations a stable import to exercise before scientific APIs
    are introduced in later reviewed changes.
    """

    if not isinstance(name, str):
        raise TypeError("name must be a string")
    normalized = name.strip()
    if not normalized:
        raise ValueError("name must not be empty")
    return f"Hello, {normalized}!"


def main() -> None:
    """Print the scaffold greeting for the temporary console command."""

    print(hello_world("MOC adjustment model"))

