"""The package exposes a semantic version string."""

from app import __version__


def test_version_is_semver() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3, f"expected MAJOR.MINOR.PATCH, got {__version__!r}"
    assert all(part.isdigit() for part in parts), f"non-numeric component in {__version__!r}"
