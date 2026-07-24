"""Template smoke tests."""

import importlib


def test_template_submodule_is_importable() -> None:
    """The template package should be importable after installation."""
    module = importlib.import_module("fiery.things")
    assert module is not None
