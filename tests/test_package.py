from __future__ import annotations

import importlib.metadata

import xclimatology as m


def test_version() -> None:
    assert importlib.metadata.version("xclimatology") == m.__version__
