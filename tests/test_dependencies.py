#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from importlib.metadata import PackageNotFoundError, requires, version

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

# One Python version either side of each Pillow marker boundary in pyproject.toml.
SUPPORTED_PYTHONS = ["3.10", "3.11", "3.13"]


def _pillow_floors(environment=None):
    """Return the Pillow requirements that apply to environment (default: this Python).

    The floors are read from the installed package metadata, so pyproject.toml stays the
    only place they are written down. Lowering a floor is not caught here by design.
    """
    try:
        declared = requires("tt-perf-report") or []
    except PackageNotFoundError:
        pytest.fail(
            "tt-perf-report is not installed, so its Pillow floors cannot be read; "
            "run `pip install -e .[test]` (reinstall after editing pyproject.toml)"
        )
    return [
        req
        for req in map(Requirement, declared)
        if req.name.lower() == "pillow"
        and (req.marker is None or req.marker.evaluate(environment))
    ]


@pytest.mark.parametrize("python_version", SUPPORTED_PYTHONS)
def test_each_supported_python_has_one_pillow_floor(python_version):
    # Runs for every Python from any interpreter, so a deleted floor is caught locally.
    floors = _pillow_floors({"python_version": python_version})
    assert len(floors) == 1, f"expected one Pillow floor for Python {python_version}, got {floors}"


def test_pillow_meets_security_floor():
    # Catches a Pillow installed around the resolver, such as a pre-existing environment
    # or a --no-deps install.
    floors = _pillow_floors()
    assert floors, "pyproject.toml declares no Pillow floor for this Python version"

    installed = Version(version("pillow"))
    for req in floors:
        assert req.specifier.contains(installed, prereleases=True), (
            f"Pillow {installed} does not satisfy {req}"
        )
