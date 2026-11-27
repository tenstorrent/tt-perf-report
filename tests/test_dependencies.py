#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from importlib.metadata import requires, version

from packaging.requirements import Requirement
from packaging.version import Version


def test_pillow_meets_security_floor():
    # The floors are read from the installed package metadata, so pyproject.toml stays the
    # only place they are written down. Catches a Pillow installed around the resolver,
    # such as a pre-existing environment or a --no-deps install.
    pillow_requirements = [
        req
        for req in map(Requirement, requires("tt-perf-report") or [])
        if req.name.lower() == "pillow"
    ]
    applicable = [
        req for req in pillow_requirements if req.marker is None or req.marker.evaluate()
    ]
    assert applicable, "pyproject.toml declares no Pillow floor for this Python version"

    installed = Version(version("pillow"))
    for req in applicable:
        assert req.specifier.contains(installed, prereleases=True), (
            f"Pillow {installed} does not satisfy {req}"
        )
