#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest

from tt_perf_report.perf_report import parse_id_range


@pytest.mark.parametrize(
    "id_range, expected",
    [
        (None, None),
        ("10-20", (10, 20)),
        ("1,000-2,000", (1000, 2000)),
        ("10-", (10, None)),
        ("-20", (None, 20)),
    ],
)
def test_parse_id_range(id_range, expected):
    assert parse_id_range(id_range) == expected


@pytest.mark.parametrize("id_range", ["-", "10", "1-2-3", "a-b"])
def test_parse_id_range_rejects_invalid(id_range):
    with pytest.raises(ValueError):
        parse_id_range(id_range)
