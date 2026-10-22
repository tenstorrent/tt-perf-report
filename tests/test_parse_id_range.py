#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest

from tt_perf_report import perf_report
from tt_perf_report.perf_report import parse_id_range


@pytest.mark.parametrize(
    "id_range, expected",
    [
        (None, None),
        ("10-20", (10, 20)),
        ("1,000-2,000", (1000, 2000)),
        ("10-", (10, None)),
        ("-20", (None, 20)),
        # ID 0 is a valid bound and must not be mistaken for a missing one.
        ("0-", (0, None)),
        ("-0", (None, 0)),
    ],
)
def test_parse_id_range(id_range, expected):
    assert parse_id_range(id_range) == expected


@pytest.mark.parametrize(
    "id_range",
    [
        # "-" must not reach filter_by_id_range as (None, None).
        pytest.param("-", id="dash-only"),
        pytest.param("", id="empty"),
        pytest.param(",-", id="separator-without-digits"),
        pytest.param("-,", id="comma-only-end"),
        pytest.param("10", id="no-dash"),
        pytest.param("1-2-3", id="two-dashes"),
        pytest.param("a-b", id="not-numbers"),
    ],
)
def test_parse_id_range_rejects_invalid(id_range):
    with pytest.raises(ValueError):
        parse_id_range(id_range)


def test_dash_only_is_reported_as_a_format_error(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["tt-perf-report", "dummy.csv", "--id-range", "-"])

    with pytest.raises(SystemExit) as excinfo:
        perf_report.parse_args()

    assert excinfo.value.code == 1
    assert "Invalid --id-range format" in capsys.readouterr().out
