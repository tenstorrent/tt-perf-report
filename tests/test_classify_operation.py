#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest

from tt_perf_report import perf_report
from tt_perf_report.perf_report import classify_operation


# Ops newly classified for issue #61 / PR #63 (base names; DeviceOperation aliases auto-added).
NEWLY_CLASSIFIED_OPS = [
    ("RMSAllGather", "Compute"),
    ("SdpaDecode", "Compute"),
    ("RotaryEmbedding", "Compute"),
    ("FastReduceNC", "Compute"),
    ("ArgMax", "Compute"),
    ("AllGather", "DM"),
    ("ReshapeView", "TM"),
    ("Repeat", "TM"),
]


@pytest.fixture(autouse=True)
def _reset_classification_cache():
    """Rebuild category cache and unclassified warning set around each test."""
    perf_report.OPERATION_CATEGORIES_EXTENDED = None
    perf_report._UNCLASSIFIED_OPS_WARNED.clear()
    yield
    perf_report.OPERATION_CATEGORIES_EXTENDED = None
    perf_report._UNCLASSIFIED_OPS_WARNED.clear()


@pytest.mark.parametrize("op_code,expected_category", NEWLY_CLASSIFIED_OPS)
def test_newly_classified_ops_have_expected_category(op_code, expected_category):
    assert classify_operation(op_code) == expected_category
    assert classify_operation(f"{op_code}DeviceOperation") == expected_category


@pytest.mark.parametrize("op_code,expected_category", NEWLY_CLASSIFIED_OPS)
def test_newly_classified_ops_do_not_warn(op_code, expected_category, capsys):
    classify_operation(f"{op_code}DeviceOperation")
    captured = capsys.readouterr()
    assert "Unclassified operation" not in captured.out
    assert "Unclassified operation" not in captured.err
    assert expected_category != "Other"


def test_unknown_op_warns_once_and_returns_other(capsys):
    assert classify_operation("TotallyUnknownOpDeviceOperation") == "Other"
    first = capsys.readouterr()
    assert "Unclassified operation 'TotallyUnknownOpDeviceOperation'" in first.out

    assert classify_operation("TotallyUnknownOpDeviceOperation") == "Other"
    second = capsys.readouterr()
    assert second.out == ""
