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
    ("AllGather", "CCL"),
    ("ReshapeView", "TM"),
    ("Repeat", "TM"),
    ("PaddedSlice", "TM"),
    ("SliceWrite", "TM"),
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


# Collectives from tt-metal's ttnn/operations/ccl and experimental/ccl, plus the DeepSeek prefill
# MoE dispatch/combine pair, which also run over the fabric.
CCL_OPS = [
    "AllGather", "AllGatherAsync", "AllGatherConcat",
    "ReduceScatter", "ReduceScatterMinimalAsync", "ReduceScatterMinimalDirect",
    "StridedReduceScatterAsync", "LlamaReduceScatter", "DeepseekMoEReduceScatter",
    "AllReduceAsync",
    "AllToAllAsync", "AllToAllAsyncGeneric", "AllToAllDispatch", "AllToAllDispatchMetadata",
    "AllToAllCombine",
    "AllBroadcast", "Broadcast",
    "SendAsync", "RecvAsync", "SendDirectAsync", "RecvDirectAsync",
    "Dispatch", "Combine",
]


@pytest.mark.parametrize("op_code", CCL_OPS)
def test_collectives_are_ccl(op_code):
    assert classify_operation(op_code) == "CCL"
    assert classify_operation(f"{op_code}DeviceOperation") == "CCL"


@pytest.mark.parametrize("op_code", ["AllGatherMatmul", "RMSAllGather"])
def test_fused_collective_compute_ops_stay_compute(op_code):
    assert classify_operation(f"{op_code}DeviceOperation") == "Compute"


def test_no_op_belongs_to_two_categories():
    seen = {}
    for category, operations in perf_report.OPERATION_CATEGORIES.items():
        for operation in operations:
            assert operation not in seen, f"{operation} is in both {seen[operation]} and {category}"
            seen[operation] = category


@pytest.mark.parametrize("op_code", ["", "   "])
def test_empty_op_code_is_other_without_warning(op_code, capsys):
    assert classify_operation(op_code) == "Other"
    captured = capsys.readouterr()
    assert "Unclassified operation" not in captured.out


def test_every_device_category_has_a_chart_order_and_colours():
    # Host ops are never charted; everything else classify_operation returns must be orderable and
    # coloured, or pd.Categorical sorts it as NaN and the plot falls back to generic colours.
    device_categories = (set(perf_report.OPERATION_CATEGORIES) - {"Host"}) | {"Other"}

    assert set(perf_report.CATEGORY_ORDER) == device_categories
    assert set(perf_report._get_category_color_palettes()) == device_categories
    assert set(perf_report._get_category_border_colors()) == device_categories
