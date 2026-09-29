#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import csv
import os
from io import StringIO

import pytest

from tt_perf_report import perf_report
from tt_perf_report.perf_report import (
    CATEGORY_ORDER,
    Cell,
    add_derived_columns,
    generate_perf_report,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# Conv, Halo, Matmul and signpost rows.
CONV_FIXTURE = os.path.join(DATA_DIR, "ops_perf_results_2025_09_18_11_39_20.csv")
# AllGatherAsync and ReduceScatterMinimalAsync rows.
CCL_FIXTURE = os.path.join(DATA_DIR, "bh_8xp150_deepseek_v3_d_p.csv")


@pytest.fixture(autouse=True)
def _reset_classification_cache():
    perf_report.OPERATION_CATEGORIES_EXTENDED = None
    perf_report._UNCLASSIFIED_OPS_WARNED.clear()
    yield
    perf_report.OPERATION_CATEGORIES_EXTENDED = None
    perf_report._UNCLASSIFIED_OPS_WARNED.clear()


def _run_report(csv_path, tmp_path, mocker, group_by="op", stacked=False):
    stdout = mocker.patch("sys.stdout", new_callable=StringIO)
    output_csv = tmp_path / "report.csv"
    summary_base = tmp_path / "summary"
    generate_perf_report(
        csv_files=[csv_path],
        start_signpost=None,
        end_signpost=None,
        ignore_signposts=True,
        print_signposts=True,
        min_percentage=0.5,
        id_range=None,
        arch=None,
        csv_output_file=str(output_csv),
        no_advice=True,
        tracing_mode=False,
        raw_op_codes=True,
        no_host_ops=False,
        no_summary=not stacked,
        group_by=group_by,
        classic_colors=False,
        summary_file=str(summary_base) if stacked else None,
        no_stacked_report=not stacked,
        no_stack_by_in0=True,
        stacked_csv=None,
        no_merge_devices=False,
    )
    with open(output_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    stacked_rows = []
    if stacked:
        with open(f"{summary_base}.csv", newline="") as f:
            stacked_rows = list(csv.DictReader(f))
    return rows, stacked_rows, stdout.getvalue()


def _expected_bound_analysis(raw_op_code):
    if "Matmul" in raw_op_code:
        return "full"
    if "OptimizedConvNew" in raw_op_code or "Conv2d" in raw_op_code:
        return "flops_only"
    return "none"


def test_bound_analysis_names_the_model_that_ran(tmp_path, mocker):
    rows, _, _ = _run_report(CONV_FIXTURE, tmp_path, mocker)

    seen = {row["Bound Analysis"] for row in rows}
    assert seen == {"full", "flops_only", "none"}
    for row in rows:
        assert row["Bound Analysis"] == _expected_bound_analysis(row["Raw OP Code"]), row["Raw OP Code"]


def test_unanalysed_ops_leave_dram_and_flops_blank(tmp_path, mocker):
    rows, _, _ = _run_report(CONV_FIXTURE, tmp_path, mocker)

    for row in rows:
        if row["Bound Analysis"] == "none":
            assert row["DRAM %"] == "" and row["FLOPs %"] == "", row["Raw OP Code"]
        elif row["Bound Analysis"] == "flops_only":
            assert row["DRAM %"] == "", row["Raw OP Code"]


def test_signposts_have_no_op_category_and_do_not_warn(tmp_path, mocker):
    rows, _, stdout = _run_report(CONV_FIXTURE, tmp_path, mocker)

    signposts = [row for row in rows if "(signpost)" in row["OP Code"]]
    assert signposts, "fixture should contain signposts"
    assert all(row["Op Category"] == "" for row in signposts)
    for row in signposts:
        base = row["Raw OP Code"].split()[0]
        assert f"Unclassified operation '{base}'" not in stdout

    device_rows = [row for row in rows if "(signpost)" not in row["OP Code"]]
    assert all(row["Op Category"] in CATEGORY_ORDER for row in device_rows)


def test_async_collectives_are_classified_ccl_per_op(tmp_path, mocker):
    rows, _, _ = _run_report(CCL_FIXTURE, tmp_path, mocker)

    categories = {row["Raw OP Code"]: row["Op Category"] for row in rows}
    assert categories["AllGatherAsyncDeviceOperation"] == "CCL"
    assert categories["ReduceScatterMinimalAsyncDeviceOperation"] == "CCL"
    assert categories["MatmulDeviceOperation"] == "Compute"


def test_stacked_report_grouped_by_category_has_a_ccl_group(tmp_path, mocker):
    _, stacked_rows, _ = _run_report(CCL_FIXTURE, tmp_path, mocker, group_by="category", stacked=True)

    assert "CCL" in {row["Op Code"] for row in stacked_rows}


def test_stacked_report_sorts_ccl_without_nan(tmp_path, mocker):
    _, stacked_rows, _ = _run_report(CCL_FIXTURE, tmp_path, mocker, stacked=True)

    categories = [row["Op Category"] for row in stacked_rows]
    assert "CCL" in categories
    assert all(category in CATEGORY_ORDER for category in categories)


def _matmul_row(dram_percentage, flops_percentage):
    return {
        "OP Code": Cell("MatmulDeviceOperation 32 x 64 x 128"),
        "Device Time": Cell(10.0),
        "Op-to-Op Gap": Cell(1.0),
        "DRAM %": Cell(dram_percentage),
        "FLOPs %": Cell(flops_percentage),
        "Bound": Cell(""),
    }


@pytest.mark.parametrize(
    "dram_percentage,flops_percentage,expected",
    [
        (0.0, 0.0, "SLOW"),
        (0.0, 80.0, "FLOP"),
        (80.0, 0.0, "DRAM"),
        (None, 80.0, ""),
        (80.0, None, ""),
    ],
)
def test_zero_percent_matmul_is_measured_not_missing(dram_percentage, flops_percentage, expected):
    rows = [_matmul_row(dram_percentage, flops_percentage)]
    add_derived_columns(rows)

    assert rows[0]["Bound"].raw_value == expected
