#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import csv
import os
from io import StringIO

import pandas as pd
import pytest

from tt_perf_report import perf_report
from tt_perf_report.perf_report import (
    CATEGORY_ORDER,
    Cell,
    _sort_dataframe_by_category,
    add_derived_columns,
    generate_perf_report,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# Conv, Halo, Matmul and signpost rows.
CONV_FIXTURE = os.path.join(DATA_DIR, "ops_perf_results_2025_09_18_11_39_20.csv")
# AllGatherAsync and ReduceScatterMinimalAsync rows.
CCL_FIXTURE = os.path.join(DATA_DIR, "bh_8xp150_deepseek_v3_d_p.csv")

pytestmark = pytest.mark.usefixtures("reset_classification_cache")


def _run_report(csv_path, tmp_path, mocker, group_by="op", stacked=False, no_host_ops=False):
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
        no_host_ops=no_host_ops,
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


# Pinned by name rather than re-deriving the production branch conditions, so a change to how
# analyze_op dispatches fails here instead of being mirrored.
EXPECTED_BOUND_ANALYSIS = {
    "Matmul": "full",
    "OptimizedConvNew": "flops_only",
    "HaloDeviceOperation": "none",
    "MoveDeviceOperation": "none",
    "GroupNorm": "none",
}


def _base_op(row):
    return row["Raw OP Code"].split()[0]


def test_bound_analysis_names_the_model_that_ran(tmp_path, mocker):
    rows, _, _ = _run_report(CONV_FIXTURE, tmp_path, mocker)

    by_op = {}
    for row in rows:
        by_op.setdefault(_base_op(row), set()).add(row["Bound Analysis"])
    for op, expected in EXPECTED_BOUND_ANALYSIS.items():
        assert by_op[op] == {expected}, op


def test_bound_analysis_matches_which_figures_are_present(tmp_path, mocker):
    rows, _, _ = _run_report(CONV_FIXTURE, tmp_path, mocker)
    device_rows = [row for row in rows if "(signpost)" not in row["OP Code"]]

    for row in device_rows:
        if row["Bound Analysis"] == "none":
            assert (row["DRAM %"], row["FLOPs %"], row["Bound"]) == ("", "", ""), row["Raw OP Code"]
        elif row["Bound Analysis"] == "flops_only":
            assert (row["DRAM %"], row["Bound"]) == ("", ""), row["Raw OP Code"]

    # Guards the other direction: blanking every figure would satisfy the loop above.
    assert any(row["DRAM %"] and row["FLOPs %"] for row in device_rows if row["Bound Analysis"] == "full")
    assert any(row["FLOPs %"] for row in device_rows if row["Bound Analysis"] == "flops_only")


def test_signposts_have_no_op_category_and_do_not_warn(tmp_path, mocker):
    rows, _, stdout = _run_report(CONV_FIXTURE, tmp_path, mocker)

    signposts = [row for row in rows if "(signpost)" in row["OP Code"]]
    assert signposts, "fixture should contain signposts"
    assert all(row["Op Category"] == "" for row in signposts)
    assert all(row["Bound Analysis"] == "none" for row in signposts)
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


def test_stacked_report_reuses_the_per_op_category(tmp_path, mocker):
    rows, stacked_rows, _ = _run_report(CCL_FIXTURE, tmp_path, mocker, stacked=True)

    per_op = {_base_op(row): row["Op Category"] for row in rows}
    for row in stacked_rows:
        assert row["Op Category"] == per_op[row["Op Code"]], row["Op Code"]


def test_category_sort_follows_category_order():
    stacked_df = pd.DataFrame(
        {
            "Op_Category": ["Other", "CCL", "Host", "TM", "Compute", "DM"],
            "Device_Time_Sum_us": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    result = _sort_dataframe_by_category(stacked_df)

    assert list(result["Op_Category"]) == CATEGORY_ORDER
    assert "category_sort" not in result.columns


def test_unknown_op_names_are_sanitised_before_the_warning(tmp_path, mocker):
    csv_path = tmp_path / "hostile.csv"
    with open(CCL_FIXTURE, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        first = next(reader)
    first["OP CODE"] = "\x1b]0;pwned\x07Evil\x1b[2JDeviceOperation"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(first)

    rows, _, stdout = _run_report(str(csv_path), tmp_path, mocker)

    assert "\x1b" not in stdout and "\x07" not in stdout
    assert "Unclassified operation" in stdout
    assert rows[0]["Op Category"] == "Other"


def _matmul_row(dram_percentage, flops_percentage):
    return {
        "OP Code": Cell("MatmulDeviceOperation 32 x 64 x 128"),
        "Bound Analysis": Cell("full"),
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
        (65.0, 0.0, "DRAM"),
        (0.0, 65.0, "FLOP"),
        (65.0, 65.0, "BOTH"),
        (64.9, 64.9, "SLOW"),
    ],
)
def test_zero_percent_matmul_is_measured_not_missing(dram_percentage, flops_percentage, expected):
    rows = [_matmul_row(dram_percentage, flops_percentage)]
    add_derived_columns(rows)

    assert rows[0]["Bound"].raw_value == expected


def test_host_ops_are_categorised_host_without_warning(tmp_path, mocker):
    csv_path = tmp_path / "host.csv"
    with open(CCL_FIXTURE, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        device_row = next(reader)
    host_row = dict(device_row, **{"OP CODE": "ttnn.to_torch (torch)"})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(device_row)
        writer.writerow(host_row)

    rows, _, stdout = _run_report(str(csv_path), tmp_path, mocker)

    host = next(row for row in rows if "(torch)" in row["OP Code"])
    assert host["Bound"] == "HOST"
    assert host["Bound Analysis"] == "none"
    assert host["Op Category"] == "Host"
    assert "Unclassified operation 'ttnn.to_torch'" not in stdout
