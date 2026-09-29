#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import csv
import os
from io import StringIO

import pandas as pd
import pytest

from tt_perf_report.perf_report import (
    CATEGORY_ORDER,
    MAX_WARNED_OP_NAME_CHARS,
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

HOST_OP_CODE = "ttnn.to_torch (torch)"
ROOFLINE_BOUNDS = {"BOTH", "DRAM", "FLOP", "SLOW"}

pytestmark = pytest.mark.usefixtures("reset_classification_cache")


def _run_report(csv_path, tmp_path, mocker, group_by="op", stacked=False, no_host_ops=False):
    stdout = mocker.patch("sys.stdout", new_callable=StringIO)
    # The stacked CSV is written before the chart, and rendering it is the slowest step.
    mocker.patch("tt_perf_report.perf_report.plot_stacked_report")
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


def _write_variant(tmp_path, source, *op_codes):
    """Write `source`'s header plus one copy of its first row per op code (None keeps the original)."""
    with open(source, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        base_row = next(reader)
    path = tmp_path / "variant.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for op_code in op_codes:
            writer.writerow(base_row if op_code is None else dict(base_row, **{"OP CODE": op_code}))
    return str(path)


def _first_row_of(source, base_op):
    with open(source, newline="") as f:
        return next(row for row in csv.DictReader(f) if row["OP CODE"].split()[0] == base_op)


def _base_op(row):
    return row["Raw OP Code"].split()[0]


def _is_signpost(row):
    return "(signpost)" in row["OP Code"]


# Pinned by name rather than re-deriving the production branch conditions, so a change to how
# analyze_op dispatches fails here instead of being mirrored.
EXPECTED_BOUND_ANALYSIS = {
    "Matmul": "full",
    "OptimizedConvNew": "flops_only",
    "HaloDeviceOperation": "none",
    "MoveDeviceOperation": "none",
    "GroupNorm": "none",
}


def test_bound_analysis_names_the_model_that_ran(tmp_path, mocker):
    rows, _, _ = _run_report(CONV_FIXTURE, tmp_path, mocker)

    by_op = {}
    for row in rows:
        by_op.setdefault(_base_op(row), set()).add(row["Bound Analysis"])
    for op, expected in EXPECTED_BOUND_ANALYSIS.items():
        assert by_op[op] == {expected}, op


def test_conv2d_gets_flops_only_analysis(tmp_path, mocker):
    # The fixture only carries OptimizedConvNew; Conv2d is its current name and takes the same branch.
    conv_row = _first_row_of(CONV_FIXTURE, "OptimizedConvNew")
    csv_path = tmp_path / "conv2d.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(conv_row))
        writer.writeheader()
        writer.writerow(dict(conv_row, **{"OP CODE": conv_row["OP CODE"].replace("OptimizedConvNew", "Conv2d", 1)}))

    rows, _, _ = _run_report(str(csv_path), tmp_path, mocker)

    assert rows[0]["Bound Analysis"] == "flops_only"
    assert (rows[0]["DRAM %"], rows[0]["Bound"]) == ("", "")


def test_bound_analysis_matches_which_figures_are_present(tmp_path, mocker):
    rows, _, _ = _run_report(CONV_FIXTURE, tmp_path, mocker)
    device_rows = [row for row in rows if not _is_signpost(row)]

    for row in device_rows:
        if row["Bound Analysis"] == "none":
            assert (row["DRAM %"], row["FLOPs %"], row["Bound"]) == ("", "", ""), row["Raw OP Code"]
        elif row["Bound Analysis"] == "flops_only":
            assert (row["DRAM %"], row["Bound"]) == ("", ""), row["Raw OP Code"]
        elif row["DRAM %"] and row["FLOPs %"]:
            assert row["Bound"] in ROOFLINE_BOUNDS, row["Raw OP Code"]

    # Guards the other direction: blanking every figure would satisfy the loop above.
    assert any(row["DRAM %"] and row["FLOPs %"] for row in device_rows if row["Bound Analysis"] == "full")
    assert any(row["FLOPs %"] for row in device_rows if row["Bound Analysis"] == "flops_only")


def test_signposts_have_no_op_category_and_do_not_warn(tmp_path, mocker):
    rows, _, stdout = _run_report(CONV_FIXTURE, tmp_path, mocker)

    signposts = [row for row in rows if _is_signpost(row)]
    assert signposts, "fixture should contain signposts"
    assert all(row["Op Category"] == "" for row in signposts)
    assert all(row["Bound Analysis"] == "none" for row in signposts)
    for row in signposts:
        assert f"Unclassified operation '{_base_op(row)}'" not in stdout

    assert all(row["Op Category"] in CATEGORY_ORDER for row in rows if not _is_signpost(row))


def test_async_collectives_are_classified_ccl_per_op(tmp_path, mocker):
    rows, _, _ = _run_report(CCL_FIXTURE, tmp_path, mocker)

    categories = {row["Raw OP Code"]: row["Op Category"] for row in rows}
    assert categories["AllGatherAsyncDeviceOperation"] == "CCL"
    assert categories["ReduceScatterMinimalAsyncDeviceOperation"] == "CCL"
    assert categories["MatmulDeviceOperation"] == "Compute"


def test_stacked_report_grouped_by_category_uses_only_known_categories(tmp_path, mocker):
    _, stacked_rows, _ = _run_report(CCL_FIXTURE, tmp_path, mocker, group_by="category", stacked=True)

    groups = {row["Op Code"] for row in stacked_rows}
    assert "CCL" in groups
    assert groups <= set(CATEGORY_ORDER)


def test_stacked_report_reuses_the_per_op_category(tmp_path, mocker):
    rows, stacked_rows, _ = _run_report(CCL_FIXTURE, tmp_path, mocker, stacked=True)

    per_op = {_base_op(row): row["Op Category"] for row in rows}
    for row in stacked_rows:
        assert row["Op Category"] == per_op[row["Op Code"]], row["Op Code"]


@pytest.mark.parametrize("category_column", ["Op_Category", "OP Code Joined"])
def test_category_sort_follows_category_order(category_column):
    # Op_Category orders bars grouped by op; OP Code Joined orders bars that are categories.
    stacked_df = pd.DataFrame(
        {
            category_column: ["Other", "CCL", "Host", "TM", "Compute", "DM"],
            "Device_Time_Sum_us": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    result = _sort_dataframe_by_category(stacked_df, category_column=category_column)

    assert list(result[category_column]) == CATEGORY_ORDER
    assert "category_sort" not in result.columns


def test_unknown_op_names_are_sanitised_before_the_warning(tmp_path, mocker):
    csv_path = _write_variant(tmp_path, CCL_FIXTURE, "\x1b]0;pwned\x07Evil\x1b[2JDeviceOperation")

    rows, _, stdout = _run_report(csv_path, tmp_path, mocker)

    # Assert on the payload itself, not on "\x1b" in general, which colour output would also emit.
    assert "\x1b]0;" not in stdout and "\x1b[2J" not in stdout and "\x07" not in stdout
    assert "Unclassified operation" in stdout
    assert rows[0]["Op Category"] == "Other"


def test_unclassified_warning_truncates_long_op_names(tmp_path, mocker):
    long_name = "X" * (MAX_WARNED_OP_NAME_CHARS * 100)
    csv_path = _write_variant(tmp_path, CCL_FIXTURE, long_name)

    rows, _, stdout = _run_report(csv_path, tmp_path, mocker)

    warning = next(line for line in stdout.splitlines() if "Unclassified operation" in line)
    assert f"'{'X' * MAX_WARNED_OP_NAME_CHARS}…'" in warning
    assert "X" * (MAX_WARNED_OP_NAME_CHARS + 1) not in stdout
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
    csv_path = _write_variant(tmp_path, CCL_FIXTURE, None, HOST_OP_CODE)

    rows, _, stdout = _run_report(csv_path, tmp_path, mocker)

    host = next(row for row in rows if "(torch)" in row["OP Code"])
    assert host["Bound"] == "HOST"
    assert host["Bound Analysis"] == "none"
    assert host["Op Category"] == "Host"
    assert "Unclassified operation 'ttnn.to_torch'" not in stdout


def test_stacked_report_groups_host_ops_as_host(tmp_path, mocker):
    csv_path = _write_variant(tmp_path, CCL_FIXTURE, None, HOST_OP_CODE)

    _, stacked_rows, stdout = _run_report(csv_path, tmp_path, mocker, group_by="category", stacked=True)

    groups = {row["Op Code"] for row in stacked_rows}
    assert "Host" in groups
    assert groups <= set(CATEGORY_ORDER)
    assert "Unclassified operation 'ttnn.to_torch'" not in stdout


def test_no_host_ops_removes_host_rows_from_both_outputs(tmp_path, mocker):
    csv_path = _write_variant(tmp_path, CCL_FIXTURE, None, HOST_OP_CODE)

    rows, stacked_rows, _ = _run_report(csv_path, tmp_path, mocker, stacked=True, no_host_ops=True)

    assert rows and not any("(torch)" in row["OP Code"] for row in rows)
    assert stacked_rows and not any(row["Op Category"] == "Host" for row in stacked_rows)
    assert not any("(torch)" in row["Op Code"] for row in stacked_rows)
