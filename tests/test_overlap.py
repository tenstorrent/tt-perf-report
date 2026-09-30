# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import glob
import os

import numpy as np
import pandas as pd
import pytest

from tt_perf_report.overlap import (
    OVERLAP_BUSY_COLUMN,
    OVERLAP_COLUMN,
    annotate_overlap,
)
from tt_perf_report.perf_report import is_invalid_device_duration, valid_device_duration_mask

from overlap_timing import fw_timing

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _op(start_us, duration_us, device_id=0, op_type="tt_dnn_device", fw_overhead_us=1):
    return {
        "DEVICE ID": device_id,
        "OP TYPE": op_type,
        **fw_timing(start_us, duration_us, fw_overhead_us),
    }


def _annotate(ops, valid=None):
    df = pd.DataFrame(ops)
    out = annotate_overlap(df, valid if valid is not None else [True] * len(df))
    return list(out[OVERLAP_COLUMN] / 1000), list(out[OVERLAP_BUSY_COLUMN] / 1000)


def test_two_concurrent_ops_count_once():
    # The issue's example: two 100 μs ops starting together took 100 μs.
    overlap, busy = _annotate([_op(1000, 100), _op(1000, 100)])

    assert overlap == pytest.approx([0, 100])
    assert sum(busy) == pytest.approx(100)


@pytest.mark.parametrize("second,expected_overlap", [
    ((1100, 50), 0),      # back to back
    ((1200, 50), 0),      # disjoint
    ((1020, 50), 50),     # contained
    ((1080, 50), 20),     # partial
])
def test_overlap_is_the_part_already_covered(second, expected_overlap):
    overlap, busy = _annotate([_op(1000, 100), _op(*second)])

    assert overlap == pytest.approx([0, expected_overlap])
    assert busy == pytest.approx([100, 50 - expected_overlap])


def test_an_op_contained_in_an_earlier_one_does_not_extend_its_cover():
    # The third op starts after the contained one ends but inside the first.
    overlap, _ = _annotate([_op(1000, 100), _op(1010, 10), _op(1050, 100)])

    assert overlap == pytest.approx([0, 10, 50])


def test_row_order_does_not_matter():
    # Tracing mode leaves rows unsorted, so intervals are ordered by start.
    overlap, busy = _annotate([_op(1080, 50), _op(1000, 100)])

    assert overlap == pytest.approx([20, 0])
    assert sum(busy) == pytest.approx(130)


def test_devices_are_never_compared_with_each_other():
    # Device clocks are not synchronised, so equal cycle counts on two devices
    # say nothing about whether the ops ran at the same moment.
    overlap, _ = _annotate([_op(1000, 100, device_id=0), _op(1000, 100, device_id=1)])

    assert overlap == [0, 0]


def test_missing_cycle_columns_leave_overlap_unknown():
    ops = [_op(1000, 100), _op(1000, 100)]
    for op in ops:
        del op["DEVICE FW END CYCLE"]

    overlap, busy = _annotate(ops)

    assert all(np.isnan(overlap)) and all(np.isnan(busy))


def test_invalid_durations_and_non_device_rows_are_excluded():
    ops = [_op(1000, 100), _op(1000, 100), _op(1000, 100, op_type="signpost")]

    overlap, busy = _annotate(ops, valid=[False, True, True])

    assert np.isnan(overlap[0]) and np.isnan(busy[0])
    assert overlap[1] == 0
    assert np.isnan(overlap[2])


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))), ids=os.path.basename)
def test_captured_sequential_runs_report_no_overlap(path):
    # Pins the interval anchoring. Raw FW spans overlap on most of these
    # captures; kernel intervals ending at FW end must not.
    df = pd.read_csv(path)
    out = annotate_overlap(df, valid_device_duration_mask(df))

    assert out[OVERLAP_COLUMN].notna().any()
    assert (out[OVERLAP_COLUMN].fillna(0) == 0).all()


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))), ids=os.path.basename)
def test_valid_duration_mask_matches_the_per_row_rule_on_captures(path):
    df = pd.read_csv(path)

    expected = [not is_invalid_device_duration(row) for _, row in df.iterrows()]
    assert list(valid_device_duration_mask(df)) == expected


def test_valid_duration_mask_matches_the_per_row_rule_on_malformed_cells():
    # Text columns take the cell-by-cell path, so each of these must be judged
    # exactly as finite_float judges it.
    df = pd.DataFrame({
        "DEVICE KERNEL DURATION [ns]": ["1000", "", "abc", "inf", "-5", "1_000", " 7 ", "3000000", "3000000", None],
        "OP TO OP LATENCY [ns]": ["0", "0", "0", "0", "0", "0", "0", "-2000000", "-1000000", "0"],
    })

    expected = [not is_invalid_device_duration(row) for _, row in df.iterrows()]
    assert list(valid_device_duration_mask(df)) == expected
    assert expected == [True, False, False, False, False, True, True, False, True, False]


def test_valid_duration_mask_without_the_columns_marks_every_row_invalid():
    df = pd.DataFrame({"OP CODE": ["a", "b"]})

    assert list(valid_device_duration_mask(df)) == [False, False]
