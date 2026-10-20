# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import glob
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from tt_perf_report.overlap import (
    OVERLAP_BUSY_COLUMN,
    OVERLAP_COLUMN,
    annotate_overlap,
    measure_overlap,
    split_duration,
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
    busy, overlap, _ = measure_overlap(df, valid if valid is not None else [True] * len(df))
    return list(overlap / 1000), list(busy / 1000)


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


@pytest.mark.parametrize("overlap_ns,expected_ns", [
    (0.5, 0),     # rounding when intervals are rebuilt from cycles: not overlap
    (0.99, 0),
    (1.0, 1.0),   # one cycle or more is kept
    (2.0, 2.0),
])
def test_sub_nanosecond_touches_are_rounding_not_overlap(overlap_ns, expected_ns):
    # One cycle per ns, so each interval is exactly [end - kernel, end] and the
    # overlap is exactly what the test asks for.
    def op(end_ns, kernel_ns=5000):
        return {
            "DEVICE ID": 0,
            "OP TYPE": "tt_dnn_device",
            "DEVICE FW START CYCLE": end_ns - kernel_ns - 1000,
            "DEVICE FW END CYCLE": end_ns,
            "DEVICE FW DURATION [ns]": kernel_ns + 1000,
            "DEVICE KERNEL DURATION [ns]": kernel_ns,
        }
    df = pd.DataFrame([op(10_000), op(15_000 - overlap_ns)])

    _, overlap, _ = measure_overlap(df, [True, True])

    assert list(overlap) == pytest.approx([0, expected_ns])


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


def test_durations_are_read_as_the_rest_of_the_report_reads_them():
    # finite_float accepts "1_000", so the validity rule calls it a valid
    # duration; the sweep must measure it rather than silently skip it.
    ops = [_op(1000, 100), _op(1000, 100)]
    for op in ops:
        op["DEVICE KERNEL DURATION [ns]"] = "100_000"
    df = pd.DataFrame(ops)

    _, overlap, _ = measure_overlap(df, valid_device_duration_mask(df))

    assert list(overlap) == [0, 100_000]


@pytest.mark.parametrize("cell", ["x", "inf", "1e400", "-1", "0", "1e300"])
def test_malformed_cycle_cells_leave_overlap_bounded_or_unknown(cell):
    for column in ("DEVICE FW START CYCLE", "DEVICE FW END CYCLE", "DEVICE FW DURATION [ns]"):
        ops = [_op(1000, 100), _op(1000, 100)]
        ops[1][column] = cell
        df = pd.DataFrame(ops).astype({column: str})

        busy, overlap, _ = measure_overlap(df, [True, True])

        for row_overlap, row_busy in zip(overlap, busy):
            assert np.isnan(row_overlap) or 0 <= row_overlap <= 100_000
            assert np.isnan(row_busy) or 0 <= row_busy <= 100_000


@pytest.mark.parametrize("overlap_ns,expected", [(0, (0, 100)), (40, (40, 60)), (250, (100, 0))])
def test_split_duration_caps_overlap_at_the_duration(overlap_ns, expected):
    assert split_duration(100, overlap_ns) == expected


def test_overflowing_cycle_counts_stay_bounded_without_warnings():
    # A tiny cycles-per-ns ratio pushes every interval end past the float range.
    def op():
        return {
            "DEVICE ID": 0, "OP TYPE": "tt_dnn_device", "DEVICE FW START CYCLE": 1e-300,
            "DEVICE FW END CYCLE": 1e300, "DEVICE FW DURATION [ns]": 1e300, "DEVICE KERNEL DURATION [ns]": 5.0,
        }
    df = pd.DataFrame([op(), op()])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        busy, overlap, _ = measure_overlap(df, [True, True])

    assert all(np.isnan(value) or 0 <= value <= 5 for value in [*busy, *overlap])


def test_idle_time_is_the_uncovered_gap_before_an_op():
    # The third op starts 50 us after everything before it has finished; the
    # second's gap is covered by the first. The first op has nothing before it.
    df = pd.DataFrame([_op(1000, 300), _op(1100, 100), _op(1350, 50)])

    idle = measure_overlap(df, [True, True, True]).idle

    assert np.isnan(idle[0])
    assert list(idle[1:] / 1000) == pytest.approx([0, 50])


def test_a_reversed_fw_span_is_not_placed_in_time():
    # The third row's FW end precedes its start, so its counters contradict
    # each other; though its end lines up with the first op, it gets no overlap.
    ops = [_op(1000, 100), _op(1200, 100), _op(1000, 100)]
    ops[2]["DEVICE FW START CYCLE"] = ops[2]["DEVICE FW END CYCLE"] + 1
    busy, overlap, _ = measure_overlap(pd.DataFrame(ops), [True, True, True])

    assert np.isnan(overlap[2]) and np.isnan(busy[2])
    assert list(overlap[:2]) == [0, 0]


def test_valid_duration_must_match_the_rows():
    df = pd.DataFrame([_op(1000, 100), _op(1000, 100)])

    with pytest.raises(ValueError):
        measure_overlap(df, [True])


def test_annotate_overlap_adds_columns_in_place_on_a_filtered_frame():
    # The report annotates whatever filter_by_signpost returned, which may be a
    # boolean-indexed slice; adding columns there must neither copy the frame
    # nor raise pandas' chained-assignment warning.
    frame = pd.DataFrame([_op(1000, 100), _op(1000, 100, op_type="signpost"), _op(1000, 100)])
    window = frame[frame["OP TYPE"] != "signpost"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        annotate_overlap(window, [True, True])

    assert list(window[OVERLAP_COLUMN]) == [0, 100_000]
    assert list(window[OVERLAP_BUSY_COLUMN]) == [100_000, 0]


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))), ids=os.path.basename)
def test_captured_sequential_runs_report_no_overlap(path):
    # Pins the interval anchoring. Raw FW spans overlap on most of these
    # captures; kernel intervals ending at FW end must not.
    df = pd.read_csv(path)
    _, overlap, _ = measure_overlap(df, valid_device_duration_mask(df))

    assert not np.isnan(overlap).all()
    assert (np.nan_to_num(overlap) == 0).all()


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
