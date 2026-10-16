# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Wall-clock overlap between ops on the same device.

Every total in the report adds op durations together, which is only honest when
ops ran one after another. Subdevices exist so that they need not: two ops on
disjoint core ranges can run concurrently, and summing their durations then
overcounts the time the device was actually busy.

This module measures that from the profiler's cycle counters. Three properties
of real captures decide how:

- The raw [DEVICE FW START CYCLE, DEVICE FW END CYCLE] span is a loose envelope.
  On ordinary sequential runs consecutive spans routinely overlap by several
  microseconds while OP TO OP LATENCY stays positive, so a union of those spans
  would report overlap on every capture. Each op is instead given the interval
  [FW end - kernel duration, FW end]. That anchoring is empirical rather than
  documented by the profiler, but it yields no overlap on any sequential
  capture in tests/data.
- Cycle counters are per device and are not synchronised across devices, so
  intervals are only ever compared within one DEVICE ID.
- The cycles-per-ns ratio is recovered from the file itself, as the median of
  FW cycles over DEVICE FW DURATION [ns], so no clock frequency is assumed.

The result is attributed per row - the part of each op's interval not already
covered by an earlier interval on its device - so that summing it over the rows
merge_device_rows picks still gives busy time without needing the intervals
again. That holds only while the ops doing the covering stay in the report: a
filter that drops them must rerun the sweep without them.
"""

import numpy as np

from tt_perf_report.csv_values import finite_float_column

FW_START_CYCLE_COLUMN = "DEVICE FW START CYCLE"
FW_END_CYCLE_COLUMN = "DEVICE FW END CYCLE"
FW_DURATION_COLUMN = "DEVICE FW DURATION [ns]"
KERNEL_DURATION_COLUMN = "DEVICE KERNEL DURATION [ns]"

OVERLAP_BUSY_COLUMN = "OVERLAP BUSY [ns]"
OVERLAP_COLUMN = "OVERLAP [ns]"

# Below one cycle at any supported clock. Intervals are rebuilt from a cycle
# count and a separately rounded nanosecond duration, so back-to-back ops can
# appear to touch by a fraction of a nanosecond; that is rounding, not overlap.
MIN_OVERLAP_NS = 1.0

# Every column measure_overlap reads; without all of them nothing is measured.
OVERLAP_INPUT_COLUMNS = (
    "DEVICE ID",
    "OP TYPE",
    FW_START_CYCLE_COLUMN,
    FW_END_CYCLE_COLUMN,
    FW_DURATION_COLUMN,
    KERNEL_DURATION_COLUMN,
)


def _cycles_per_ns(fw_start, fw_end, fw_duration_ns):
    usable = (fw_end > fw_start) & (fw_duration_ns > 0)
    if not usable.any():
        return None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = float(np.median((fw_end[usable] - fw_start[usable]) / fw_duration_ns[usable]))
    return ratio if np.isfinite(ratio) and ratio > 0 else None


def split_duration(duration_ns, overlap_ns):
    """
    Return (overlap, busy) for an op: overlap capped at the op's duration, and
    busy as the rest. An op cannot have overlapped for longer than it ran.
    """
    overlap_ns = min(overlap_ns, duration_ns)
    return overlap_ns, duration_ns - overlap_ns


def measure_overlap(df, valid_duration):
    """
    Return (busy, overlap) in ns per row of df, as float arrays aligned with it.

    valid_duration is a boolean sequence aligned with df's rows, false where the
    row's kernel duration is not to be trusted; the caller owns that rule. Rows
    that are not device ops, have an invalid duration, or lack usable cycle
    data get NaN in both, as does every row when an input column is absent.
    For the rest, busy is the part of the op's interval no earlier interval on
    the same device covered, and overlap is the remainder of its kernel
    duration. Cells are read through finite_float_column, so a duration counts
    as a number here exactly when it does everywhere else in the report.
    """
    valid_duration = np.asarray(valid_duration, dtype=bool)
    if valid_duration.shape != (len(df),):
        raise ValueError(f"valid_duration has {valid_duration.size} entries for {len(df)} rows")

    busy = np.full(len(df), np.nan)
    overlap = np.full(len(df), np.nan)
    if not all(column in df.columns for column in OVERLAP_INPUT_COLUMNS):
        return busy, overlap

    device_ids = finite_float_column(df, "DEVICE ID")
    fw_start = finite_float_column(df, FW_START_CYCLE_COLUMN)
    fw_end = finite_float_column(df, FW_END_CYCLE_COLUMN)
    fw_duration_ns = finite_float_column(df, FW_DURATION_COLUMN)
    kernel_ns = finite_float_column(df, KERNEL_DURATION_COLUMN)
    with np.errstate(invalid="ignore"):
        usable = (
            (df["OP TYPE"].to_numpy() == "tt_dnn_device")
            & valid_duration
            & ~np.isnan(device_ids)
            & ~np.isnan(fw_start)
            & ~np.isnan(fw_end)
            & ~np.isnan(kernel_ns)
            & (kernel_ns >= 0)
        )

    for device_id in np.unique(device_ids[usable]):
        on_device = usable & (device_ids == device_id)
        cycles_per_ns = _cycles_per_ns(fw_start[on_device], fw_end[on_device], fw_duration_ns[on_device])
        if cycles_per_ns is None:
            continue

        positions = np.flatnonzero(on_device)
        # A crafted file can push these past the float range. The result stays
        # bounded - each op's overlap is capped at its own duration - so only
        # the warnings are silenced, not the arithmetic changed.
        with np.errstate(over="ignore", invalid="ignore"):
            ends = fw_end[positions] / cycles_per_ns
            starts = ends - kernel_ns[positions]
            covered_until = -np.inf
            for index in np.argsort(starts, kind="stable"):
                row_kernel_ns = kernel_ns[positions[index]]
                row_overlap, row_busy = split_duration(row_kernel_ns, max(0.0, covered_until - starts[index]))
                if row_overlap < MIN_OVERLAP_NS:
                    row_overlap, row_busy = 0.0, row_kernel_ns
                overlap[positions[index]], busy[positions[index]] = row_overlap, row_busy
                covered_until = max(covered_until, ends[index])

    return busy, overlap


def annotate_overlap(df, valid_duration):
    """
    Add OVERLAP BUSY [ns] and OVERLAP [ns] to df in place, from measure_overlap.

    In place because df is the whole pre-merge frame: on pandas without
    copy-on-write, assign() deep-copies every column to add these two. insert()
    neither copies nor trips the chained-assignment warning on a frame that a
    signpost filter sliced.
    """
    busy, overlap = measure_overlap(df, valid_duration)
    for column, values in ((OVERLAP_BUSY_COLUMN, busy), (OVERLAP_COLUMN, overlap)):
        if column in df.columns:
            del df[column]
        df.insert(len(df.columns), column, values)
