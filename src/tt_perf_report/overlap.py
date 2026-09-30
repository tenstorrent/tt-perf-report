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
import pandas as pd

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

_REQUIRED_COLUMNS = (
    "DEVICE ID",
    "OP TYPE",
    FW_START_CYCLE_COLUMN,
    FW_END_CYCLE_COLUMN,
    FW_DURATION_COLUMN,
    KERNEL_DURATION_COLUMN,
)


def _numeric(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    values[~np.isfinite(values)] = np.nan
    return values


def _cycles_per_ns(fw_start, fw_end, fw_duration_ns):
    usable = (fw_end > fw_start) & (fw_duration_ns > 0)
    if not usable.any():
        return None
    ratio = float(np.median((fw_end[usable] - fw_start[usable]) / fw_duration_ns[usable]))
    return ratio if np.isfinite(ratio) and ratio > 0 else None


def annotate_overlap(df, valid_duration):
    """
    Return df with OVERLAP BUSY [ns] and OVERLAP [ns] per row; df is not modified.

    valid_duration is a boolean sequence aligned with df's rows, false where the
    row's kernel duration is not to be trusted; the caller owns that rule. Rows
    that are not device ops, have an invalid duration, or lack usable cycle
    data get NA in both columns, as does every row when a required column is
    absent. For the rest, busy is the part of the op's interval no earlier
    interval on the same device covered, and overlap is the remainder of its
    kernel duration.
    """
    busy = np.full(len(df), np.nan)
    overlap = np.full(len(df), np.nan)

    if all(column in df.columns for column in _REQUIRED_COLUMNS):
        device_ids = _numeric(df["DEVICE ID"])
        fw_start = _numeric(df[FW_START_CYCLE_COLUMN])
        fw_end = _numeric(df[FW_END_CYCLE_COLUMN])
        fw_duration_ns = _numeric(df[FW_DURATION_COLUMN])
        kernel_ns = _numeric(df[KERNEL_DURATION_COLUMN])
        usable = (
            (df["OP TYPE"].to_numpy() == "tt_dnn_device")
            & np.asarray(valid_duration, dtype=bool)
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
            ends = fw_end[positions] / cycles_per_ns
            starts = ends - kernel_ns[positions]
            covered_until = -np.inf
            for index in np.argsort(starts, kind="stable"):
                row_kernel_ns = kernel_ns[positions[index]]
                row_overlap = min(row_kernel_ns, max(0.0, covered_until - starts[index]))
                if row_overlap < MIN_OVERLAP_NS:
                    row_overlap = 0.0
                busy[positions[index]] = row_kernel_ns - row_overlap
                overlap[positions[index]] = row_overlap
                covered_until = max(covered_until, ends[index])

    # assign rather than copy-and-set: the pre-merge frame is wide, and only
    # these two columns are new.
    return df.assign(**{OVERLAP_BUSY_COLUMN: busy, OVERLAP_COLUMN: overlap})
