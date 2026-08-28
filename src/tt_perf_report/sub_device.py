# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Subdevice fields carried by op perf reports.

Models are starting to partition a chip's worker grid into subdevices so that
two ops can run concurrently on disjoint core ranges. When that happens the
report carries a per-op subdevice id and a per-op worker core budget, and an
op confined to a subdevice must be measured against that budget rather than
against the whole chip.

This module owns the CSV column names and the per-row extraction. It knows
nothing about architectures: callers pass the fallback core count they want
used when a row carries no usable budget of its own.
"""

import math

import pandas as pd

AVAILABLE_WORKER_CORE_COUNT_COLUMN = "AVAILABLE WORKER CORE COUNT"
SUB_DEVICE_ID_COLUMN = "SUB DEVICE ID"


def get_op_sub_device_id(row):
    """
    Get the subdevice this op ran on, or None when the op used the full grid.

    A blank value is meaningful rather than missing: it denotes the full worker
    grid, so it is reported as an empty cell rather than as a subdevice.

    An id that is not a finite number is passed through as text rather than
    discarded, so that unexpected profiler output is visible in the report
    instead of silently reading as a full-grid op.
    """
    if SUB_DEVICE_ID_COLUMN not in row:
        return None

    value = row[SUB_DEVICE_ID_COLUMN]
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return text

    if not math.isfinite(number):
        return text

    return int(number)


def get_op_available_cores(row, fallback_cores: int) -> int:
    """
    Get the worker cores available to this op's subdevice.

    Falls back to fallback_cores - normally the architecture's full grid - when
    the CSV predates the column, or carries no usable value for this row.
    """
    if AVAILABLE_WORKER_CORE_COUNT_COLUMN not in row:
        return fallback_cores

    value = row[AVAILABLE_WORKER_CORE_COUNT_COLUMN]
    if pd.isna(value):
        return fallback_cores

    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback_cores

    if not math.isfinite(number) or number <= 0:
        return fallback_cores

    return int(number)


def count_sub_devices(sub_device_ids) -> int:
    """
    Number of distinct real subdevices among already-extracted ids.

    Takes ids produced by get_op_sub_device_id rather than a DataFrame, so that
    there is exactly one place that decides what a subdevice id is and what
    counts as blank.
    """
    return len({sub_device_id for sub_device_id in sub_device_ids if sub_device_id is not None})
