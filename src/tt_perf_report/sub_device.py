# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Subdevice fields carried by op perf reports.

Models are starting to partition a chip's worker grid into subdevices so that
two ops can run concurrently on disjoint core ranges. When that happens the
report carries a per-op subdevice id and a per-op worker core budget, and an
op confined to a subdevice must be measured against that budget rather than
against the whole chip.

This module reads those two fields per row. Coercion policy lives in
csv_values; this module knows nothing about architectures, and callers pass the
fallback core count they want used when a row carries no usable budget.
"""

import pandas as pd

from tt_perf_report.csv_values import (
    AVAILABLE_WORKER_CORE_COUNT_COLUMN,
    finite_float,
    get_core_count,
)

SUB_DEVICE_ID_COLUMN = "SUB DEVICE ID"


def get_op_sub_device_id(row):
    """
    Get the subdevice this op ran on, or None when the op used the full grid.

    A blank value is meaningful rather than missing: it denotes the full worker
    grid, so it is reported as an empty cell rather than as a subdevice.

    Anything that is not a whole number is passed through as text rather than
    normalised, so that unexpected profiler output stays visible in the report.
    In particular a fractional id is not truncated: reading "1.5" as subdevice 1
    would silently merge a malformed row into a real subdevice.
    """
    if SUB_DEVICE_ID_COLUMN not in row:
        return None

    value = row[SUB_DEVICE_ID_COLUMN]
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    number = finite_float(text)
    if number is None or not number.is_integer():
        return text

    return int(number)


def get_op_available_cores(row, fallback_cores: int) -> int:
    """
    Get the worker cores available to this op's subdevice.

    Falls back to fallback_cores - normally the architecture's full grid - when
    the CSV predates the column, or carries no usable value for this row.
    """
    return get_core_count(row, AVAILABLE_WORKER_CORE_COUNT_COLUMN, default=fallback_cores)


def count_sub_devices(sub_device_ids) -> int:
    """
    Number of distinct non-blank subdevice ids among already-extracted ids.

    Takes ids produced by get_op_sub_device_id rather than a DataFrame, so that
    there is exactly one place that decides what a subdevice id is and what
    counts as blank. Unparseable ids count, deliberately: a capture full of
    garbage ids should surface the column rather than read as a full-grid run.
    """
    return len({sub_device_id for sub_device_id in sub_device_ids if sub_device_id is not None})
