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

import pandas as pd

AVAILABLE_WORKER_CORE_COUNT_COLUMN = "AVAILABLE WORKER CORE COUNT"

# Subdevice id column, present only in the widest op perf reports. Accept the
# obvious spelling variant so a rename upstream degrades to an empty column
# rather than to silently dropping the field.
SUB_DEVICE_ID_COLUMNS = ("SUB DEVICE ID", "SUBDEVICE ID")


def get_op_sub_device_id(row):
    """
    Get the subdevice this op ran on, or None when the op used the full grid.

    A blank value is meaningful rather than missing: it denotes the full worker
    grid, so it is reported as an empty cell rather than as a subdevice.
    """
    for column in SUB_DEVICE_ID_COLUMNS:
        if column not in row:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return int(float(text))
        except ValueError:
            return text
    return None


def get_op_available_cores(row, fallback_cores: int) -> int:
    """
    Get the worker cores available to this op's subdevice.

    Falls back to fallback_cores - normally the architecture's full grid - when
    the CSV predates the column, or carries no usable value for this row.
    """
    if AVAILABLE_WORKER_CORE_COUNT_COLUMN in row:
        value = row[AVAILABLE_WORKER_CORE_COUNT_COLUMN]
        if not pd.isna(value):
            try:
                count = int(float(value))
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                return count
    return fallback_cores


def count_sub_devices(df) -> int:
    """Number of distinct non-blank subdevice ids in the report."""
    for column in SUB_DEVICE_ID_COLUMNS:
        if column in df.columns:
            values = df[column].dropna()
            values = values[values.astype(str).str.strip() != ""]
            return values.nunique()
    return 0
