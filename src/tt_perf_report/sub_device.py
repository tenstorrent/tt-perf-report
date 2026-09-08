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
    get_core_count,
    sanitize_text,
    whole_number,
)

SUB_DEVICE_ID_COLUMN = "SUB DEVICE ID"

# An id that is not a whole number is reported as text, so its length is the
# CSV's to choose. The terminal table sizes each column to its widest cell, so
# one long id would set the width of the whole column for every row and push the
# rest of the table off screen. No real id needs this many characters; a value
# that does is corrupt, and enough of it survives to show that.
MAX_SUB_DEVICE_ID_LENGTH = 32


def get_op_sub_device_id(row):
    """
    Get the subdevice this op ran on, or None when there is no id to report.

    None is deliberately ambiguous, because the CSV cannot distinguish the two
    cases it covers. When the column is present, a blank cell is meaningful
    rather than missing: it denotes the full worker grid. When the column is
    absent - a capture predating it - no row has an id at all, and None means
    unknown, even on a run whose per-op budgets show the chip was partitioned.
    Callers that need to tell the two apart must test for the column itself;
    count_sub_devices collapses both to "no subdevices reported" on purpose,
    since neither gives the reader a subdevice to look at.

    Anything that is not a whole number is passed through as text rather than
    normalised, so that unexpected profiler output stays visible in the report.
    In particular a fractional id is not truncated: reading "1.5" as subdevice 1
    would silently merge a malformed row into a real subdevice. Such text has
    its control characters stripped, and is clipped to MAX_SUB_DEVICE_ID_LENGTH,
    which keeps it visible without letting one corrupt cell break a table row or
    size the column for every row.
    """
    if SUB_DEVICE_ID_COLUMN not in row:
        return None

    value = row[SUB_DEVICE_ID_COLUMN]
    if pd.isna(value):
        return None

    text = sanitize_text(str(value)).strip()
    if not text:
        return None

    number = whole_number(text)
    if number is None:
        return text[:MAX_SUB_DEVICE_ID_LENGTH]

    return number


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
