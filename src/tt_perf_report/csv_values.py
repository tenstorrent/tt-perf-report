# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Coercion of individual cells from an op perf report.

The CSV is untrusted input: a cell may be absent, blank, textual, or a value
pandas parses into something no consumer wants, such as infinity. Every numeric
read goes through here so that one malformed cell degrades that one value
rather than aborting the report.
"""

import math

import pandas as pd

# Column carrying the worker grid available to each op, added in the v2.1 op
# perf report format.
AVAILABLE_WORKER_CORE_COUNT_COLUMN = "AVAILABLE WORKER CORE COUNT"


def get_numeric_value(row, column):
    """
    Coerce one cell to a finite float, or None when it is absent, blank or
    unusable.

    Infinity is rejected rather than returned: `pd.isna` does not catch it, and
    every caller ultimately wants a duration or a core count, so passing it on
    only moves the failure to an `int()` conversion further downstream.
    """
    if column not in row:
        return None
    value = row[column]
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def get_positive_int(row, column, default=None):
    """Coerce one cell to a positive int, or default when it is not one."""
    number = get_numeric_value(row, column)
    if number is None:
        return default
    # Truncate before testing: a fractional value below 1 must fall back rather
    # than silently become a zero core count.
    truncated = int(number)
    return truncated if truncated > 0 else default


def get_value_physical_logical(value, is_physical: bool = True):
    """
    Read a tensor dimension cell, which newer reports write as "physical[logical]".
    """
    # Handle numeric inputs (old format)
    if isinstance(value, (int, float)):
        return int(value)

    # Handle string inputs (new format)
    if isinstance(value, str) and "[" in value and "]" in value:
        physical_part = value.split("[")[0]
        logical_part = value.split("[")[1].split("]")[0]

        if is_physical:
            return int(physical_part)
        else:
            return int(logical_part)
    else:
        # backwards compatibility - convert string to int
        return int(value)
