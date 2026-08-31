# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Coercion of individual cells from an op perf report.

The CSV is untrusted input: a cell may be absent, blank, textual, or a value
pandas parses into something no consumer wants - an infinity, a fraction where
only whole numbers make sense, or a magnitude no hardware could report. Every
numeric read in the tool goes through this module so that one malformed cell
costs that one value rather than aborting the report, and so that "what counts
as usable" is defined once instead of per call site.

Nothing here raises on bad input. A function that cannot produce a usable value
returns None (or the caller's default), and it is the caller's job to omit the
metric that depended on it.
"""

import math

import pandas as pd

AVAILABLE_WORKER_CORE_COUNT_COLUMN = "AVAILABLE WORKER CORE COUNT"

# No real report describes a worker grid this large. Rejecting implausible
# magnitudes keeps one corrupt cell from defining the whole file's grid, and
# matters doubly for a column-wide reduction, where a vectorised astype(int)
# saturates such a value to the int64 maximum instead of raising.
MAX_PLAUSIBLE_CORE_COUNT = 1 << 20


def finite_float(value):
    """
    Coerce a single value to a finite float, or None when it cannot be one.

    Infinity is rejected rather than returned: pd.isna does not catch it, and
    every caller ultimately wants a duration, a dimension or a core count, so
    passing it on only moves the failure to an int() conversion downstream.
    """
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def whole_number(value):
    """
    Coerce a single value to an int, but only when it is already a whole number.

    A fractional value is rejected rather than truncated. The quantities read
    through here - core counts, device ids, tensor dimensions - are all discrete,
    so a fraction means the cell is malformed, and truncating it would invent a
    plausible-looking value out of corrupt data.
    """
    number = finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def core_count_from_value(value, default=None):
    """Positive, whole, plausible core count from a single value, or default."""
    count = whole_number(value)
    if count is None or count <= 0 or count > MAX_PLAUSIBLE_CORE_COUNT:
        return default
    return count


def get_numeric_value(row, column):
    """Read one cell as a finite float, or None when absent or unusable."""
    if column not in row:
        return None
    return finite_float(row[column])


def get_int(row, column, default=None):
    """
    Read one cell as a whole int, or default.

    Zero and negatives are allowed: this is for quantities like DEVICE ID where
    zero is a legal value. Use get_core_count for counts that must be positive.
    """
    if column not in row:
        return default
    count = whole_number(row[column])
    return default if count is None else count


def get_core_count(row, column, default=None):
    """Read one cell as a positive, plausible core count, or default."""
    if column not in row:
        return default
    return core_count_from_value(row[column], default=default)


def get_value_physical_logical(value, is_physical: bool = True):
    """
    Read a tensor dimension cell, which newer reports write as "physical[logical]".

    Returns None when the cell holds no usable dimension, so that one malformed
    shape omits that op's modelled figures rather than aborting the report.
    """
    # Handle numeric inputs (old format)
    if isinstance(value, (int, float)):
        return whole_number(value)

    # Handle string inputs (new format)
    if isinstance(value, str) and "[" in value and "]" in value:
        physical_part = value.split("[")[0]
        logical_part = value.split("[")[1].split("]")[0]
        return whole_number(physical_part if is_physical else logical_part)

    # backwards compatibility - a bare value, numeric or as text
    return whole_number(value)
