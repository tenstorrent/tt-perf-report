#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest

from tt_perf_report.csv_values import (
    MAX_PLAUSIBLE_CORE_COUNT,
    MAX_PLAUSIBLE_TENSOR_DIM,
    core_count_from_value,
    finite_float,
    get_core_count,
    get_int,
    get_numeric_value,
    get_value_physical_logical,
    tensor_dim_from_value,
    whole_number,
)


@pytest.mark.parametrize("value,expected", [
    (108, 108.0),
    ("108", 108.0),
    (108.5, 108.5),
    # Not a finite number: pd.isna does not catch infinity, and passing it on
    # only moves the failure to an int() conversion downstream.
    (float("inf"), None),
    (float("-inf"), None),
    (float("nan"), None),
    ("unknown", None),
    ("", None),
    (None, None),
])
def test_finite_float(value, expected):
    assert finite_float(value) == expected


@pytest.mark.parametrize("value,expected", [
    (108, 108),
    # An integral float is the common real case: pandas types the column float
    # as soon as any row in it is blank.
    (108.0, 108),
    ("108", 108),
    (0, 0),
    (-8, -8),
    # Fractions are rejected rather than truncated.
    (108.5, None),
    (0.9, None),
    (float("inf"), None),
    ("unknown", None),
])
def test_whole_number_rejects_fractions(value, expected):
    assert whole_number(value) == expected


@pytest.mark.parametrize("value,expected", [
    (108, 108),
    (108.0, 108),
    # Discrete quantity: a fraction means the cell is malformed.
    (12.9, None),
    (0.5, None),
    # Not a count.
    (0, None),
    (-8, None),
    # No hardware reports a grid this large, and in a column-wide reduction such
    # a value would saturate a vectorised astype(int) to the int64 maximum.
    (MAX_PLAUSIBLE_CORE_COUNT + 1, None),
    (1e30, None),
    (float("inf"), None),
])
def test_core_count_from_value(value, expected):
    assert core_count_from_value(value) == expected


def test_getters_honour_absent_columns_and_defaults():
    assert get_numeric_value({}, "MISSING") is None
    assert get_core_count({}, "MISSING", default=64) == 64
    assert get_core_count({"C": 12.9}, "C", default=64) == 64
    assert get_int({}, "MISSING", default=-1) == -1
    # Zero is a legal device id, so get_int must not reject it the way
    # get_core_count rejects a zero core count.
    assert get_int({"DEVICE ID": 0}, "DEVICE ID") == 0
    assert get_core_count({"CORE COUNT": 0}, "CORE COUNT") is None
    # A device id pandas typed as text is still readable.
    assert get_int({"DEVICE ID": "3"}, "DEVICE ID") == 3


def test_get_value_physical_logical_reads_both_parts():
    # Every fixture in tests/data writes N[N], so the two branches are only
    # distinguishable with unequal parts.
    assert get_value_physical_logical("512[480]") == 512
    assert get_value_physical_logical("512[480]", is_physical=False) == 480
    # Back-compatible forms.
    assert get_value_physical_logical(64) == 64
    assert get_value_physical_logical("64") == 64


@pytest.mark.parametrize("value", [
    float("inf"),
    float("nan"),
    "unknown",
    "512.5[512]",
    "[]",
    "",
])
def test_get_value_physical_logical_returns_none_rather_than_raising(value):
    # These reach matmul and conv analysis through get_tensor_dim; raising here
    # would abort the whole report over one malformed shape cell.
    assert get_value_physical_logical(value) is None


@pytest.mark.parametrize("value,expected", [
    (512, 512),
    (512.0, 512),
    ("512", 512),
    (MAX_PLAUSIBLE_TENSOR_DIM, MAX_PLAUSIBLE_TENSOR_DIM),
    # Discrete quantity: a fraction means the cell is malformed.
    (512.5, None),
    # A degenerate axis has no modellable figures, so it takes the same path as
    # a corrupt cell rather than producing a zero-FLOP row.
    (0, None),
    (-8, None),
    # Finite and whole, but no tensor is this large. Left unbounded, the
    # 301-digit int this coerces to overflows the modelled FLOP arithmetic.
    (1e300, None),
    (MAX_PLAUSIBLE_TENSOR_DIM + 1, None),
    (float("inf"), None),
])
def test_tensor_dim_from_value(value, expected):
    assert tensor_dim_from_value(value) == expected


@pytest.mark.parametrize("cell", [1e300, "1e300", "1e300[1e300]", 0, -8])
def test_implausible_dimensions_do_not_reach_flop_arithmetic(cell):
    # The regression this guards: get_value_physical_logical used to accept any
    # finite whole number, and (M * K * N * 2) / duration then raised
    # OverflowError, aborting the whole report instead of omitting one op's
    # modelled figures. Both analyze_matmul and analyze_conv rely on the None.
    dimension = get_value_physical_logical(cell)
    assert dimension is None
    # Sanity-check the arithmetic those callers would otherwise perform.
    if cell == 1e300:
        with pytest.raises(OverflowError):
            (int(cell) ** 3 * 2) / 1e-9
