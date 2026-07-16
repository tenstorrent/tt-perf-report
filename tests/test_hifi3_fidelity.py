#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import pandas as pd

from tt_perf_report.perf_report import (
    ArchitectureSpec,
    CsvFormat,
    analyze_matmul,
    evaluate_fidelity,
)

REGISTERED_ARCHS = ("wormhole", "blackhole", "bh20", "n1")


@pytest.mark.parametrize("arch_name", REGISTERED_ARCHS)
def test_tflops_per_core_hifi3_resolves(arch_name):
    spec = ArchitectureSpec.from_name(arch_name)
    assert spec.tflops_per_core("HiFi3") == spec.tflops_hifi3


def test_wormhole_hifi3_peak():
    spec = ArchitectureSpec.from_name("wormhole")
    assert spec.tflops_per_core("HiFi3") == pytest.approx((74 * 4 / 3) / 72)


@pytest.mark.parametrize("arch_name", ("blackhole", "bh20", "n1"))
def test_bh_family_hifi3_is_lofi_over_3(arch_name):
    spec = ArchitectureSpec.from_name(arch_name)
    assert spec.tflops_hifi3 == pytest.approx(spec.tflops_lofi / 3)
    assert spec.tflops_hifi4 == pytest.approx(spec.tflops_lofi / 4)
    assert spec.tflops_hifi2 == pytest.approx(spec.tflops_lofi / 2)


@pytest.mark.parametrize("arch_name", REGISTERED_ARCHS)
def test_fidelity_tflops_ordering(arch_name):
    spec = ArchitectureSpec.from_name(arch_name)
    assert spec.tflops_hifi4 < spec.tflops_hifi3 < spec.tflops_hifi2 < spec.tflops_lofi


def test_from_name_worker_cores_override_preserves_hifi3():
    base = ArchitectureSpec.from_name("wormhole")
    overridden = ArchitectureSpec.from_name("wormhole", worker_cores=32)
    assert overridden.worker_cores == 32
    assert overridden.tflops_per_core("HiFi3") == base.tflops_hifi3


def test_unknown_math_fidelity_raises():
    spec = ArchitectureSpec.from_name("wormhole")
    with pytest.raises(ValueError, match="Unknown math fidelity"):
        spec.tflops_per_core("HiFi5")


@pytest.mark.parametrize(
    "inputs,expected_status,advice_must_contain",
    [
        (
            ("BFLOAT16", "BFLOAT16", "BFLOAT16", "HiFi3"),
            "too_low",
            ("HiFi4", "BF16"),
        ),
        (
            ("BFLOAT16", "BFLOAT16", "BFLOAT16", "HiFi2"),
            "too_low",
            ("HiFi4", "BF16"),
        ),
        (
            ("BFLOAT16", "BFLOAT16", "BFLOAT4_B", "HiFi3"),
            "too_high",
            ("HiFi2", "BFP4", "than HiFi3"),
        ),
        (
            ("BFLOAT8_B", "BFLOAT8_B", "BFLOAT8_B", "HiFi3"),
            "too_high",
            ("HiFi2", "than HiFi3"),
        ),
        (
            ("BFLOAT8_B", "BFLOAT8_B", "BFLOAT4_B", "HiFi3"),
            "too_high",
            ("HiFi2", "than HiFi3"),
        ),
        (
            ("BFLOAT8_B", "BFLOAT4_B", "BFLOAT8_B", "HiFi3"),
            "too_high",
            ("LoFi", "BFP4"),
        ),
    ],
)
def test_evaluate_fidelity_hifi3(inputs, expected_status, advice_must_contain):
    status, advice = evaluate_fidelity(*inputs)
    assert status == expected_status
    assert advice is not None
    for fragment in advice_must_contain:
        assert fragment in advice, f"expected {fragment!r} in {advice!r}"


def test_analyze_matmul_uses_hifi3_peak():
    arch_spec = ArchitectureSpec.from_name("wormhole")
    duration_ns = 1_000_000  # 1 ms
    m, k, n = 32, 32, 32
    core_count = 1
    dim = lambda value: f"{value}[{value}]"
    row = pd.Series(
        {
            "INPUT_0_MEMORY": "DEV_0_L1",
            "INPUT_1_MEMORY": "DEV_0_L1",
            "OUTPUT_0_MEMORY": "DEV_0_L1",
            "INPUT_0_DATATYPE": "BFLOAT16",
            "INPUT_1_DATATYPE": "BFLOAT16",
            "OUTPUT_0_DATATYPE": "BFLOAT16",
            "INPUT_0_W_PAD[LOGICAL]": dim(1),
            "INPUT_0_Z_PAD[LOGICAL]": dim(1),
            "INPUT_0_Y_PAD[LOGICAL]": dim(m),
            "INPUT_0_X_PAD[LOGICAL]": dim(k),
            "INPUT_1_W_PAD[LOGICAL]": dim(1),
            "INPUT_1_Z_PAD[LOGICAL]": dim(1),
            "INPUT_1_Y_PAD[LOGICAL]": dim(k),
            "INPUT_1_X_PAD[LOGICAL]": dim(n),
            "OUTPUT_0_W_PAD[LOGICAL]": dim(1),
            "OUTPUT_0_Z_PAD[LOGICAL]": dim(1),
            "OUTPUT_0_Y_PAD[LOGICAL]": dim(m),
            "OUTPUT_0_X_PAD[LOGICAL]": dim(n),
            "DEVICE KERNEL DURATION [ns]": duration_ns,
            "CORE COUNT": core_count,
            "MATH FIDELITY": "HiFi3",
            "OP CODE": "Matmul",
            "ATTRIBUTES": "",
        }
    )

    (
        _,
        _,
        flops,
        flops_percentage,
        _,
        _,
        math_fidelity,
        _,
        _,
        *_,
    ) = analyze_matmul(row, CsvFormat.V2, arch_spec)

    assert math_fidelity == "HiFi3"
    peak_flops = arch_spec.tflops_per_core("HiFi3") * 1e12 * core_count
    assert flops_percentage == pytest.approx((flops / peak_flops) * 100)
