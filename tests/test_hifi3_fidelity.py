#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import pandas as pd

from tt_perf_report.perf_report import (
    ArchitectureSpec,
    Cell,
    CsvFormat,
    analyze_conv,
    analyze_matmul,
    color_row,
    evaluate_fidelity,
    generate_matmul_advice,
)

REGISTERED_ARCHS = ("wormhole", "blackhole", "blackhole_p100", "bh20", "n1")


@pytest.mark.parametrize("arch_name", REGISTERED_ARCHS)
def test_tflops_per_core_hifi3_resolves(arch_name):
    spec = ArchitectureSpec.from_name(arch_name)
    assert spec.tflops_per_core("HiFi3") == spec.tflops_hifi3


def test_wormhole_hifi3_peak():
    spec = ArchitectureSpec.from_name("wormhole")
    assert spec.tflops_per_core("HiFi3") == pytest.approx((74 * 4 / 3) / 72)


@pytest.mark.parametrize("arch_name", ("blackhole", "blackhole_p100", "bh20", "n1"))
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


def test_blackhole_card_specs_and_aliases():
    p150 = ArchitectureSpec.from_name("p150")
    p100 = ArchitectureSpec.from_name("p100a")

    assert p150.name == "blackhole"
    assert p150.worker_cores == 110
    assert p150.dram_sharded_matmul_cores == 8
    assert p150.dram_bandwidth_gb_s == 512

    assert p100.name == "blackhole_p100"
    assert p100.worker_cores == 110
    assert p100.dram_sharded_matmul_cores == 7
    assert p100.dram_bandwidth_gb_s == 448


def test_blackhole_csv_fallback_and_p100_override():
    df = pd.DataFrame(
        {
            "DEVICE ARCH": ["blackhole"],
            "AVAILABLE WORKER CORE COUNT": [0],
        }
    )

    detected = ArchitectureSpec.from_df(df)
    overridden = ArchitectureSpec.from_df(df, "p100")

    assert detected.name == "blackhole"
    assert detected.worker_cores == 110
    assert detected.dram_bandwidth_gb_s == 512
    assert overridden.name == "blackhole_p100"
    assert overridden.worker_cores == 110
    assert overridden.dram_bandwidth_gb_s == 448


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


def _matmul_row(core_count=1, attributes="", math_fidelity="HiFi3"):
    duration_ns = 1_000_000  # 1 ms
    m, k, n = 32, 32, 32
    dim = lambda value: f"{value}[{value}]"
    return pd.Series(
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
            "MATH FIDELITY": math_fidelity,
            "OP CODE": "Matmul",
            "ATTRIBUTES": attributes,
        }
    )


def test_analyze_matmul_uses_hifi3_peak():
    arch_spec = ArchitectureSpec.from_name("wormhole")
    core_count = 1
    row = _matmul_row(core_count=core_count)

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


@pytest.mark.parametrize(
    "arch_name,expected_cores",
    [("wormhole", 12), ("blackhole", 8), ("p100", 7)],
)
def test_dram_sharded_matmul_uses_architecture_worker_count(arch_name, expected_cores):
    arch_spec = ArchitectureSpec.from_name(arch_name)
    row = _matmul_row(
        core_count=arch_spec.worker_cores,
        attributes="MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
    )

    result = analyze_matmul(row, CsvFormat.V2, arch_spec)
    flops = result[2]
    flops_percentage = result[3]
    adjusted_core_count = result[8]

    assert adjusted_core_count == expected_cores
    peak_flops = arch_spec.tflops_per_core("HiFi3") * 1e12 * expected_cores
    assert flops_percentage == pytest.approx((flops / peak_flops) * 100)


def test_matmul_with_zero_core_metadata_omits_flops_utilization():
    result = analyze_matmul(
        _matmul_row(core_count=0),
        CsvFormat.V2,
        ArchitectureSpec.from_name("blackhole"),
    )
    assert result[3] is None
    assert result[8] is None


def test_blackhole_conv_uses_operation_core_count():
    row = pd.read_csv("tests/data/bh_p100_dlv3.csv", low_memory=False)
    row = row[
        row["OP CODE"].str.contains("Conv2d", na=False)
        & (row["CORE COUNT"] == 33)
    ].iloc[0]
    assert int(row["CORE COUNT"]) == 33

    arch_spec = ArchitectureSpec.from_name("blackhole", worker_cores=110)
    flops, flops_percentage, *_ = analyze_conv(row, CsvFormat.V2_1, arch_spec)
    expected_peak = arch_spec.tflops_per_core(row["MATH FIDELITY"]) * 1e12 * 33
    assert flops_percentage == pytest.approx((flops / expected_peak) * 100)


def _colorable_op(core_count, worker_cores, dram_sharded=False):
    return {
        "OP Code": Cell("CopyDeviceOperation"),
        "Cores": Cell(core_count),
        "Architecture Worker Cores": Cell(worker_cores),
        "DRAM Sharded": Cell(dram_sharded),
        "Bound": Cell(""),
        "Op-to-Op Gap": Cell(None),
    }


def test_core_highlighting_is_architecture_aware():
    full_bh = color_row(_colorable_op(110, 110), percentage=10, min_percentage=0.5)
    dram_sharded_bh = color_row(_colorable_op(8, 110, True), percentage=10, min_percentage=0.5)
    partial_bh = color_row(_colorable_op(8, 110), percentage=10, min_percentage=0.5)

    assert full_bh["Cores"].color == "green"
    assert dram_sharded_bh["Cores"].color == "green"
    assert partial_bh["Cores"].color == "red"


def _flop_bound_matmul_op(core_count, worker_cores, dram_sharded=False):
    return {
        "Sparse Active Batches Missing": Cell(False),
        "Sparse Matmul": Cell(False),
        "OP Code": Cell("MatmulDeviceOperation"),
        "Math Fidelity": Cell("LoFi BF16 x BFP4 => BF16"),
        "Output Datatype": Cell("BFLOAT16"),
        "Input 0 Datatype": Cell("BFLOAT16"),
        "Input 1 Datatype": Cell("BFLOAT4_B"),
        "Cores": Cell(core_count),
        "Architecture Worker Cores": Cell(worker_cores),
        "Bound": Cell("FLOP"),
        "DRAM Sharded": Cell(dram_sharded),
        "FLOPs %": Cell(70),
    }


def test_grid_advice_uses_architecture_ceiling_and_skips_dram_sharded_matmuls():
    partial_bh = generate_matmul_advice(_flop_bound_matmul_op(80, 110))
    full_bh = generate_matmul_advice(_flop_bound_matmul_op(110, 110))
    dram_sharded_bh = generate_matmul_advice(_flop_bound_matmul_op(8, 110, True))

    # The advice now names the ceiling it measured against, which on a subdevice
    # run is that subdevice's budget rather than the architecture grid.
    assert "Increase grid size (currently using 80 of 110)" in partial_bh
    assert not any("Increase grid size" in advice for advice in full_bh)
    assert not any("Increase grid size" in advice for advice in dram_sharded_bh)


def test_dram_sharded_advice_does_not_require_output_subblock_fields():
    op_data = _flop_bound_matmul_op(8, 110, True)
    op_data.update(
        {
            "Bound": Cell("SLOW"),
            "Input 0 Memory": Cell("DEV_0_L1_WIDTH_SHARDED"),
            "Inner Dim Block Size": Cell(4),
            "Output Subblock H": Cell(None),
            "Output Subblock W": Cell(None),
        }
    )

    advice = generate_matmul_advice(op_data)

    assert "in0_block_w=4 looks good 🤷" in advice
    assert not any("output subblock" in item.lower() for item in advice)
