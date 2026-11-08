#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pandas as pd
import pytest

from tt_perf_report.perf_report import (
    BOUND_ANALYSIS_FULL,
    BOUND_ANALYSIS_NONE,
    ArchitectureSpec,
    analyze_declared_roofline,
    analyze_op,
)

BLACKHOLE = ArchitectureSpec.from_name("blackhole")

pytestmark = pytest.mark.usefixtures("reset_classification_cache")


def _row(attributes, duration_ns=10_000, cores=110, math_fidelity="HiFi4"):
    return pd.Series(
        {
            "OP CODE": "GenericOpDeviceOperation",
            "OP TYPE": "tt_dnn_device",
            "DEVICE ID": 0,
            "CORE COUNT": cores,
            "DEVICE KERNEL DURATION [ns]": duration_ns,
            "OP TO OP LATENCY [ns]": 0,
            "ATTRIBUTES": attributes,
            "MATH FIDELITY": math_fidelity,
            "INPUT_0_MEMORY": "DEV_0_DRAM_INTERLEAVED",
            "INPUT_0_DATATYPE": "BFLOAT16",
            "OUTPUT_0_DATATYPE": "BFLOAT16",
        }
    )


def _attributes(
    read="std::nullopt",
    written="std::nullopt",
    flops="std::nullopt",
    fidelity="std::nullopt",
):
    return (
        f"{{'declared_dram_bytes_read': '{read}'; 'declared_dram_bytes_written': '{written}'; "
        f"'declared_flops': '{flops}'; 'declared_math_fidelity': '{fidelity}'; 'num_mesh_programs': '1'}}"
    )


def test_no_declaration_is_not_analysed():
    assert analyze_declared_roofline(_row(_attributes()), BLACKHOLE) is None
    assert (
        analyze_declared_roofline(_row("{'num_mesh_programs': '1'}"), BLACKHOLE) is None
    )

    op, _ = analyze_op(_row(_attributes()), None, arch_spec=BLACKHOLE)
    assert op["Bound Analysis"].raw_value == BOUND_ANALYSIS_NONE
    assert op["DRAM %"].raw_value is None


def test_declared_dram_bytes_against_peak_bandwidth():
    # 4,096,000 + 1,024,000 bytes in 10 µs = 512 GB/s, the Blackhole peak
    declared = analyze_declared_roofline(
        _row(_attributes(read=4_096_000, written=1_024_000, flops=0)), BLACKHOLE
    )

    assert declared.dram_bytes == 5_120_000
    assert declared.dram_speed_gb_s == pytest.approx(512)
    assert declared.dram_percentage == pytest.approx(100)
    assert declared.flops_percentage == pytest.approx(0)


def test_declared_flops_use_declared_fidelity_over_kernel_fidelity():
    peak_lofi = BLACKHOLE.tflops_per_core("LoFi") * 1e12 * 110
    flops = int(peak_lofi * 10e-6 / 2)  # half of the LoFi peak over 10 µs
    row = _row(_attributes(flops=flops, fidelity="LoFi"), math_fidelity="HiFi4")

    declared = analyze_declared_roofline(row, BLACKHOLE)

    assert declared.math_fidelity == "LoFi"
    assert declared.flops_percentage == pytest.approx(50, rel=1e-6)
    assert declared.dram_percentage is None


def test_declared_flops_fall_back_to_kernel_fidelity():
    peak_hifi2 = BLACKHOLE.tflops_per_core("HiFi2") * 1e12 * 64
    row = _row(
        _attributes(flops=int(peak_hifi2 * 10e-6)), cores=64, math_fidelity="HiFi2"
    )

    declared = analyze_declared_roofline(row, BLACKHOLE)

    assert declared.math_fidelity == "HiFi2"
    assert declared.flops_percentage == pytest.approx(100, rel=1e-6)


def test_declared_op_gets_full_bound_analysis():
    op, _ = analyze_op(
        _row(_attributes(read=4_000_000, flops=1_000)), None, arch_spec=BLACKHOLE
    )

    assert op["Bound Analysis"].raw_value == BOUND_ANALYSIS_FULL
    assert op["DRAM Bytes"].raw_value == 4_000_000
    assert op["DRAM"].raw_value == pytest.approx(400)
    assert op["DRAM %"].raw_value == pytest.approx(400 / 512 * 100)
    assert op["FLOPs %"].raw_value is not None


def test_invalid_duration_keeps_bytes_but_no_rates():
    declared = analyze_declared_roofline(
        _row(_attributes(read=1000), duration_ns=None), BLACKHOLE
    )

    assert declared.dram_bytes == 1000
    assert declared.dram_speed_gb_s is None
    assert declared.dram_percentage is None
