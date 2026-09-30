# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Cycle-counter columns for synthetic ops, shared by the overlap tests."""

BLACKHOLE_CYCLES_PER_NS = 1.35


def fw_timing(start_us, duration_us=100, fw_overhead_us=1):
    """
    Cycle columns for an op whose kernel runs [start_us, start_us + duration_us].

    The FW span opens fw_overhead_us before the kernel, as on real captures, so
    that a test comparing raw FW spans would find overlap where there is none.
    """
    end_ns = (start_us + duration_us) * 1000
    fw_start_ns = (start_us - fw_overhead_us) * 1000
    return {
        "DEVICE FW START CYCLE": round(fw_start_ns * BLACKHOLE_CYCLES_PER_NS),
        "DEVICE FW END CYCLE": round(end_ns * BLACKHOLE_CYCLES_PER_NS),
        "DEVICE FW DURATION [ns]": end_ns - fw_start_ns,
        "DEVICE KERNEL DURATION [ns]": duration_us * 1000,
    }
