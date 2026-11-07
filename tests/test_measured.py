#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import csv
import json
import os
import shutil
from io import StringIO

import pytest

from tt_perf_report import measured
from tt_perf_report.perf_report import (
    generate_perf_report,
    report_csv_column_labels as CSV_COLUMN,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# One Matmul (GLOBAL CALL COUNT 1024 + device) then one ReduceScatter (2048 + device) on 8 devices.
CCL_FIXTURE = os.path.join(DATA_DIR, "bh_8xp150_deepseek_v3_d_p.csv")

pytestmark = pytest.mark.usefixtures("reset_classification_cache")


def _tracy_layout(tmp_path):
    """The folder layout `python -m tracy -r -o DIR` writes: DIR/reports/<date>/ops_perf_results_*.csv, DIR/.logs."""
    report_dir = tmp_path / "reports" / "2026_11_07_00_00_00"
    report_dir.mkdir(parents=True)
    csv_path = report_dir / "ops_perf_results_2026_11_07_00_00_00.csv"
    shutil.copy(CCL_FIXTURE, csv_path)
    logs = tmp_path / ".logs"
    logs.mkdir()
    return csv_path, logs


def _write_noc_trace(logs, device, global_call_count, events):
    path = (
        logs / f"noc_trace_dev{device}_MatmulDeviceOperation_ID{global_call_count}.json"
    )
    path.write_text(json.dumps(events))


def _run(csv_path, tmp_path, mocker, **kwargs):
    mocker.patch("sys.stdout", new_callable=StringIO)
    mocker.patch("tt_perf_report.perf_report.plot_stacked_report")
    out = tmp_path / "report.csv"
    out_json = tmp_path / "report.json"
    generate_perf_report(
        csv_files=[str(csv_path)],
        start_signpost=None,
        end_signpost=None,
        ignore_signposts=True,
        print_signposts=False,
        min_percentage=0,
        id_range=None,
        arch=None,
        csv_output_file=str(out),
        no_advice=True,
        tracing_mode=False,
        raw_op_codes=False,
        no_host_ops=False,
        no_summary=True,
        group_by=None,
        classic_colors=False,
        summary_file=None,
        no_stacked_report=True,
        no_stack_by_in0=True,
        stacked_csv=None,
        no_merge_devices=True,
        json_output_file=str(out_json),
        **kwargs,
    )
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, json.loads(out_json.read_text())


def test_noc_trace_dram_bytes_with_state_and_l1():
    events = [
        {
            "type": "READ",
            "dx": 0,
            "dy": 3,
            "num_bytes": 16384,
            "sx": 1,
            "sy": 2,
            "proc": "NCRISC",
            "noc": "NOC_0",
        },
        {
            "type": "READ_SET_STATE",
            "dx": 9,
            "dy": 5,
            "num_bytes": 0,
            "sx": 1,
            "sy": 2,
            "proc": "NCRISC",
            "noc": "NOC_0",
        },
        {
            "type": "READ_WITH_STATE",
            "num_bytes": 8192,
            "sx": 1,
            "sy": 2,
            "proc": "NCRISC",
            "noc": "NOC_0",
        },
        {
            "type": "READ_WITH_STATE",
            "num_bytes": 64,
            "sx": 2,
            "sy": 2,
            "proc": "NCRISC",
            "noc": "NOC_0",
        },
        {
            "type": "WRITE_",
            "dx": 3,
            "dy": 3,
            "num_bytes": 2048,
            "sx": 1,
            "sy": 2,
            "proc": "BRISC",
            "noc": "NOC_1",
        },
        {
            "type": "WRITE_",
            "dx": 9,
            "dy": 1,
            "num_bytes": 4096,
            "sx": 1,
            "sy": 2,
            "proc": "BRISC",
            "noc": "NOC_1",
        },
    ]
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(events, f)
    try:
        assert measured.dram_bytes_from_noc_trace(f.name, {0, 9}) == {
            "r": 24576,
            "w": 4096,
            "unattributed": 64,
        }
    finally:
        os.unlink(f.name)


def test_measured_dram_and_chip_arrival_from_tracy_logs(tmp_path, mocker):
    csv_path, logs = _tracy_layout(tmp_path)
    # device 0's matmul (93,591 ns) reads 46,795,500 bytes from DRAM: 500 GB/s
    _write_noc_trace(
        logs, 0, 1024, [{"type": "READ", "dx": 0, "dy": 1, "num_bytes": 46_795_500}]
    )
    # put every device on device 0's clock at the matmul start, then delay device 3 by 10 µs (13,500 cycles)
    with open(CCL_FIXTURE, newline="") as f:
        starts = {
            int(r["DEVICE ID"]): int(r["DEVICE FW START CYCLE"])
            for r in csv.DictReader(f)
            if r["GLOBAL CALL COUNT"] == str(1024 + int(r["DEVICE ID"]))
        }
    with open(logs / "sync_device_info.csv", "w") as f:
        for device, start in starts.items():
            shift = starts[0] - start + (13_500 if device == 3 else 0)
            f.write(
                f"{device:5},    1,    2, 97888290008, 33720, 11240.15, 33720, 63058, 1007.8, 3.5, 0.33, 63058,1.0,0\n"
            )
            # the device sync at open and the one at close; they disagree by 20 cycles (about 0.015 µs)
            f.write(f"{device:5},,,,,,,,,,,,{1.0:20.15f},{shift - 20:20}\n")
            f.write(f"{device:5},,,,,,,,,,,,{1.0:20.15f},{shift:20}\n")

    rows, report = _run(csv_path, tmp_path, mocker)

    assert report["schema"] == "tt-perf-report/ops/v1"
    assert set(report["inputs"]) == {"noc_traces", "chip_sync"}
    matmul = [
        r for r in rows if r["Global Call Count"] in {str(1024 + d) for d in range(8)}
    ]
    assert len(matmul) == 8
    dev0 = next(r for r in matmul if r["Device"] == "0")
    assert float(dev0[CSV_COLUMN["Measured DRAM"]]) == pytest.approx(500, rel=1e-3)
    assert float(dev0[CSV_COLUMN["Measured DRAM %"]]) == pytest.approx(
        500 / 512 * 100, rel=1e-3
    )
    assert dev0["Bound By"].startswith("DRAM")
    assert [r["Last Chip In"] for r in matmul] == ["3"] * 8
    assert float(dev0[CSV_COLUMN["Arrival Spread"]]) == pytest.approx(10, abs=0.05)
    assert float(dev0[CSV_COLUMN["Chip Sync Error"]]) == pytest.approx(
        20 / 1350, abs=0.005
    )
    lead = {r["Device"]: float(r[CSV_COLUMN["Self Arrival Lead"]]) for r in matmul}
    assert lead.pop("3") == pytest.approx(0)
    assert all(v == pytest.approx(10, abs=0.05) for v in lead.values())


def test_without_inputs_measured_columns_are_blank(tmp_path, mocker):
    rows, report = _run(CCL_FIXTURE, tmp_path, mocker)

    assert report["inputs"] == {}
    assert all(
        r[CSV_COLUMN["Measured DRAM"]] == "" and r["Last Chip In"] == "" for r in rows
    )
    assert all(r["Bound By"] for r in rows if r["OP Code"].startswith("Matmul"))


def test_bound_by_order():
    counters = {"math": 80.0, "fpu": None, "sfpu": None, "cb_wait": 50.0}
    assert measured.bound_by(
        0, 10_000, None, 70.0, None, counters, None, None
    ).startswith("DRAM")
    assert measured.bound_by(
        0, 10_000, None, 10.0, None, counters, None, None
    ).startswith("math")
    idle = {"math": 5.0, "fpu": None, "sfpu": None, "cb_wait": 50.0}
    assert measured.bound_by(0, 10_000, None, 10.0, None, idle, 2, 4_000).startswith(
        "waiting on chip 2"
    )
    assert measured.bound_by(0, 10_000, None, 10.0, None, idle, 0, 0).startswith(
        "waiting on CB"
    )
    quiet = {"math": None, "fpu": None, "sfpu": None, "cb_wait": None}
    assert measured.bound_by(
        0, 10_000, 30_000, None, None, quiet, None, None
    ).startswith("dispatch")
    assert (
        measured.bound_by(0, 10_000, None, None, None, quiet, None, None)
        == "unexplained"
    )
