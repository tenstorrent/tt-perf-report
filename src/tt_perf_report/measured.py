# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Measured visibility for any op: DRAM bytes from NoC event traces, chip arrival from synchronized clocks, and the
inputs the per-op "Bound By" verdict reads (perf counters and sum zones are columns of the ops CSV already).

Everything here is optional. A column is filled only when its input is present:
- NoC traces: noc_trace_dev<D>[_<op>]_ID<global call count>[_traceID<n>].json, written by
  `python -m tracy -r --collect-noc-traces` into <profiler dir>/.logs.
- Chip sync: sync_device_info.csv, written by `python -m tracy -r --sync-host-device` into the same folder; or
  SYNCED START [ns] / SYNCED END [ns] columns in the ops CSV.
"""

import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd

# Columns annotate_measured adds to the ops dataframe before devices are merged.
MEAS_DRAM_READ = "MEASURED DRAM READ [B]"
MEAS_DRAM_WRITE = "MEASURED DRAM WRITE [B]"
MEAS_UNATTRIBUTED = "MEASURED UNATTRIBUTED NOC [B]"
XCHIP_LAST_IN = "XCHIP LAST IN"
XCHIP_IN_SPREAD = "XCHIP IN SPREAD [ns]"
XCHIP_LAST_OUT = "XCHIP LAST OUT"
XCHIP_OUT_SPREAD = "XCHIP OUT SPREAD [ns]"
XCHIP_SELF_WAIT = "XCHIP SELF ARRIVAL LEAD [ns]"
XCHIP_SYNC_ERROR = "XCHIP SYNC ERROR [ns]"

# NoC0 x of the DRAM endpoints, per architecture family.
DRAM_NOC0_X = {"blackhole": {0, 9}, "wormhole": {0, 5}}

_READS = {
    "READ",
    "READ_WITH_STATE",
    "READ_WITH_STATE_AND_TRID",
    "READ_DRAM_SHARDED_WITH_STATE",
}
_WRITES = {
    "WRITE_",
    "WRITE_WITH_TRID",
    "WRITE_INLINE",
    "WRITE_WITH_STATE",
    "WRITE_WITH_TRID_WITH_STATE",
}
_SET_STATE = {
    "READ_SET_STATE": "r",
    "READ_DRAM_SHARDED_SET_STATE": "r",
    "WRITE_SET_STATE": "w",
    "WRITE_WITH_TRID_SET_STATE": "w",
}
_TRACE_NAME = re.compile(
    r"noc_trace_dev(\d+)(?:_.*?)?_ID(\d+)(?:_traceID(\d+))?\.json$"
)
_DEVICE_ID_BITS = 10
_DEVICE_OP_ID_BITS = 31


def find_profiler_logs(csv_path: str) -> Optional[str]:
    """The .logs folder of the tracy run that wrote csv_path (<dir>/reports/<date>/ops_perf_results_*.csv)."""
    here = os.path.dirname(os.path.abspath(csv_path))
    for up in (os.path.join(here, "..", "..", ".logs"), os.path.join(here, ".logs")):
        if os.path.isdir(up):
            return os.path.normpath(up)
    return None


def dram_bytes_from_noc_trace(path: str, dram_x) -> Dict[str, int]:
    """DRAM read/write bytes of one program's NoC trace: a transfer whose remote endpoint is a DRAM endpoint. A
    transfer that carries no coordinates (with-state) takes those of the last set-state on its core, RISC and NoC.
    """
    with open(path) as f:
        events = json.load(f)
    out = {"r": 0, "w": 0, "unattributed": 0}
    state = {}
    for e in events:
        kind = e.get("type")
        if kind is None:
            continue
        source = (e.get("sx"), e.get("sy"), e.get("proc"), e.get("noc"))
        if kind in _SET_STATE:
            if "dx" in e:
                state[(source, _SET_STATE[kind])] = e["dx"]
            continue
        if kind in _READS:
            direction = "r"
        elif kind in _WRITES:
            direction = "w"
        else:
            continue
        x = e["dx"] if "dx" in e else state.get((source, direction))
        if x is None:
            out["unattributed"] += int(e.get("num_bytes", 0))
        elif x in dram_x:
            out[direction] += int(e.get("num_bytes", 0))
    return out


def load_noc_traces(
    folder: str, arch_name: str
) -> Dict[Tuple[int, int, Optional[int]], Dict[str, int]]:
    dram_x = DRAM_NOC0_X[
        "wormhole" if arch_name.startswith("wormhole") else "blackhole"
    ]
    result = {}
    for path in glob.glob(os.path.join(folder, "noc_trace_dev*.json")):
        m = _TRACE_NAME.search(os.path.basename(path))
        if not m:
            continue
        trace = int(m.group(3)) if m.group(3) else None
        result[(int(m.group(1)), int(m.group(2)), trace)] = dram_bytes_from_noc_trace(
            path, dram_x
        )
    return result


def load_sync_file(path: str) -> Dict[int, list]:
    """sync_device_info.csv: besides the host-sync rows, one '<device>,,,...,<scale>,<shift>' row per device and
    device sync (one at device open, one at close), mapping the device's cycles onto the root device's:
    t_root = t * scale + shift. Returns every fit per device, in file order."""
    maps = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = [x.strip() for x in line.split(",")]
            if len(fields) < 4 or not fields[0].isdigit() or any(fields[1:-2]):
                continue
            try:
                maps[int(fields[0])].append((float(fields[-2]), float(fields[-1])))
            except ValueError:
                continue
    return dict(maps)


def _num(row, column):
    if column not in row:
        return None
    value = pd.to_numeric(row[column], errors="coerce")
    return None if pd.isna(value) else float(value)


def _base_op_id(global_call_count) -> Optional[int]:
    try:
        gcc = int(global_call_count)
    except (TypeError, ValueError):
        return None
    return (gcc & ((1 << _DEVICE_OP_ID_BITS) - 1)) >> _DEVICE_ID_BITS


def annotate_measured(
    df: pd.DataFrame, arch_name: str, noc_folder=None, sync_file=None
) -> Dict[str, str]:
    """Adds the measured columns to df in place (per device, before merging). Returns which inputs were used."""
    used = {}
    if noc_folder:
        traces = load_noc_traces(noc_folder, arch_name)
        if traces:
            used["noc_traces"] = f"{noc_folder} ({len(traces)} programs)"
            reads, writes, unattributed = [], [], []
            for _, row in df.iterrows():
                key = None
                dev, gcc = _num(row, "DEVICE ID"), _num(row, "GLOBAL CALL COUNT")
                if dev is not None and gcc is not None:
                    trace_id = _num(row, "METAL TRACE ID")
                    key = traces.get((int(dev), int(gcc), None))
                    if key is None and trace_id is not None:
                        # traced programs: the file names carry (trace id << 32) | replay counter
                        candidates = [
                            v
                            for k, v in traces.items()
                            if k[0] == int(dev) and k[1] == int(gcc)
                        ]
                        key = candidates[0] if len(candidates) == 1 else None
                reads.append(key["r"] if key else None)
                writes.append(key["w"] if key else None)
                unattributed.append(key["unattributed"] if key else None)
            df[MEAS_DRAM_READ] = reads
            df[MEAS_DRAM_WRITE] = writes
            df[MEAS_UNATTRIBUTED] = unattributed

    synced = "SYNCED START [ns]" in df.columns and "SYNCED END [ns]" in df.columns
    maps = load_sync_file(sync_file) if sync_file and not synced else {}
    if not synced and not maps:
        return used
    used["chip_sync"] = (
        "SYNCED START/END [ns] columns"
        if synced
        else f"{sync_file} ({len(maps)} devices)"
    )

    # cycles -> ns from the rows themselves (FW cycles over FW ns)
    ratios = []
    for _, row in df.head(5000).iterrows():
        s, e, ns = (
            _num(row, "DEVICE FW START CYCLE"),
            _num(row, "DEVICE FW END CYCLE"),
            _num(row, "DEVICE FW DURATION [ns]"),
        )
        if s is not None and e is not None and ns and ns > 1000:
            ratios.append((e - s) / ns)
    cycles_per_ns = sorted(ratios)[len(ratios) // 2] if ratios else 1.0

    spans = {}
    sync_error = {}
    groups = defaultdict(dict)
    for index, row in df.iterrows():
        dev = _num(row, "DEVICE ID")
        base = _base_op_id(row.get("GLOBAL CALL COUNT"))
        if dev is None or base is None:
            continue
        dev = int(dev)
        if synced:
            s, e = _num(row, "SYNCED START [ns]"), _num(row, "SYNCED END [ns]")
        else:
            s, e = _num(row, "DEVICE FW START CYCLE"), _num(row, "DEVICE FW END CYCLE")
            if dev not in maps or s is None or e is None:
                continue
            # the last fit places the op; the first and last fits disagree by about their error where it ran
            scale, shift = maps[dev][-1]
            first_scale, first_shift = maps[dev][0]
            sync_error[index] = (
                abs((s * scale + shift) - (s * first_scale + first_shift))
                / cycles_per_ns
            )
            s = (s * scale + shift) / cycles_per_ns
            e = (e * scale + shift) / cycles_per_ns
        if s is None or e is None:
            continue
        key = (
            base,
            _num(row, "METAL TRACE ID"),
            _num(row, "METAL TRACE REPLAY SESSION ID"),
        )
        groups[key][dev] = index
        spans[index] = (s, e)

    columns = {
        c: pd.Series([None] * len(df), index=df.index, dtype=object)
        for c in (
            XCHIP_LAST_IN,
            XCHIP_IN_SPREAD,
            XCHIP_LAST_OUT,
            XCHIP_OUT_SPREAD,
            XCHIP_SELF_WAIT,
            XCHIP_SYNC_ERROR,
        )
    }
    for members in groups.values():
        if len(members) < 2:
            continue
        starts = {d: spans[i][0] for d, i in members.items()}
        ends = {d: spans[i][1] for d, i in members.items()}
        last_in = max(starts, key=starts.get)
        last_out = max(ends, key=ends.get)
        for d, i in members.items():
            columns[XCHIP_LAST_IN][i] = last_in
            columns[XCHIP_IN_SPREAD][i] = starts[last_in] - min(starts.values())
            columns[XCHIP_LAST_OUT][i] = last_out
            columns[XCHIP_OUT_SPREAD][i] = ends[last_out] - min(ends.values())
            columns[XCHIP_SELF_WAIT][i] = starts[last_in] - starts[d]
            # the largest sync error of the chips compared bounds the comparison
            columns[XCHIP_SYNC_ERROR][i] = max(
                sync_error.get(j, 0.0) for j in members.values()
            )
    for c, series in columns.items():
        df[c] = series
    return used


# ---------------------------------------------------------------- verdict

# Perf-counter columns (process_ops_logs, from --profiler-capture-perf-counters): mean over the cores that ran a
# compute kernel. Data-movement-only cores have no counters.
FPU_UTIL = "FPU Util Avg (%)"
SFPU_UTIL = "SFPU Util Avg (%)"
MATH_UTIL = "MATH Util Avg (%)"
PACK_UTIL = "Packer Utilization Avg (%)"
UNPACK_WAIT = "Math Waiting on Unpack (T1) Avg (%)"
STALLS = (
    "Thread 0 Stall Rate Avg (%)",
    "Thread 1 Stall Rate Avg (%)",
    "Thread 2 Stall Rate Avg (%)",
)
# Sum zones (--enable-sum-profiling): compute-side CB waits summed over cores
CB_WAITS = ("DEVICE COMPUTE CB WAIT FRONT [ns]", "DEVICE COMPUTE CB RESERVE BACK [ns]")

BOUND_THRESHOLD_PCT = 65.0
WAIT_CHIP_MIN_NS = 2000.0
WAIT_CHIP_SHARE = 0.2
CB_WAIT_SHARE_PCT = 30.0
DECLARED_MISMATCH = 0.2


def counter_values(row) -> Dict[str, Optional[float]]:
    stalls = [v for v in (_num(row, c) for c in STALLS) if v is not None]
    cb = [v for v in (_num(row, c) for c in CB_WAITS) if v is not None]
    duration = _num(row, "DEVICE KERNEL DURATION [ns]")
    cores = _num(row, "CORE COUNT")
    cb_share = None
    if cb and duration and cores:
        cb_share = min(100.0, sum(cb) / (duration * cores) * 100)
    return {
        "fpu": _num(row, FPU_UTIL),
        "sfpu": _num(row, SFPU_UTIL),
        "math": _num(row, MATH_UTIL),
        "pack": _num(row, PACK_UTIL),
        "unpack_wait": _num(row, UNPACK_WAIT),
        "stall": max(stalls) if stalls else None,
        "cb_wait": cb_share,
    }


def bound_by(
    device,
    duration_ns,
    gap_ns,
    dram_pct,
    flops_pct,
    counters,
    last_in,
    self_lead_ns,
    declared_bound="",
    sync_error_ns=None,
) -> str:
    """One-line verdict for an op, from the measured figures first and the declared roofline last."""
    if dram_pct is not None and dram_pct >= BOUND_THRESHOLD_PCT:
        return f"DRAM ({dram_pct:.0f}% of peak)"
    math = max(
        [
            v
            for v in (counters.get("math"), counters.get("fpu"), counters.get("sfpu"))
            if v is not None
        ],
        default=None,
    )
    if math is not None and math >= BOUND_THRESHOLD_PCT:
        return f"math ({math:.0f}% util)"
    if flops_pct is not None and flops_pct >= BOUND_THRESHOLD_PCT:
        return f"math ({flops_pct:.0f}% of peak FLOPs)"
    if (
        last_in is not None
        and device is not None
        and last_in != device
        and self_lead_ns is not None
        and duration_ns
        and self_lead_ns
        >= max(
            WAIT_CHIP_MIN_NS, WAIT_CHIP_SHARE * duration_ns, 2 * (sync_error_ns or 0.0)
        )
    ):
        return f"waiting on chip {int(last_in)} ({self_lead_ns / 1000:.1f} µs)"
    if counters.get("cb_wait") is not None and counters["cb_wait"] >= CB_WAIT_SHARE_PCT:
        return f"waiting on CB ({counters['cb_wait']:.0f}% of compute time)"
    if gap_ns is not None and duration_ns and gap_ns > duration_ns:
        return f"dispatch (gap {gap_ns / 1000:.1f} µs > op {duration_ns / 1000:.1f} µs)"
    if declared_bound:
        return f"{declared_bound} (modelled roofline)"
    return "unexplained"
