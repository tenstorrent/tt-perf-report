#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import csv
import sys
import argparse
import re
from typing import Any, Optional, Union
from collections import defaultdict
import pandas as pd

# Global variable to store color preference
color_output = None  # None means auto-detect, True forces color, False forces no color


def set_color_output(force_color, force_no_color):
    global color_output
    if force_no_color:
        color_output = False
    elif force_color:
        color_output = True
    else:
        color_output = None  # Auto-detect


def colored(text, color):
    if color_output is None:
        should_color = sys.stdout.isatty()
    else:
        should_color = color_output

    if should_color and color:
        colors = {
            "grey": "\033[38;5;8m",
            "red": "\033[38;5;9m",
            "green": "\033[38;5;10m",
            "yellow": "\033[38;5;11m",
            "blue": "\033[38;5;12m",
            "magenta": "\033[38;5;13m",
            "orange": "\033[38;5;208m",
            "cyan": "\033[38;5;14m",
            "white": "\033[38;5;15m",
            "end": "\033[0m",
        }
        return f"{colors[color]}{text}{colors['end']}"
    else:
        return text


def tflops_per_core(math_fidelity):
    """Source: https://tenstorrent.com/assets/one-pagers/08.01.24_Wormhole.pdf"""
    if math_fidelity == "HiFi4":
        return 74 / 72
    elif math_fidelity == "HiFi2":
        return 148 / 72
    elif math_fidelity == "LoFi":
        return 262 / 72
    else:
        assert False, f"Unknown math fidelity: {math_fidelity}"


class Cell:
    def __init__(self, value: Any, unit: Optional[str] = None, decimals=0, color=None):
        self.raw_value = value
        self.unit = unit
        self.decimals = decimals
        self.color = color

    def format(self):
        if self.raw_value is None or pd.isna(self.raw_value):
            return ""

        if isinstance(self.raw_value, str) and ("Matmul" in self.raw_value or "OptimizedConvNew" in self.raw_value):
            parts = self.raw_value.split(maxsplit=1)
            op_name = parts[0]
            size = parts[1] if len(parts) > 1 else ""
            formatted = f"{colored(op_name, self.color) if self.color else op_name} {colored(size, 'grey')}"
        else:
            try:
                formatted = f"{float(self.raw_value):,.{self.decimals}f}"
            except (ValueError, TypeError):
                formatted = str(self.raw_value)

            if self.color:
                formatted = colored(formatted, self.color)

        if self.unit:
            formatted += f" {colored(self.unit, 'grey')}"

        return formatted

    def __str__(self):
        return self.format()


def filter_by_signpost(df, signpost=None, ignore_signposts=False):
    signpost_rows = df[df["OP TYPE"] == "signpost"]

    if ignore_signposts:
        print(colored("Ignoring all signposts. Using the entire file for analysis.", "cyan"))
        return df

    if signpost:
        if signpost in signpost_rows["OP CODE"].values:
            print(colored(f"Using specified signpost: {signpost}", "cyan"))
            return df[df["OP CODE"].eq(signpost).cummax()].iloc[1:]
        print(colored(f"Specified signpost '{signpost}' not found. Defaulting to the last signpost.", "yellow"))

    if signpost_rows.empty:
        print(colored("No signposts found in the file. Using the entire file for analysis.", "yellow"))
        return df

    last_signpost = signpost_rows.iloc[-1]["OP CODE"]
    print(colored(f"Detected signposts: {', '.join(signpost_rows['OP CODE'])}", "cyan"))
    print(colored(f"Using last signpost: {last_signpost} for analysis.", "cyan"))
    return df[df["OP CODE"].eq(last_signpost).cummax()].iloc[1:]


def get_datatype_size(datatype):
    match = re.search(r"\d+", datatype)
    return int(match.group()) / 8 if match else 4


def visible_length(s):
    return len(re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", s))


def pad_string(string, length, align="left"):
    visible_len = visible_length(string)
    padding = " " * (length - visible_len)
    return padding + string if align == "right" else string + padding


def evaluate_fidelity(input_0_datatype, input_1_datatype, output_datatype, math_fidelity):
    mantissa_bits = {"BFLOAT16": 8, "BFLOAT8_B": 7, "BFLOAT4_B": 3}
    in0_bits = mantissa_bits[input_0_datatype]  # activations -> srcB (7 bits)
    in1_bits = mantissa_bits[input_1_datatype]  # weights -> srcA (5 bits)
    out_bits = mantissa_bits[output_datatype]
    if in0_bits == 8 and out_bits >= 7:
        if math_fidelity == "HiFi4":
            return (
                "sufficient",
                "HiFi2 may also work, it discards the lowest bit of the activations and has 2x the throughput of HiFi4",
            )
        elif math_fidelity == "HiFi2":
            return "too_low", "If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy"
        elif math_fidelity == "LoFi":
            return "too_low", "Use HiFi2 or HiFi4 with BF16 activations for improved accuracy"
        else:
            assert False, f"Unknown math fidelity: {math_fidelity}"
    elif in0_bits == 8 and out_bits == 3:
        if math_fidelity == "HiFi4":
            return (
                "too_high",
                "HiFi2 is very likely to work for BFP8 output; it discards the lowest bit of the activations and has 2x the throughput of HiFi4",
            )
        elif math_fidelity == "HiFi2":
            return (
                "sufficient",
                "LoFi might also be sufficient with BFP4 output and has almost 2x the throughput of HiFi2",
            )
        elif math_fidelity == "LoFi":
            return (
                "too_low",
                "HiFi2 may give better accuracy for large matmuls with many intermediate accumulations",
            )
        else:
            assert False, f"Unknown math fidelity: {math_fidelity}"
    elif in1_bits >= 7 and out_bits >= 7:
        if math_fidelity == "HiFi4":
            return "too_high", "HiFi2 is sufficient for BFP8 multiplication and has 2x the throughput of HiFi4"
        elif math_fidelity == "HiFi2":
            return "sufficient", None
        elif math_fidelity == "LoFi":
            return "too_low", "HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights"
        else:
            assert False, f"Unknown math fidelity: {math_fidelity}"
    elif in1_bits >= 7 and out_bits == 3:
        if math_fidelity == "HiFi4":
            return "too_high", "HiFi2 is sufficient for BFP8 multiplication and has 2x the throughput of HiFi4"
        elif math_fidelity == "HiFi2":
            return (
                "sufficient",
                "LoFi might also be sufficient with BFP4 output and has almost 2x the throughput of HiFi2",
            )
        elif math_fidelity == "LoFi":
            return (
                "too_low",
                "HiFi2 may give slightly better accuracy for large matmuls with many intermediate accumulations",
            )
        else:
            assert False, f"Unknown math fidelity: {math_fidelity}"
    elif in1_bits == 3:
        if math_fidelity == "LoFi":
            return "sufficient", None
        else:
            return "too_high", "LoFi is sufficient with BFP4 weights, use it for much higher throughput"
    else:
        print(f"Using {math_fidelity} for {input_0_datatype}/{input_1_datatype} inputs and {output_datatype} output")
        print(f"Bits: {in0_bits}/{in1_bits}/{out_bits}")
        return (
            "unknown",
            f"Using {math_fidelity} for {input_0_datatype}/{input_1_datatype} inputs and {output_datatype} output",
        )


def analyze_matmul(row):
    input_0_from_dram = "DRAM" in row["INPUT_0_MEMORY"]
    input_1_from_dram = "DRAM" in row["INPUT_1_MEMORY"]

    total_data_size_bytes = 0
    if input_0_from_dram:
        total_data_size_bytes += (
            row["INPUT_0_W"]
            * row["INPUT_0_Y"]
            * row["INPUT_0_Z"]
            * row["INPUT_0_X"]
            * get_datatype_size(row["INPUT_0_DATATYPE"])
        )
    if input_1_from_dram:
        total_data_size_bytes += (
            row["INPUT_1_W"]
            * row["INPUT_1_Y"]
            * row["INPUT_1_Z"]
            * row["INPUT_1_X"]
            * get_datatype_size(row["INPUT_1_DATATYPE"])
        )

    # Always include output if it's written to DRAM
    if "DRAM" in row["OUTPUT_0_MEMORY"]:
        total_data_size_bytes += (
            row["OUTPUT_0_W"]
            * row["OUTPUT_0_Y"]
            * row["OUTPUT_0_Z"]
            * row["OUTPUT_0_X"]
            * get_datatype_size(row["OUTPUT_0_DATATYPE"])
        )

    duration_s = row["DEVICE KERNEL DURATION [ns]"] * 1e-9
    dram_speed_gb_s = (total_data_size_bytes / duration_s) / 1e9 if total_data_size_bytes > 0 else None

    core_count = row["CORE COUNT"]
    math_fidelity = row["MATH FIDELITY"]

    # Check for DRAM-sharded program config
    attributes = row["ATTRIBUTES"] if pd.notna(row["ATTRIBUTES"]) else ""
    is_dram_sharded = "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" in attributes

    # Override core count for DRAM-sharded matmuls
    if is_dram_sharded:
        core_count = 12

    peak_flops_value = tflops_per_core(math_fidelity) * 1e12 * core_count

    M, K, N = int(row["INPUT_0_Y"]), int(row["INPUT_0_X"]), int(row["INPUT_1_X"])
    W, Z = int(row["INPUT_0_W"]), int(row["INPUT_0_Z"])

    flops = (M * K * N * W * Z * 2) / duration_s

    size = f"{M} x {K} x {N}"
    memory_info = f"({row['INPUT_0_DATATYPE']} {row['INPUT_0_MEMORY'].replace('DEV_0_', '')} @ {row['INPUT_1_DATATYPE']} {row['INPUT_1_MEMORY'].replace('DEV_0_', '')} => {row['OUTPUT_0_DATATYPE']} {row['OUTPUT_0_MEMORY'].replace('DEV_0_', '')})"

    dram_percentage = (dram_speed_gb_s / 288) * 100 if dram_speed_gb_s is not None else None
    flops_percentage = (flops / peak_flops_value) * 100

    return (
        dram_speed_gb_s,
        dram_percentage,
        flops,
        flops_percentage,
        size,
        memory_info,
        math_fidelity,
        is_dram_sharded,
        core_count,  # Return the potentially adjusted core count
    )

def analyze_conv(row):
    duration_s = row["DEVICE KERNEL DURATION [ns]"] * 1e-9

    core_count = 64 # we decided to normalize to the max core count
    math_fidelity = row["MATH FIDELITY"]

    # Check for DRAM-sharded program config
    attributes = row["ATTRIBUTES"] if pd.notna(row["ATTRIBUTES"]) else ""

    peak_flops_value = tflops_per_core(math_fidelity) * 1e12 * core_count

    NHW = int(row["OUTPUT_0_Y"])
    CH_IN = int(row["INPUT_0_X"])
    W = [int(x) for x in (attributes.split("window_hw")[1].split("; ")[0][2:-1].split(";"))]
    CH_OUT = int(row["INPUT_1_X"])

    M, K, N = NHW, CH_IN * W[0] * W[1], CH_OUT
    flops = (M * K * N * 2) / duration_s

    size = f"{M} x {K} x {N}"
    memory_info = f"({row['INPUT_0_DATATYPE']} {row['INPUT_0_MEMORY'].replace('DEV_0_', '')} @ {row['INPUT_1_DATATYPE']} {row['INPUT_1_MEMORY'].replace('DEV_0_', '')} => {row['OUTPUT_0_DATATYPE']} {row['OUTPUT_0_MEMORY'].replace('DEV_0_', '')})"

    flops_percentage = (flops / peak_flops_value) * 100

    return (
        flops,
        flops_percentage,
        size,
        memory_info,
        math_fidelity
    )

def analyze_op(row, prev_row):
    op_code = Cell(row["OP CODE"])
    cores = Cell(int(row["CORE COUNT"]) if pd.notna(row["CORE COUNT"]) else None)
    device_time = Cell(
        row["DEVICE KERNEL DURATION [ns]"] / 1000 if pd.notna(row["DEVICE KERNEL DURATION [ns]"]) else None,
        unit="us",
        decimals=0,
    )

    if prev_row is not None and pd.notna(prev_row["OP TO OP LATENCY [ns]"]):
        op_to_op_gap = Cell(
            row["OP TO OP LATENCY [ns]"] / 1000 if pd.notna(row["OP TO OP LATENCY [ns]"]) else None,
            unit="us",
            decimals=0,
        )
    else:
        op_to_op_gap = Cell(None, unit="us", decimals=0)

    def get_entry(k: str) -> Union[str, None]:
        return row[k] if k in row else None

    output_datatype = get_entry("OUTPUT_0_DATATYPE")
    input_0_datatype = get_entry("INPUT_0_DATATYPE")
    input_1_datatype = get_entry("INPUT_1_DATATYPE")
    output_datatype_cell = Cell(output_datatype)
    input_0_datatype_cell = Cell(input_0_datatype)
    input_1_datatype_cell = Cell(input_1_datatype)
    short_name = lambda n: {"BFLOAT16": "BF16", "BFLOAT8_B": "BFP8", "BFLOAT4_B": "BFP4"}.get(n, n)

    dram_speed = Cell(None, unit="GB/s", decimals=0)
    dram_percentage = Cell(None, unit="%", decimals=1)
    flops = Cell(None, unit="TFLOPs", decimals=1)
    flops_percentage = Cell(None, unit="%", decimals=1)

    math_fidelity = ""
    math_fidelity += f"{short_name(input_0_datatype)}" if pd.notna(input_0_datatype) else ""
    math_fidelity += f", {short_name(input_1_datatype)}" if pd.notna(input_1_datatype) else ""
    math_fidelity += f" => {short_name(output_datatype)}" if pd.notna(output_datatype) else ""
    math_fidelity_cell = Cell(math_fidelity.strip())

    is_dram_sharded = False

    if "Matmul" in op_code.raw_value:
        (
            dram_speed,
            dram_percentage,
            flops,
            flops_percentage,
            size,
            memory_info,
            math_fidelity,
            is_dram_sharded,
            adjusted_core_count,  # Get the potentially adjusted core count
        ) = analyze_matmul(row)
        op_code = Cell(f"{op_code.raw_value} {size}")
        dram_speed = Cell(dram_speed, unit="GB/s", decimals=0)
        dram_percentage = Cell(dram_percentage, unit="%", decimals=1)
        flops = Cell(flops / 1e12 if pd.notna(flops) else None, unit="TFLOPs", decimals=1)
        flops_percentage = Cell(flops_percentage, unit="%", decimals=1)
        cores.raw_value = adjusted_core_count

        math_fidelity_cell = Cell(
            f"{math_fidelity} {short_name(input_0_datatype)} x {short_name(input_1_datatype)} => {short_name(output_datatype)}".strip()
            if math_fidelity
            else None
        )
    elif "OptimizedConvNew" in op_code.raw_value:
        (
            flops,
            flops_percentage,
            size,
            memory_info,
            math_fidelity,
        ) = analyze_conv(row)
        op_code = Cell(f"{op_code.raw_value} {size}")
        dram_speed = Cell(None, unit="GB/s", decimals=0)
        dram_percentage = Cell(None, unit="%", decimals=1)
        flops = Cell(flops / 1e12 if pd.notna(flops) else None, unit="TFLOPs", decimals=1)
        flops_percentage = Cell(flops_percentage, unit="%", decimals=1)
        math_fidelity_cell = Cell(
            f"{math_fidelity} {short_name(input_0_datatype)} x {short_name(input_1_datatype)} => {short_name(output_datatype)}".strip()
            if math_fidelity
            else None
        )

    output = {
        "ID": None,
        "Bound": Cell(""),
        "OP Code": op_code,
        "Device Time": device_time,
        "Op-to-Op Gap": op_to_op_gap,
        "Cores": cores,
        "DRAM": dram_speed,
        "DRAM %": dram_percentage,
        "FLOPs": flops,
        "FLOPs %": flops_percentage,
        "Math Fidelity": math_fidelity_cell,
        "Output Datatype": output_datatype_cell,
        "Input 0 Datatype": input_0_datatype_cell,
        "Input 1 Datatype": input_1_datatype_cell,
        "DRAM Sharded": Cell(is_dram_sharded),
    }

    input_0_memory = Cell(row["INPUT_0_MEMORY"] if pd.notna(row["INPUT_0_MEMORY"]) else None)

    # Extract program config details
    attributes = row["ATTRIBUTES"] if pd.notna(row["ATTRIBUTES"]) else ""
    in0_block_w = Cell(None)
    out_subblock_h = Cell(None)
    out_subblock_w = Cell(None)

    if "program_config" in attributes:
        match = re.search(r"in0_block_w=(\d+)", attributes)
        if match:
            in0_block_w = Cell(int(match.group(1)))

        match = re.search(r"out_subblock_h=(\d+)", attributes)
        if match:
            out_subblock_h = Cell(int(match.group(1)))

        match = re.search(r"out_subblock_w=(\d+)", attributes)
        if match:
            out_subblock_w = Cell(int(match.group(1)))

    output["Input 0 Memory"] = input_0_memory
    output["Inner Dim Block Size"] = in0_block_w
    output["Output Subblock H"] = out_subblock_h
    output["Output Subblock W"] = out_subblock_w

    return output, op_to_op_gap.raw_value


def add_derived_columns(rows):
    total_duration = sum(
        op_data["Device Time"].raw_value for op_data in rows if op_data["Device Time"].raw_value is not None
    ) + sum(op_data["Op-to-Op Gap"].raw_value for op_data in rows if op_data["Op-to-Op Gap"].raw_value is not None)
    for op_data in rows:
        device_time = op_data["Device Time"].raw_value if op_data["Device Time"].raw_value is not None else 0
        op_to_op_gap = op_data["Op-to-Op Gap"].raw_value if op_data["Op-to-Op Gap"].raw_value is not None else 0
        op_data["Total %"] = Cell(((device_time + op_to_op_gap) / total_duration) * 100, unit="%", decimals=1)
        if op_data["Device Time"].raw_value is None and op_data["Op-to-Op Gap"].raw_value is None:
            op_data["Total %"].raw_value = None

        if "Matmul" in op_data["OP Code"].raw_value:
            dram_percentage = op_data["DRAM %"].raw_value
            flops_percentage = op_data["FLOPs %"].raw_value
            if dram_percentage and flops_percentage:
                if dram_percentage >= 65 and flops_percentage >= 65:
                    op_data["Bound"] = Cell("BOTH")
                elif dram_percentage >= 65:
                    op_data["Bound"] = Cell("DRAM")
                elif flops_percentage >= 65:
                    op_data["Bound"] = Cell("FLOP")
                else:
                    op_data["Bound"] = Cell("SLOW")
        elif "(torch)" in op_data["OP Code"].raw_value:
            op_data["Bound"] = Cell("HOST")


def print_row(row, col_widths, headers):
    def format_cell(header, cell):
        # Avoid thousand separators for ID column
        text = colored(str(cell.raw_value), cell.color) if header == "ID" else str(cell)
        return pad_string(text, col_widths[headers.index(header)], align="left" if header == "OP Code" else "right")

    print("  ".join(format_cell(header, row[header]) for header in headers))


def color_row(op_data, percentage, min_percentage):
    if percentage is not None and percentage < min_percentage:
        for v in op_data.values():
            v.color = "grey"
    else:
        op_colors = {
            "(torch)": "red",
            "Matmul": "magenta",
            "OptimizedConvNew" : "orange",
            "LayerNorm": "cyan",
            "AllGather": "cyan",
            "AllReduce": "cyan",
            "ScaledDotProductAttentionDecode": "blue",
            "ScaledDotProductAttentionGQADecode": "blue",
            "NlpCreateHeadsDeviceOperation": "blue",
            "NLPConcatHeadsDecodeDeviceOperation": "blue",
            "UpdateCache": "blue",
        }
        for op, color in op_colors.items():
            if op in op_data["OP Code"].raw_value:
                op_data["OP Code"].color = color
                break
        else:
            op_data["OP Code"].color = "white"

        num_cores = op_data["Cores"].raw_value
        if num_cores is not None:
            if num_cores < 10:
                op_data["Cores"].color = "red"
            elif num_cores == 64:
                op_data["Cores"].color = "green"
        else:
            op_data["Cores"].color = "grey"

        if op_data["Bound"].raw_value == "DRAM":
            op_data["Bound"].color = "green"
            op_data["DRAM"].color = "green"
            op_data["DRAM %"].color = "green"
        elif op_data["Bound"].raw_value == "FLOP":
            op_data["Bound"].color = "green"
            op_data["FLOPs"].color = "green"
            op_data["FLOPs %"].color = "green"
        elif op_data["Bound"].raw_value == "SLOW":
            op_data["Bound"].color = "yellow"
            dram_percentage = op_data["DRAM %"].raw_value
            flops_percentage = op_data["FLOPs %"].raw_value
            if dram_percentage is not None and flops_percentage is not None:
                if dram_percentage > flops_percentage:
                    op_data["DRAM"].color = "yellow"
                    op_data["DRAM %"].color = "yellow"
                else:
                    op_data["FLOPs"].color = "yellow"
                    op_data["FLOPs %"].color = "yellow"
        elif op_data["Bound"].raw_value == "HOST":
            op_data["Bound"].color = "red"

        if op_data["Op-to-Op Gap"].raw_value is not None and op_data["Op-to-Op Gap"].raw_value > 6.5:
            op_data["Op-to-Op Gap"].color = "red"

        if ("Matmul" in op_data["OP Code"].raw_value 
            or "OptimizedConvNew" in op_data["OP Code"].raw_value) and op_data["Math Fidelity"].raw_value:
            math_fidelity = op_data["Math Fidelity"].raw_value.split()[0]
            input_0_datatype = op_data["Input 0 Datatype"].raw_value
            input_1_datatype = op_data["Input 1 Datatype"].raw_value
            output_datatype = op_data["Output Datatype"].raw_value

            fidelity_evaluation, _ = evaluate_fidelity(
                input_0_datatype, input_1_datatype, output_datatype, math_fidelity
            )

            if fidelity_evaluation == "sufficient":
                op_data["Math Fidelity"].color = "green"
            elif fidelity_evaluation == "too_high":
                op_data["Math Fidelity"].color = "red"
            elif fidelity_evaluation == "too_low":
                op_data["Math Fidelity"].color = "cyan"
            else:
                op_data["Math Fidelity"].color = "white"

    return op_data


def print_performance_table(rows, headers, col_widths, device_ops, host_ops):
    print("\n🚀 Performance Report 🚀\n========================\n")

    print("  ".join(pad_string(header, col_widths[i], align="left") for i, header in enumerate(headers)))
    print("-" * sum(col_widths) + "-" * (len(headers) - 1) * 2)

    for idx, op_data in enumerate(rows):
        print_row(op_data, col_widths, headers)

    print("-" * (sum(col_widths) + (len(headers) - 1) * 2))

    total_device_time = sum(
        op_data["Device Time"].raw_value for op_data in rows if op_data["Device Time"].raw_value is not None
    )
    total_visible_gap = sum(
        op_data["Op-to-Op Gap"].raw_value for op_data in rows if op_data["Op-to-Op Gap"].raw_value is not None
    )
    total_row = {
        "ID": Cell(""),
        "Total %": Cell(100.0, unit="%", decimals=1),
        "Bound": Cell(""),
        "OP Code": Cell(f"{device_ops} device ops, {host_ops} host ops"),
        "Device Time": Cell(total_device_time, unit="us", decimals=0),
        "Op-to-Op Gap": Cell(total_visible_gap, unit="us", decimals=0),
    }
    for header in headers:
        if header not in total_row:
            total_row[header] = Cell("")
    print_row(
        {k: Cell(v.raw_value, v.unit, v.decimals, color="grey") for k, v in total_row.items()}, col_widths, headers
    )


def print_advice_section(rows, headers, col_widths):
    print("\n💡 Advice 💡\n============\n")

    print_fallback_advice(rows, headers, col_widths)
    print_op_to_op_gap_advice(rows, headers, col_widths)
    print_matmul_advice(rows, headers, col_widths)


def print_fallback_advice(rows, headers, col_widths):
    host_ops = [op_data for op_data in rows if "(torch)" in op_data["OP Code"].raw_value]
    if host_ops:
        print("Fallback\n--------")
        for op_data in host_ops:
            print_row(op_data, col_widths, headers)
        print("\nThese ops should be moved to run on device.\n")


def print_op_to_op_gap_advice(rows, headers, col_widths):
    high_gap_ops = [
        (idx + 1, op_data)
        for idx, op_data in enumerate(rows)
        if op_data["Op-to-Op Gap"].raw_value is not None and op_data["Op-to-Op Gap"].raw_value > 6.5
    ]

    if high_gap_ops:
        print("High Op-to-Op Gap\n----------------")
        for idx, op_data in high_gap_ops:
            print_row(op_data, col_widths, headers)
        max_gap_overhead = sum(op_data["Op-to-Op Gap"].raw_value - 6 for _, op_data in high_gap_ops)

        total_duration = sum(
            op_data["Device Time"].raw_value for op_data in rows if op_data["Device Time"].raw_value is not None
        ) + sum(op_data["Op-to-Op Gap"].raw_value for op_data in rows if op_data["Op-to-Op Gap"].raw_value is not None)

        percentage_saved = (max_gap_overhead / total_duration) * 100
        print(
            f"\nThese ops have a >6us gap since the previous operation. Running with tracing could save {max_gap_overhead:.0f} us ({percentage_saved:.1f}% of overall time)"
        )
        print(
            "Alternatively ensure device is not waiting for the host and use device.enable_async(True). Experts can try moving runtime args in the kernels to compile-time args.\n"
        )


def is_matmul_op(op_data):
    return "Matmul" in op_data["OP Code"].raw_value


def print_matmul_advice(rows, headers, col_widths):
    matmul_ops = [op_data for op_data in rows if is_matmul_op(op_data)]

    if matmul_ops:
        print("Matmul Optimization\n-------------------")
        for op_data in matmul_ops:
            print_row(op_data, col_widths, headers)
            advice = generate_matmul_advice(op_data)
            color = "grey" if op_data["OP Code"].color == "grey" else "white"

            if advice:
                for item in advice:
                    print(colored(f"- {item}", color))
            else:
                print(colored("✅ Optimized", color))
            print()  # Add a blank line between matmuls


def generate_matmul_advice(op_data):
    advice = []

    math_fidelity = (
        op_data["Math Fidelity"].raw_value.split()[0] if op_data["Math Fidelity"].raw_value else None
    )
    output_datatype = op_data["Output Datatype"].raw_value
    input_0_datatype = op_data["Input 0 Datatype"].raw_value
    input_1_datatype = op_data["Input 1 Datatype"].raw_value
    cores = op_data["Cores"].raw_value
    fidelity_evaluation, fidelity_advice = evaluate_fidelity(
        input_0_datatype, input_1_datatype, output_datatype, math_fidelity
    )

    if op_data["Bound"].raw_value in ["DRAM", "BOTH"]:
        if not op_data["DRAM Sharded"].raw_value:
            advice.append(
                "Try a DRAM-sharded program config (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig) to improve throughput further"
            )
        if fidelity_evaluation == "too_low" and op_data["FLOPs %"].raw_value < 40:
            advice.append(fidelity_advice)
        if fidelity_evaluation == "too_high":
            advice.append(fidelity_advice)
    elif op_data["Bound"].raw_value in ["FLOP", "BOTH"]:
        if cores < 64:
            advice.append(f"Increase grid size (currently using {cores})")
        if fidelity_evaluation == "too_high":
            advice.append(fidelity_advice)
    elif op_data["Bound"].raw_value == "SLOW":
        input_0_memory = op_data["Input 0 Memory"].raw_value
        if input_0_memory and "L1" not in input_0_memory:
            advice.append(f"If possible place input 0 in L1 (currently in {input_0_memory})")

        inner_dim_block = op_data["Inner Dim Block Size"].raw_value
        out_h = op_data["Output Subblock H"].raw_value
        out_w = op_data["Output Subblock W"].raw_value

        if inner_dim_block is None and out_h is None and out_w is None:
            advice.append(
                "No program_config specified, try using one to override in0_block_w and out_subblock_h/w"
            )
        else:
            all_good = True
            if inner_dim_block is not None:
                if inner_dim_block < 2:
                    advice.append(f"in0_block_w={inner_dim_block} is small, try in0_block_w=2 or above")
                    all_good = False
            else:
                advice.append("No inner dim block size found")
                all_good = False

            if out_h is not None and out_w is not None:
                out_area = out_h * out_w
                if out_area < 2:
                    advice.append(
                        f"Output subblock {out_h}x{out_w} is small, try out_subblock_h * out_subblock_w >= 2 if possible"
                    )
                    all_good = False
            else:
                advice.append("No output subblock size found")
                all_good = False

            if all_good:
                advice.append(
                    f"in0_block_w={inner_dim_block} and output subblock {out_h}x{out_w} look good 🤷"
                )
            if fidelity_advice:
                advice.append(fidelity_advice)

    return advice


def merge_device_rows(df):
    block_by_device = defaultdict(list)

    for _, row in df.iterrows():
        op_name = row["OP CODE"]
        op_type = row["OP TYPE"]

        if op_type == "tt_dnn_device":
            device_id = int(row["DEVICE ID"])
            block_by_device[device_id].append((op_name, row.to_dict()))

    device_ids = sorted(block_by_device.keys())
    merged_blocks = []

    for blocks in zip(*[block_by_device[device_id] for device_id in device_ids]):
        op_name = blocks[0][0]

        if "AllGather" in op_name or "ReduceScatter" in op_name:
            # For collective ops, take the row with minimum duration
            min_duration_block = min(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(min_duration_block[1])
        else:
            # For non-collective ops, take the row with maximum duration
            max_duration_block = max(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(max_duration_block[1])

    return pd.DataFrame(merged_blocks)


def parse_id_range(id_range_str):
    if id_range_str is None:
        return None

    parts = id_range_str.split("-")
    if len(parts) != 2:
        raise ValueError("Invalid ID range format")

    start = int(parts[0].replace(",", "")) if parts[0] else None
    end = int(parts[1].replace(",", "")) if parts[1] else None

    return (start, end)


def filter_by_id_range(rows, id_range):
    if id_range:
        start, end = id_range
        if start is None:
            print(colored(f"Filtering rows with IDs up to {end}", "cyan"))
            filtered_rows = [row for row in rows if row["ID"].raw_value <= end]
        elif end is None:
            print(colored(f"Filtering rows with IDs from {start} onwards", "cyan"))
            filtered_rows = [row for row in rows if row["ID"].raw_value >= start]
        else:
            print(colored(f"Filtering rows with IDs from {start} to {end}", "cyan"))
            filtered_rows = [row for row in rows if start <= row["ID"].raw_value <= end]

        # Reset the op-to-op gap for the first item in the filtered range
        if filtered_rows:
            filtered_rows[0]["Op-to-Op Gap"] = Cell(None, unit="us", decimals=0)

        return filtered_rows
    return rows


def main():
    args, id_range = parse_args()
    generate_perf_report(
        args.csv_file, args.signpost, args.ignore_signposts, args.min_percentage, id_range, args.csv, args.no_advice, args.tracing_mode, args.raw_op_codes,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="User-friendly Performance Report Analysis Tool")
    parser.add_argument("csv_file", type=str, help="Path to the performance report CSV file")
    parser.add_argument("--signpost", type=str, help="Specify a signpost to use for analysis", default=None)
    parser.add_argument(
        "--ignore-signposts", action="store_true", help="Ignore all signposts and use the entire file for analysis"
    )
    parser.add_argument(
        "--min-percentage", type=float, default=0.5, help="Minimum percentage for coloring (default: 0.5)"
    )
    parser.add_argument(
        "--id-range", type=str, help="Show only rows with IDs in the specified range (e.g., '5-10', '31-', or '-12')"
    )
    parser.add_argument("--color", action="store_true", help="Force colored output even when output is redirected")
    parser.add_argument("--no-color", action="store_true", help="Force output without color")
    parser.add_argument("--csv", type=str, help="Output filename for CSV format", metavar="OUTPUT_FILE")
    parser.add_argument("--no-advice", action="store_true", help="Only show the table section of the report")
    parser.add_argument("--tracing-mode", action="store_true", help="Do not sort when in tracing mode")
    parser.add_argument("--raw-op-codes", action="store_true", help="Include raw op codes in output")
    args = parser.parse_args()

    # Set the global color_output variable
    set_color_output(args.color, args.no_color)

    # Parse id_range
    try:
        id_range = parse_id_range(args.id_range)
    except ValueError:
        print(colored("Invalid --id-range format. Please use 'START-END', 'START-', or '-END'.", "red"))
        exit(1)

    return args, id_range


def generate_perf_report(csv_file, signpost, ignore_signposts, min_percentage, id_range, csv_output_file, no_advice, tracing_mode, raw_op_codes):
    df = pd.read_csv(csv_file, low_memory=False)

    # Add a column for original row numbers
    df["ORIGINAL_ROW"] = df.index + 2  # +2 to match Excel row numbers (1-based + header)

    # Sort the DataFrame by "HOST START TS" column
    # Sorting by HOST START TS is incorrect when using tracing mode since the tracing ops timestamps are the ones when captured and not executed
    if "HOST START TS" in df.columns and not tracing_mode:
        print(colored("Sorting CSV by 'HOST START TS' column...", "cyan"))
        df = df.sort_values(by="HOST START TS")
    else:
        print(colored("Warning: 'HOST START TS' column not found. CSV will not be sorted.", "yellow"))

    df = filter_by_signpost(df, signpost, ignore_signposts)

    # Check if the file contains multiple devices
    if "DEVICE ID" in df.columns and df["DEVICE ID"].nunique() > 1:
        print(colored(f"Detected data from {df['DEVICE ID'].nunique()} devices. Merging device data...", "cyan"))
        df = merge_device_rows(df)

    rows = []
    prev_row = None
    device_ops = 0
    host_ops = 0
    for _, row in df.iterrows():
        op_data, current_gap = analyze_op(row, prev_row)
        op_data["ID"] = Cell(row["ORIGINAL_ROW"])  # Use the original row number
        if raw_op_codes:
            op_data["Raw OP Code"] = Cell(row["OP CODE"])
        rows.append(op_data)
        prev_row = row

        # Count device and host ops
        if "(torch)" in op_data["OP Code"].raw_value:
            host_ops += 1
        else:
            device_ops += 1

    # Calculate total duration and add derived columns
    add_derived_columns(rows)

    # Filter rows based on id_range
    rows = filter_by_id_range(rows, id_range)

    # Recalculate derived columns after filtering
    add_derived_columns(rows)

    rows = [color_row(op_data, op_data["Total %"].raw_value, min_percentage) for op_data in rows]

    visible_headers = [
        "ID",
        "Total %",
        "Bound",
        "OP Code",
        "Device Time",
        "Op-to-Op Gap",
        "Cores",
        "DRAM",
        "DRAM %",
        "FLOPs",
        "FLOPs %",
        "Math Fidelity",
    ]

    if csv_output_file:
        all_headers = visible_headers + [
            "Output Datatype",
            "Input 0 Datatype",
            "Input 1 Datatype",
            "DRAM Sharded",
            "Input 0 Memory",
            "Inner Dim Block Size",
            "Output Subblock H",
            "Output Subblock W",
        ]
        if not no_advice:
            all_headers.append("Advice")
        if raw_op_codes:
            all_headers.append("Raw OP Code")
        print(colored(f"Writing CSV output to {csv_output_file}", "cyan"))
        with open(csv_output_file, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=all_headers)
            csv_writer.writeheader()
            for op_data in rows:
                row = {header: op_data[header].raw_value for header in all_headers if header in op_data}
                if not no_advice:
                    advice = generate_matmul_advice(op_data) if is_matmul_op(op_data) else ""
                    row["Advice"] = " • ".join(advice)
                csv_writer.writerow(row)
    else:
        col_widths = [
            max(max(visible_length(str(row[header])) for row in rows), visible_length(header))
            for header in visible_headers
        ]
        print_performance_table(rows, visible_headers, col_widths, device_ops, host_ops)
        if not no_advice:
            print_advice_section(rows, visible_headers, col_widths)


if __name__ == "__main__":
    main()
