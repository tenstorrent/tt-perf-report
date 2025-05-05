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
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Process and stack CSV data.")
    parser.add_argument("input_csv", type=str, help="Input CSV file")
    parser.add_argument(
        "--output_csv", type=str, default=None, help="Output CSV file (optional)"
    )
    parser.add_argument("-t", "--threshold", type=float, default=0.02, help="Threshold for significant data overlay", required=False)
    parser.add_argument("-s", "--stack_by_input_type", type=bool, default=False, help="Stack op by input0 type", required=False)
    args = parser.parse_args()

    input_csv = args.input_csv
    input_filename = os.path.basename(input_csv)
    input_dir = os.path.dirname(input_csv)
    output_csv = args.output_csv or os.path.join(input_dir, f"stacked_{input_filename}")

    try:
        # Read the input CSV
        df = pd.read_csv(input_csv)

        # Perform stacking operation: aggregate data by "OP Code", summing "Device Time" and counting operations
        if args.stack_by_input_type:
            df["OP Code Joined"] = df["OP Code"].str.split().str[0] + "_in0_" + df["Input 0 Memory"].str.split('_').str[-2] + "_" + df["Input 0 Memory"].str.split('_').str[-1]
        else:
            df["OP Code Joined"] = df["OP Code"].str.split().str[0]

        stacked_df = df.groupby("OP Code Joined").agg(
            Device_Time_Sum_us=("Device Time", "sum"),
            Ops_Count=("Device Time", "count"),
            Flops_min=("FLOPs %", "min"),
            Flops_max=("FLOPs %", "max"),
            Flops_mean=("FLOPs %", "mean"),
            Flops_std=("FLOPs %", "std"),
        ).reset_index()

        # Sort the stacked dataframe by "Device_Time_Sum_us" in descending order
        stacked_df = stacked_df.sort_values(by="Device_Time_Sum_us", ascending=False)

        # Save the stacked dataframe to the output CSV
        stacked_df.to_csv(output_csv, index=False, float_format="%.1f")
        print(f"Stacked CSV saved to {output_csv}")

        # Plot the data using matplotlib
        import matplotlib.pyplot as plt

        # Prepare data for the stacked bar plot
        op_codes = stacked_df["OP Code Joined"]
        device_time_sum = stacked_df["Device_Time_Sum_us"]
        total_sum = device_time_sum.sum()
        ops_count = stacked_df["Ops_Count"]
        flops_min = stacked_df["Flops_min"]
        flops_max = stacked_df["Flops_max"]
        flops_mean = stacked_df["Flops_mean"]
        flops_std = stacked_df["Flops_std"]

        # Create a stacked bar plot
        plt.figure(figsize=(6, 8), dpi=300)
        width = 0.5
        bottom = 0
        colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors

        for i, (label, duration, count, flop_min, flop_max, flop_mean, flop_std) in enumerate(zip(op_codes, device_time_sum, ops_count, flops_min, flops_max, flops_mean, flops_std)):
            color = colors[i % len(colors)]
            bar = plt.bar(1, duration, width, label=label, bottom=bottom, color=color)

            text = f"{label} total={duration:.1f}us; {count} ops"
            if not pd.isna(flop_mean):
                text += f"\n Util [{flop_min:.1f} - {flop_max:.1f}] {flop_mean:.1f} ± {flop_std:.1f} %"

            # Add overlay text if the data is significant
            if duration >= total_sum * args.threshold:
                plt.text(
                    bar[0].get_x() + bar[0].get_width() / 2,
                    bottom + duration / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white"
                )
            bottom += duration

        # Set plot labels and title
        plt.xlim(1 - width / 2 - 0.05, 1 + width / 2 + 0.05)
        plt.ylabel("Device Time [us]")
        plt.title(f"Stacked Device Time (Total: {total_sum:.1f} us)")
        plt.tight_layout()

        # Save the plot to a file
        output_png = os.path.splitext(output_csv)[0] + ".png"
        plt.savefig(output_png)
        print(f"Plot saved to {output_png}")
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()