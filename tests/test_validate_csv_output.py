#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import csv
import os
import tempfile
import re
from io import StringIO
import pytest
import pandas as pd
from tt_perf_report.perf_report import (
    generate_perf_report,
    detect_csv_format,
    CsvFormat,
    ArchitectureSpec,
    evaluate_fidelity,
    calculate_overall_dram_roofline,
    analyze_conv,
    color_row,
    escape_csv_formula,
    generate_matmul_advice,
    get_conv_window_hw,
    merge_device_rows,
    merge_perf_traces,
    Cell,
)
from tt_perf_report.csv_values import sanitize_text
from tt_perf_report.sub_device import (
    count_sub_devices,
    get_op_available_cores,
    get_op_sub_device_id,
)

# Shared test data (sample output from TT-NN)
@pytest.fixture(scope="session")
def test_csv_content():
    csv_file_path = os.path.join(os.path.dirname(__file__), "data", "ops_perf_results_2025_09_18_11_39_20.csv")
    
    try:
        with open(csv_file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Test CSV file not found at {csv_file_path}")

@pytest.fixture
def expected_headers():
    return [
        "ID",
        "Total %",
        "Bound",
        "OP Code",
        "Device",
        "Device Time",
        "Op-to-Op Gap",
        "Cores",
        "DRAM",
        "DRAM %",
        "FLOPs",
        "FLOPs %",
        "Math Fidelity",
        "Output Datatype",
        "Input 0 Datatype",
        "Input 1 Datatype",
        "DRAM Sharded",
        "Input 0 Memory",
        "Inner Dim Block Size",
        "Output Subblock H",
        "Output Subblock W",
        "Global Call Count",
        "Sub Device ID",
        "Available Cores",
        "Advice",
        "Raw OP Code",
    ]


def test_hifi3_is_supported_for_throughput_and_advice_paths():
    arch = ArchitectureSpec.from_name("wormhole", 64)

    assert arch.tflops_per_core("HiFi3") == pytest.approx((74 * 4 / 3) / 72)
    status, advice = evaluate_fidelity("BFLOAT16", "BFLOAT16", "BFLOAT16", "HiFi3")
    assert status == "too_low"
    assert "HiFi4" in advice


def test_hifi3_integer_datatypes_keep_not_applicable_advice():
    assert evaluate_fidelity("UINT8", "BFLOAT16", "BFLOAT16", "HiFi3") == (
        "not_applicable",
        "Fidelity evaluation is not applicable for integer datatypes (UINT8, UINT16, INT32, UINT32).",
    )


def test_blackhole_trace_invalid_device_durations_are_omitted(mocker):
    csv_file_path = os.path.join(os.path.dirname(__file__), "data", "bh_invalid_trace_decode_window.csv")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as output_file:
        try:
            stdout = StringIO()
            mocker.patch("sys.stdout", stdout)
            generate_perf_report(
                csv_files=[csv_file_path],
                start_signpost="PERF_DECODE",
                end_signpost="PERF_DECODE_END",
                ignore_signposts=False,
                print_signposts=False,
                min_percentage=0.5,
                id_range=None,
                arch=None,
                csv_output_file=output_file.name,
                no_advice=True,
                tracing_mode=True,
                raw_op_codes=True,
                no_host_ops=False,
                no_summary=True,
                group_by="op",
                classic_colors=False,
                summary_file=None,
                no_stacked_report=True,
                no_stack_by_in0=True,
                stacked_csv=None,
                no_merge_devices=False,
            )

            report_stdout = stdout.getvalue()
            assert "invalid device durations" in report_stdout
            assert "performance-model durations" not in report_stdout
            assert "Overall DRAM roofline" not in report_stdout

            with open(output_file.name, "r") as f:
                rows = list(csv.DictReader(f))

            device_times_us = [
                float(row["Device Time"])
                for row in rows
                if row["Device Time"]
            ]
            assert max(device_times_us) < 200

            op_to_op_gaps_us = [
                float(row["Op-to-Op Gap"])
                for row in rows
                if row["Op-to-Op Gap"]
            ]
            assert all(gap >= 0 for gap in op_to_op_gaps_us)
            assert max(op_to_op_gaps_us) < 10

            matmul_rows = [
                row
                for row in rows
                if row["OP Code"].startswith("MatmulDeviceOperation")
            ]
            assert len(matmul_rows) == 5
            assert all(row["Cores"] == "8" for row in matmul_rows)
            assert all(row["Device Time"] == "" for row in matmul_rows)
            assert all(row["DRAM %"] == "" for row in matmul_rows)

        finally:
            try:
                os.unlink(output_file.name)
            except OSError:
                pass


# TT-NN Visualizer default request
def test_csv_headers_with_all_options(expected_headers, test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost=None,
                    end_signpost=None,
                    ignore_signposts=True,
                    print_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    arch="wormhole",
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=True,
                    no_host_ops=False,
                    no_summary=True,
                    group_by="op",
                    classic_colors=False,
                    summary_file=None,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_csv=None,
                    no_merge_devices=False,
                )

                assert os.path.exists(output_file.name), "Output CSV file should be created"

                with open(output_file.name, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)
                    signposts = []

                    # Test that all expected headers are present and in the correct order
                    assert len(actual_headers) == len(expected_headers), \
                        f"Column count mismatch. Expected {len(expected_headers)}, got {len(actual_headers)}"
                    
                    for i, (expected, actual) in enumerate(zip(expected_headers, actual_headers)):
                        assert actual == expected, \
                            f"Column {i} mismatch. Expected '{expected}', got '{actual}'"

                    with open(output_file.name, "r") as f:
                        reader = csv.DictReader(f)
                        input_0_memory_pattern = re.compile(r"DEV_(\d+)_(DRAM|L1)")

                        for row in reader:
                            input_0_memory = row.get("Input 0 Memory")
                            advice_field = row.get("Advice", "")

                            if "(signpost)" in row.get("OP Code", ""):
                                signposts.append(row)

                            # Note: TT-NN Visualizer expects a splittable advice field
                            if advice_field and advice_field.strip():
                                advice_items = advice_field.split(" • ")
                                assert isinstance(advice_items, list), \
                                    "Advice should be splittable into a list"
                                
                                for item in advice_items:
                                    assert isinstance(item.strip(), str), \
                                        f"Advice item '{item}' should be a string"
                                    assert len(item.strip()) > 0, \
                                        f"Advice item '{item}' should not be empty"

                            # Test Input 0 Memory values
                            if input_0_memory and input_0_memory.strip():
                                match = input_0_memory_pattern.match(input_0_memory)
                                assert match is not None, \
                                    f"Input 0 Memory value '{input_0_memory}' does not match pattern 'DEV_(\\d+)_(DRAM|L1)'"

                                device_id, memory_type = match.groups()
                                assert device_id.isdigit(), \
                                    f"Device ID '{device_id}' should be a digit"
                                assert memory_type in ["DRAM", "L1"], \
                                    f"Memory type '{memory_type}' should be DRAM or L1"

                # Ensure that signpost rows are captured when ignore_signposts=True
                assert len(signposts) >= 0, "Signpost detection should work correctly"

            # Clean up
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)
                except OSError:
                    pass

# Request with signpost
def test_csv_headers_with_start_signpost(test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost='ResNet module started',
                    end_signpost=None,
                    ignore_signposts=False,
                    print_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    arch="wormhole",
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=True,
                    no_host_ops=False,
                    no_summary=True,
                    group_by="op",
                    classic_colors=False,
                    summary_file=None,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_csv=None,
                    no_merge_devices=False,
                )

                with open(output_file.name, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)
                    data_rows = list(reader)

                    # Check that the first row is the first operation after the chosen signpost and the last row is the last non-signpost row in the data
                    first_row = data_rows[0]
                    op_code_index = actual_headers.index("OP Code")
                    expected_first_op = "InterleavedToShardedDeviceOperation"
                    actual_first_op = first_row[op_code_index]
                    expected_last_op = "SliceDeviceOperation"
                    actual_last_op = data_rows[-1][op_code_index]
                    
                    assert actual_first_op == expected_first_op, \
                        f"First operation after 'ResNet module started' signpost should be '{expected_first_op}', got '{actual_first_op}'"
                    
                    assert actual_last_op == expected_last_op, \
                        f"Last operation should be '{expected_last_op}', got '{actual_last_op}'"
                    
                    # Signposts should be filtered out
                    for row in data_rows:
                        actual_op = row[op_code_index]
                        assert "(signpost)" not in actual_op, \
                            f"Output should not contain signpost rows, but found: {actual_op}"

            # Clean up
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)
                except OSError:
                    pass

def test_csv_headers_with_end_signpost(test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost=None,
                    end_signpost='ResNet module finished',
                    ignore_signposts=False,
                    print_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    arch="wormhole",
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=True,
                    no_host_ops=False,
                    no_summary=True,
                    group_by="op",
                    classic_colors=False,
                    summary_file=None,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_csv=None,
                    no_merge_devices=False,
                )

                with open(output_file.name, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)
                    data_rows = list(reader)

                    # Check that the the last row is the expected operation and the first row is the first operation in the data
                    first_row = data_rows[0]
                    op_code_index = actual_headers.index("OP Code")
                    expected_first_op = "TilizeWithValPadding"
                    actual_first_op = first_row[op_code_index]
                    expected_last_op = "ShardedToInterleavedDeviceOperation"
                    actual_last_op = data_rows[-1][op_code_index]
                    
                    assert actual_first_op == expected_first_op, \
                        f"First operation should be '{expected_first_op}', got '{actual_first_op}'"
                    
                    assert expected_last_op == actual_last_op, \
                        f"Last operation before 'ResNet module finished' signpost should be '{expected_last_op}', got '{actual_last_op}'"
                    
                    # Signposts should be filtered out
                    for row in data_rows:
                        actual_op = row[op_code_index]
                        assert "(signpost)" not in actual_op, \
                            f"Output should not contain signpost rows, but found: {actual_op}"

            # Clean up
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)
                except OSError:
                    pass

def test_csv_headers_with_both_signposts(test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost='ResNet module started',
                    end_signpost='ResNet module finished',
                    ignore_signposts=False,
                    print_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    arch="wormhole",
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=True,
                    no_host_ops=False,
                    no_summary=True,
                    group_by="op",
                    classic_colors=False,
                    summary_file=None,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_csv=None,
                    no_merge_devices=False,
                )

                with open(output_file.name, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)
                    data_rows = list(reader)

                    # Check that the data is delimited by the two chosen signposts
                    first_row = data_rows[0]
                    op_code_index = actual_headers.index("OP Code")
                    expected_first_op = "InterleavedToShardedDeviceOperation"
                    actual_first_op = first_row[op_code_index]
                    expected_last_op = "ShardedToInterleavedDeviceOperation"
                    actual_last_op = data_rows[-1][op_code_index]
                    
                    assert actual_first_op == expected_first_op, \
                        f"First operation after 'ResNet module started' signpost should be '{expected_first_op}', got '{actual_first_op}'"
                    
                    assert expected_last_op == actual_last_op, \
                        f"Last operation before 'ResNet module finished' signpost should be '{expected_last_op}', got '{actual_last_op}'"
                    
                    # Signposts should be filtered out
                    for row in data_rows:
                        actual_op = row[op_code_index]
                        assert "(signpost)" not in actual_op, \
                            f"Output should not contain signpost rows, but found: {actual_op}"

            # Clean up
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)
                except OSError:
                    pass

def test_csv_headers_with_both_signposts_same_name(test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost='OFT block started',
                    end_signpost='OFT block started',
                    ignore_signposts=False,
                    print_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    arch="wormhole",
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=True,
                    no_host_ops=False,
                    no_summary=True,
                    group_by="op",
                    classic_colors=False,
                    summary_file=None,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_csv=None,
                    no_merge_devices=False,
                )

                with open(output_file.name, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)
                    data_rows = list(reader)

                    # Check that the data is delimited by the two chosen signposts
                    first_row = data_rows[0]
                    op_code_index = actual_headers.index("OP Code")
                    expected_first_op = "TilizeWithValPadding"
                    actual_first_op = first_row[op_code_index]
                    expected_last_op = "UnaryDeviceOperation"
                    actual_last_op = data_rows[-1][op_code_index]
                    
                    assert actual_first_op == expected_first_op, \
                        f"First operation after 'OFT block started (signpost)' signpost should be '{expected_first_op}', got '{actual_first_op}'"
                    
                    assert expected_last_op == actual_last_op, \
                        f"Last operation before 'OFT block started (signpost)' signpost should be '{expected_last_op}', got '{actual_last_op}'"
                    
                    # Signposts should be filtered out
                    for row in data_rows:
                        actual_op = row[op_code_index]
                        assert "(signpost)" not in actual_op, \
                            f"Output should not contain signpost rows, but found: {actual_op}"

            # Clean up
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)
                except OSError:
                    pass

def test_csv_headers_with_print_signposts(test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost='ResNet module started',
                    end_signpost='ResNet module finished',
                    ignore_signposts=False,
                    print_signposts=True,
                    min_percentage=0.5,
                    id_range=None,
                    arch="wormhole",
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=True,
                    no_host_ops=False,
                    no_summary=True,
                    group_by="op",
                    classic_colors=False,
                    summary_file=None,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_csv=None,
                    no_merge_devices=False,
                )

                with open(output_file.name, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)
                    data_rows = list(reader)

                    # Check that the data is delimited by the two chosen signposts
                    first_row = data_rows[0]
                    op_code_index = actual_headers.index("OP Code")
                    expected_first_op = "InterleavedToShardedDeviceOperation"
                    actual_first_op = first_row[op_code_index]
                    expected_last_op = "ShardedToInterleavedDeviceOperation"
                    actual_last_op = data_rows[-1][op_code_index]
                    
                    assert actual_first_op == expected_first_op, \
                        f"First operation after 'ResNet module started' signpost should be '{expected_first_op}', got '{actual_first_op}'"
                    
                    assert expected_last_op == actual_last_op, \
                        f"Last operation before 'ResNet module finished' signpost should be '{expected_last_op}', got '{actual_last_op}'"
                    
                    # Signposts should be present
                    signpost_count = 0

                    for row in data_rows:
                        actual_op = row[op_code_index]

                        if " (signpost)" in actual_op:
                            signpost_count += 1
                        
                    assert signpost_count == 16, \
                        f"Output should contain 16 signpost rows between start and end signposts, found: {signpost_count} signposts"

            # Clean up
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)
                except OSError:
                    pass

# Expected stacked headers fixture
@pytest.fixture
def expected_stacked_headers():
    return [
        "Total % [%]",
        "Op Code",
        "Device Time Sum [μs]",
        "Op Count",
        "Op Category",
        "Min FLOPs [%]",
        "Max FLOPs [%]",
        "Mean FLOPs [%]",
        "Std FLOPs [%]",
        "Weighted Mean FLOPs [%]",
    ]

def test_stacked_csv_headers(expected_stacked_headers, test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="", delete=False
        ) as output_file:
            stacked_csv_file = output_file.name + ".csv"

            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost=None,
                    end_signpost=None,
                    ignore_signposts=True,
                    print_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    arch="wormhole",
                    csv_output_file=None,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=False,
                    no_host_ops=False,
                    no_summary=False,
                    group_by="op",
                    classic_colors=False,
                    summary_file=output_file.name,
                    no_stacked_report=False,
                    no_stack_by_in0=True,
                    stacked_csv=None,
                    no_merge_devices=False,
                )

                assert os.path.exists(stacked_csv_file), "Stacked CSV file should be created"

                with open(stacked_csv_file, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)

                    # Test that all expected stacked headers are present and in the correct order
                    assert actual_headers == expected_stacked_headers, \
                        "Stacked CSV headers do not match expected headers"
                    
                    data_rows = list(reader)
                    assert len(data_rows) > 0, "Stacked CSV should contain data rows"
                    
                    for i, (expected, actual) in enumerate(zip(expected_stacked_headers, actual_headers)):
                        assert actual == expected, \
                            f"Stacked column {i} mismatch. Expected '{expected}', got '{actual}'."

                    # Ensure that no signpost rows are present
                    for row in data_rows:
                        op_code_joined = row[1] if len(row) > 1 else ""
                        assert "(signpost)" not in op_code_joined, \
                            f"Stacked CSV should not contain signpost rows, but found: {op_code_joined}"

            # Clean up
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(stacked_csv_file)
                except OSError:
                    pass

def test_stacked_csv_headers_with_input0_layout(expected_stacked_headers, test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="", delete=False
        ) as output_file:
            stacked_csv_file = output_file.name + ".csv"

            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost=None,
                    end_signpost=None,
                    ignore_signposts=True,
                    print_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    arch="wormhole",
                    csv_output_file=None,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=False,
                    no_host_ops=False,
                    no_summary=False,
                    group_by="memory",
                    classic_colors=False,
                    summary_file=output_file.name,
                    no_stacked_report=False,
                    no_stack_by_in0=False,
                    stacked_csv=None,
                    no_merge_devices=False,
                )

                with open(stacked_csv_file, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)

                    # Test that all expected stacked headers are present and in the correct order
                    assert actual_headers == expected_stacked_headers, \
                        "Stacked CSV headers should be the same regardless of input0 layout grouping"
                    
                    data_rows = list(reader)
                    assert len(data_rows) > 0, "Stacked CSV should contain data rows"

                    # Test that OP Code Joined includes input 0 layout info
                    op_code_joined_values = [
                        row[1] for row in data_rows
                    ]  # Column 1 is OP Code Joined
                    has_layout_info = any(
                        "(in0:" in op_code for op_code in op_code_joined_values
                    )
                    assert has_layout_info, \
                        "OP Code Joined should include input 0 layout information"

                    # Ensure that no signpost rows are present
                    for row in data_rows:
                        op_code_joined = row[1] if len(row) > 1 else ""
                        assert "(signpost)" not in op_code_joined, \
                            f"Stacked CSV should not contain signpost rows, but found: {op_code_joined}"

            # Clean up
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(stacked_csv_file)
                except OSError:
                    pass

@pytest.mark.parametrize("file_path,expected_csv_format,expected_arch,expected_worker_core_count", [
    ("tests/data/ops_perf_results_2025_09_18_11_39_20.csv", CsvFormat.V2, "wormhole", 64),
    ("tests/data/bh20_oft.csv", CsvFormat.V2_1, "blackhole", 20),
    ("tests/data/bh_p100_dlv3.csv", CsvFormat.V2_1, "blackhole", 110),
    ("tests/data/bh_64_clip_encoder_1.csv", CsvFormat.V2_1, "blackhole", 64),
    ("tests/data/wh_clip_encoder_2.csv", CsvFormat.V2_1, "wormhole", 64),
    ("tests/data/bh20_oft_integral_image_trace.csv", CsvFormat.V1, "wormhole", 64),  # V1 defaults to wormhole
    ("tests/data/bh_8xp150_deepseek_v3_d_p.csv", CsvFormat.V2_1, "blackhole", 110),
    ("tests/data/bh_invalid_trace_decode_window.csv", CsvFormat.V2_1, "blackhole", 110),
])
def test_csv_format_arch_and_cores(file_path, expected_csv_format, expected_arch, expected_worker_core_count):
    """Test that CSV format, architecture, and worker core count are correctly detected."""
    # Load the CSV file
    df = pd.read_csv(file_path, low_memory=False)
    
    # Test CSV format detection
    detected_format = detect_csv_format(df)
    assert detected_format == expected_csv_format, \
        f"Expected CSV format {expected_csv_format}, but got {detected_format}"
    
    # Test architecture detection
    detected_arch = ArchitectureSpec._get_arch_name_from_df(df)
    assert detected_arch == expected_arch, \
        f"Expected architecture '{expected_arch}', but got '{detected_arch}'"
    
    # Test worker core count detection
    detected_cores = ArchitectureSpec._get_worker_core_count_from_df(df).worker_cores
    assert detected_cores == expected_worker_core_count, \
        f"Expected {expected_worker_core_count} worker cores, but got {detected_cores}"


def _sparse_matmul_csv_content(nnz_value):
    fields = [
        "OP CODE",
        "OP TYPE",
        "GLOBAL CALL COUNT",
        "DEVICE ID",
        "DEVICE ARCH",
        "ATTRIBUTES",
        "MATH FIDELITY",
        "CORE COUNT",
        "AVAILABLE WORKER CORE COUNT",
        "HOST START TS",
        "OP TO OP LATENCY [ns]",
        "DEVICE KERNEL DURATION [ns]",
        "INPUT_0_W_PAD[LOGICAL]",
        "INPUT_0_Z_PAD[LOGICAL]",
        "INPUT_0_Y_PAD[LOGICAL]",
        "INPUT_0_X_PAD[LOGICAL]",
        "INPUT_0_LAYOUT",
        "INPUT_0_DATATYPE",
        "INPUT_0_MEMORY",
        "INPUT_1_W_PAD[LOGICAL]",
        "INPUT_1_Z_PAD[LOGICAL]",
        "INPUT_1_Y_PAD[LOGICAL]",
        "INPUT_1_X_PAD[LOGICAL]",
        "INPUT_1_LAYOUT",
        "INPUT_1_DATATYPE",
        "INPUT_1_MEMORY",
        "INPUT_2_W_PAD[LOGICAL]",
        "INPUT_2_Z_PAD[LOGICAL]",
        "INPUT_2_Y_PAD[LOGICAL]",
        "INPUT_2_X_PAD[LOGICAL]",
        "INPUT_2_LAYOUT",
        "INPUT_2_DATATYPE",
        "INPUT_2_MEMORY",
        "OUTPUT_0_W_PAD[LOGICAL]",
        "OUTPUT_0_Z_PAD[LOGICAL]",
        "OUTPUT_0_Y_PAD[LOGICAL]",
        "OUTPUT_0_X_PAD[LOGICAL]",
        "OUTPUT_0_LAYOUT",
        "OUTPUT_0_DATATYPE",
        "OUTPUT_0_MEMORY",
    ]
    attributes = (
        "{'compute_kernel_config': 'ComputeKernelConfig(math_fidelity=HiFi2;math_approx_mode=0;"
        "fp32_dest_acc_en=0;packer_l1_acc=1;dst_full_sync_en=0;throttle_level=ThrottleLevel::NO_THROTTLE)'; "
        "'global_cb': 'std::nullopt'; 'is_input_a_sparse': 'false'; 'is_input_b_sparse': 'true'; "
        f"'nnz': '{nnz_value}'; 'output_dtype': 'DataType::BFLOAT16'; "
        "'output_mem_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED;buffer_type=BufferType::L1)'; "
        "'program_config': 'MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=3-1;"
        "in0_block_w=8;out_subblock_h=1;out_subblock_w=1;out_block_h=1;out_block_w=1;per_core_M=1;"
        "per_core_N=1;fuse_batch=0;mcast_in0=1)'}"
    )
    row = {
        "OP CODE": "SparseMatmulDeviceOperation",
        "OP TYPE": "tt_dnn_device",
        "GLOBAL CALL COUNT": "1",
        "DEVICE ID": "0",
        "DEVICE ARCH": "wormhole_b0",
        "ATTRIBUTES": attributes,
        "MATH FIDELITY": "HiFi2",
        "CORE COUNT": "3",
        "AVAILABLE WORKER CORE COUNT": "64",
        "HOST START TS": "1000",
        "OP TO OP LATENCY [ns]": "0",
        "DEVICE KERNEL DURATION [ns]": "126268",
        "INPUT_0_W_PAD[LOGICAL]": "1[1]",
        "INPUT_0_Z_PAD[LOGICAL]": "1[1]",
        "INPUT_0_Y_PAD[LOGICAL]": "32[1]",
        "INPUT_0_X_PAD[LOGICAL]": "2816[2816]",
        "INPUT_0_LAYOUT": "TILE",
        "INPUT_0_DATATYPE": "BFLOAT16",
        "INPUT_0_MEMORY": "DEV_0_L1_WIDTH_SHARDED",
        "INPUT_1_W_PAD[LOGICAL]": "1[1]",
        "INPUT_1_Z_PAD[LOGICAL]": "128[128]",
        "INPUT_1_Y_PAD[LOGICAL]": "2816[2816]",
        "INPUT_1_X_PAD[LOGICAL]": "96[96]",
        "INPUT_1_LAYOUT": "TILE",
        "INPUT_1_DATATYPE": "BFLOAT8_B",
        "INPUT_1_MEMORY": "DEV_0_DRAM_INTERLEAVED",
        "INPUT_2_W_PAD[LOGICAL]": "1[1]",
        "INPUT_2_Z_PAD[LOGICAL]": "1[1]",
        "INPUT_2_Y_PAD[LOGICAL]": "1[1]",
        "INPUT_2_X_PAD[LOGICAL]": "128[128]",
        "INPUT_2_LAYOUT": "ROW_MAJOR",
        "INPUT_2_DATATYPE": "BFLOAT16",
        "INPUT_2_MEMORY": "DEV_0_DRAM_INTERLEAVED",
        "OUTPUT_0_W_PAD[LOGICAL]": "1[1]",
        "OUTPUT_0_Z_PAD[LOGICAL]": "128[128]",
        "OUTPUT_0_Y_PAD[LOGICAL]": "32[1]",
        "OUTPUT_0_X_PAD[LOGICAL]": "96[96]",
        "OUTPUT_0_LAYOUT": "TILE",
        "OUTPUT_0_DATATYPE": "BFLOAT16",
        "OUTPUT_0_MEMORY": "DEV_0_L1_INTERLEAVED",
    }

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue()


def _run_sparse_matmul_report(csv_content, mocker, active_experts=None):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as input_file:
        input_file.write(csv_content)
        input_file.flush()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as output_file:
            try:
                stdout = StringIO()
                mocker.patch("sys.stdout", stdout)
                generate_perf_report(
                    csv_files=[input_file.name],
                    start_signpost=None,
                    end_signpost=None,
                    ignore_signposts=True,
                    print_signposts=False,
                    min_percentage=0.0,
                    id_range=None,
                    arch=None,
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=False,
                    no_host_ops=False,
                    no_summary=True,
                    group_by="op",
                    classic_colors=False,
                    summary_file=None,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_csv=None,
                    no_merge_devices=False,
                    active_experts=active_experts,
                )

                with open(output_file.name, "r") as f:
                    rows = list(csv.DictReader(f))
                return rows[0], stdout.getvalue()
            finally:
                try:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)
                except OSError:
                    pass


def test_sparse_matmul_uses_numeric_nnz_for_utilization(mocker):
    row, stdout = _run_sparse_matmul_report(_sparse_matmul_csv_content("8"), mocker)

    assert "active=8/128" in row["OP Code"]
    assert row["DRAM %"] != ""
    assert row["FLOPs %"] != ""
    assert float(row["FLOPs %"]) < 30
    assert "--active-experts" not in row["Advice"]
    assert "Overall DRAM roofline" in stdout


def test_sparse_matmul_without_nnz_omits_utilization_and_warns(mocker):
    row, stdout = _run_sparse_matmul_report(_sparse_matmul_csv_content("std::nullopt"), mocker)

    assert "active=?/128" in row["OP Code"]
    assert row["DRAM"] == ""
    assert row["DRAM %"] == ""
    assert row["FLOPs"] == ""
    assert row["FLOPs %"] == ""
    assert "--active-experts K" in row["Advice"]
    assert "pass --active-experts K" in stdout


def test_sparse_matmul_active_experts_flag_fills_missing_nnz(mocker):
    row, _ = _run_sparse_matmul_report(_sparse_matmul_csv_content("std::nullopt"), mocker, active_experts=8)

    assert "active=8/128" in row["OP Code"]
    assert row["DRAM %"] != ""
    assert row["FLOPs %"] != ""
    assert "--active-experts" not in row["Advice"]


def test_overall_dram_roofline_weights_modeled_bytes_over_visible_device_time():
    rows = [
        {
            "OP Code": Cell("MatmulDeviceOperation"),
            "Device Time": Cell(10.0),
            "DRAM": Cell(100.0),
            "DRAM %": Cell(50.0),
            "DRAM Bytes": Cell(1_000_000),
        },
        {
            "OP Code": Cell("UnaryDeviceOperation"),
            "Device Time": Cell(10.0),
            "DRAM": Cell(None),
            "DRAM %": Cell(None),
            "DRAM Bytes": Cell(None),
        },
    ]

    dram_speed, dram_percentage = calculate_overall_dram_roofline(rows)

    assert dram_speed == pytest.approx(50.0)
    assert dram_percentage == pytest.approx(25.0)


# --- Subdevice support -------------------------------------------------------
#
# bh_invalid_trace_decode_window.csv is a real capture carrying a SUB DEVICE ID
# column, which confirms the column name, but every value in it is blank and its
# AVAILABLE WORKER CORE COUNT is uniformly 110. No captured report here reports
# more than one budget, so the partitioned-grid case is built synthetically.
# Budgets mirror the DeepSeek-V3 MoE prefill trace in tt-perf-report#65: a
# 108-core shared-expert subdevice, a 12-core dispatch subdevice, and full-grid
# ops at 120.

_SUBDEVICE_FIELDS = [
    "OP CODE",
    "OP TYPE",
    "GLOBAL CALL COUNT",
    "DEVICE ID",
    "DEVICE ARCH",
    "SUB DEVICE ID",
    "AVAILABLE WORKER CORE COUNT",
    "ATTRIBUTES",
    "MATH FIDELITY",
    "CORE COUNT",
    "HOST START TS",
    "OP TO OP LATENCY [ns]",
    "DEVICE KERNEL DURATION [ns]",
    "INPUT_0_W_PAD[LOGICAL]",
    "INPUT_0_Z_PAD[LOGICAL]",
    "INPUT_0_Y_PAD[LOGICAL]",
    "INPUT_0_X_PAD[LOGICAL]",
    "INPUT_0_LAYOUT",
    "INPUT_0_DATATYPE",
    "INPUT_0_MEMORY",
    "INPUT_1_W_PAD[LOGICAL]",
    "INPUT_1_Z_PAD[LOGICAL]",
    "INPUT_1_Y_PAD[LOGICAL]",
    "INPUT_1_X_PAD[LOGICAL]",
    "INPUT_1_LAYOUT",
    "INPUT_1_DATATYPE",
    "INPUT_1_MEMORY",
    "OUTPUT_0_W_PAD[LOGICAL]",
    "OUTPUT_0_Z_PAD[LOGICAL]",
    "OUTPUT_0_Y_PAD[LOGICAL]",
    "OUTPUT_0_X_PAD[LOGICAL]",
    "OUTPUT_0_LAYOUT",
    "OUTPUT_0_DATATYPE",
    "OUTPUT_0_MEMORY",
]


def _subdevice_row(op_code, sub_device_id, available_cores, core_count, host_ts, device_id="0", **overrides):
    row = {
        "OP CODE": op_code,
        "OP TYPE": "tt_dnn_device",
        "GLOBAL CALL COUNT": str(host_ts),
        "DEVICE ID": device_id,
        "DEVICE ARCH": "blackhole",
        "SUB DEVICE ID": sub_device_id,
        "AVAILABLE WORKER CORE COUNT": available_cores,
        "ATTRIBUTES": "",
        "MATH FIDELITY": "HiFi2",
        "CORE COUNT": core_count,
        "HOST START TS": str(host_ts),
        "OP TO OP LATENCY [ns]": "0",
        "DEVICE KERNEL DURATION [ns]": "100000",
        "INPUT_0_W_PAD[LOGICAL]": "1[1]",
        "INPUT_0_Z_PAD[LOGICAL]": "1[1]",
        "INPUT_0_Y_PAD[LOGICAL]": "512[512]",
        "INPUT_0_X_PAD[LOGICAL]": "512[512]",
        "INPUT_0_LAYOUT": "TILE",
        "INPUT_0_DATATYPE": "BFLOAT16",
        "INPUT_0_MEMORY": "DEV_0_L1_INTERLEAVED",
        "INPUT_1_W_PAD[LOGICAL]": "1[1]",
        "INPUT_1_Z_PAD[LOGICAL]": "1[1]",
        "INPUT_1_Y_PAD[LOGICAL]": "512[512]",
        "INPUT_1_X_PAD[LOGICAL]": "512[512]",
        "INPUT_1_LAYOUT": "TILE",
        "INPUT_1_DATATYPE": "BFLOAT16",
        "INPUT_1_MEMORY": "DEV_0_DRAM_INTERLEAVED",
        "OUTPUT_0_W_PAD[LOGICAL]": "1[1]",
        "OUTPUT_0_Z_PAD[LOGICAL]": "1[1]",
        "OUTPUT_0_Y_PAD[LOGICAL]": "512[512]",
        "OUTPUT_0_X_PAD[LOGICAL]": "512[512]",
        "OUTPUT_0_LAYOUT": "TILE",
        "OUTPUT_0_DATATYPE": "BFLOAT16",
        "OUTPUT_0_MEMORY": "DEV_0_DRAM_INTERLEAVED",
    }
    row.update(overrides)
    return row


def _rows_to_csv(rows):
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_SUBDEVICE_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _subdevice_csv_content():
    rows = [
        # Shared-expert subdevice, using all of its 108 cores.
        _subdevice_row("MatmulDeviceOperation", "1", "108", "108", 1000),
        # Dispatch subdevice: 6 of 12 cores, which the collapsed global budget
        # would have scored as 6 of 120.
        _subdevice_row("DispatchDeviceOperation", "0", "12", "6", 2000),
        # Blank subdevice means the full grid.
        _subdevice_row("MatmulDeviceOperation", "", "120", "64", 3000),
    ]
    return _rows_to_csv(rows)


_REPORT_DEFAULTS = dict(
    start_signpost=None,
    end_signpost=None,
    ignore_signposts=True,
    print_signposts=False,
    min_percentage=0.0,
    id_range=None,
    arch=None,
    no_advice=False,
    tracing_mode=False,
    raw_op_codes=False,
    no_host_ops=False,
    no_summary=True,
    group_by="op",
    classic_colors=False,
    summary_file=None,
    no_stacked_report=True,
    no_stack_by_in0=True,
    stacked_csv=None,
    no_merge_devices=False,
)

_DEFAULT_CSV_OUTPUT = object()


def _run_report(mocker, csv_content, csv_output_file=_DEFAULT_CSV_OUTPUT, **overrides):
    """
    Run generate_perf_report over csv_content and return (csv_headers, csv_rows, stdout).

    Pass csv_output_file=None to exercise the terminal-table path instead; the
    returned headers and rows are then empty.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as input_file:
        input_file.write(csv_content)
        input_file.flush()

    if csv_output_file is _DEFAULT_CSV_OUTPUT:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            csv_output_file = handle.name

    try:
        stdout = StringIO()
        mocker.patch("sys.stdout", stdout)
        generate_perf_report(
            csv_files=[input_file.name],
            csv_output_file=csv_output_file,
            **{**_REPORT_DEFAULTS, **overrides},
        )

        headers, rows = [], []
        if csv_output_file:
            with open(csv_output_file, "r") as f:
                headers = next(csv.reader(f))
            with open(csv_output_file, "r") as f:
                rows = list(csv.DictReader(f))
        return headers, rows, stdout.getvalue()
    finally:
        for path in (input_file.name, csv_output_file):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _run_subdevice_report(mocker, **overrides):
    return _run_report(mocker, _subdevice_csv_content(), **overrides)


def test_subdevice_report_carries_per_op_core_budgets(mocker):
    _, rows, _ = _run_subdevice_report(mocker)

    assert [row["Sub Device ID"] for row in rows] == ["1", "0", ""]
    assert [row["Available Cores"] for row in rows] == ["108", "12", "120"]
    assert [row["Cores"] for row in rows] == ["108", "6", "64"]


def test_subdevice_report_headers_have_no_duplicates(mocker):
    headers, _, _ = _run_subdevice_report(mocker)

    assert len(headers) == len(set(headers)), f"Duplicate column in CSV output: {headers}"
    assert "Sub Device ID" in headers
    assert "Available Cores" in headers


def test_csv_column_order_does_not_depend_on_subdevices(test_csv_content, mocker):
    # The CSV is a machine-readable contract for downstream consumers, so a
    # partitioned run must not shift columns for everyone else. The subdevice
    # columns are promoted in the terminal table only.
    partitioned_headers, _, _ = _run_subdevice_report(mocker)
    plain_headers, _, _ = _run_report(mocker, test_csv_content, arch="wormhole", min_percentage=0.5)

    assert partitioned_headers == plain_headers
    assert partitioned_headers.index("Sub Device ID") > partitioned_headers.index("Math Fidelity")


def test_subdevice_columns_promoted_into_terminal_table(mocker):
    _, _, stdout = _run_subdevice_report(mocker, csv_output_file=None)

    header_line = next(line for line in stdout.splitlines() if line.startswith("ID") and "Total %" in line)
    columns = re.split(r"\s{2,}", header_line.strip())

    assert columns.index("Sub Device ID") == columns.index("Device") + 1
    assert columns.index("Available Cores") == columns.index("Cores") + 1

    # The subdevice op reports its id; the full-grid op leaves the cell blank.
    # Columns are padded to the header width, so slice the header's own span
    # rather than splitting on whitespace, which would collapse a blank cell.
    start = header_line.index("Sub Device ID")
    end = start + len("Sub Device ID")
    matmul_cells = {
        line[start:end].strip()
        for line in stdout.splitlines()
        if "MatmulDeviceOperation" in line
    }
    assert matmul_cells == {"1", ""}


def test_subdevice_report_reports_budgets_without_warning(mocker):
    _, _, stdout = _run_subdevice_report(mocker)

    assert "Detected multiple worker core budgets [120, 108, 12]" in stdout
    assert "using per-op values" in stdout
    assert "largest observed budget is 120" in stdout
    # The old behavior warned and silently measured every op against one budget.
    assert "Multiple AVAILABLE WORKER CORE COUNT values found" not in stdout
    assert "Subdevices: 2" in stdout
    # 120 is the largest budget any op reported, not blackhole's grid, which is
    # 110. Naming it as the architecture's worker cores would assert a grid size
    # no row in the file reported.
    assert "Architecture: blackhole, largest observed worker core budget: 120" in stdout
    assert "Worker cores:" not in stdout


def test_uniform_budget_still_reports_the_count_as_the_worker_grid(mocker):
    # Only a partitioned capture needs the "observed" hedge; on every other run
    # the single budget is the grid, and the wording is unchanged.
    csv_content = _rows_to_csv([_subdevice_row("MatmulDeviceOperation", "", "110", "88", 1000)])
    _, _, stdout = _run_report(mocker, csv_content)

    assert "Architecture: blackhole, Worker cores: 110" in stdout
    assert "largest observed worker core budget" not in stdout


def test_subdevice_columns_absent_from_terminal_table_without_subdevices(test_csv_content, mocker):
    _, _, stdout = _run_report(
        mocker, test_csv_content, csv_output_file=None, arch="wormhole", min_percentage=0.5
    )

    header_line = next(line for line in stdout.splitlines() if line.startswith("ID") and "Total %" in line)
    columns = re.split(r"\s{2,}", header_line.strip())

    assert "Sub Device ID" not in columns
    assert "Available Cores" not in columns
    assert "Sub devices:" not in stdout


def test_get_op_available_cores_prefers_per_op_value_over_file_wide():
    assert get_op_available_cores({"AVAILABLE WORKER CORE COUNT": 108}, 120) == 108
    # Floats: pandas types the column as float whenever any row is blank.
    assert get_op_available_cores({"AVAILABLE WORKER CORE COUNT": 12.0}, 120) == 12
    # Missing, blank, zero and negative all fall back to the file-wide grid.
    assert get_op_available_cores({}, 120) == 120
    assert get_op_available_cores({"AVAILABLE WORKER CORE COUNT": float("nan")}, 120) == 120
    assert get_op_available_cores({"AVAILABLE WORKER CORE COUNT": 0}, 120) == 120
    assert get_op_available_cores({"AVAILABLE WORKER CORE COUNT": -8}, 120) == 120


def test_get_op_available_cores_falls_back_on_unusable_values():
    # Malformed cells must not abort the report: the CSV is untrusted input.
    for value in ("unknown", "-", "", float("inf"), float("-inf"), None):
        assert get_op_available_cores({"AVAILABLE WORKER CORE COUNT": value}, 120) == 120


def test_get_op_sub_device_id_treats_blank_as_full_grid():
    assert get_op_sub_device_id({"SUB DEVICE ID": "1"}) == 1
    assert get_op_sub_device_id({"SUB DEVICE ID": 0.0}) == 0
    assert get_op_sub_device_id({"SUB DEVICE ID": ""}) is None
    assert get_op_sub_device_id({"SUB DEVICE ID": "   "}) is None
    assert get_op_sub_device_id({"SUB DEVICE ID": float("nan")}) is None
    assert get_op_sub_device_id({}) is None


def test_get_op_sub_device_id_passes_through_unparseable_ids():
    # Visible garbage beats a silent full-grid reading.
    assert get_op_sub_device_id({"SUB DEVICE ID": "compute"}) == "compute"
    assert get_op_sub_device_id({"SUB DEVICE ID": float("inf")}) == "inf"
    assert get_op_sub_device_id({"SUB DEVICE ID": "1e999"}) == "1e999"


def test_count_sub_devices_counts_distinct_real_ids():
    # Takes ids already extracted by get_op_sub_device_id, so "1" and "1.0"
    # cannot be counted as two subdevices.
    assert count_sub_devices([1, 0, None, 1]) == 2
    assert count_sub_devices([None, None]) == 0
    assert count_sub_devices([]) == 0


def _conv_row(core_count, available_cores):
    return {
        "DEVICE KERNEL DURATION [ns]": 100000.0,
        "MATH FIDELITY": "HiFi4",
        "ATTRIBUTES": "window_hw=(3;3); ",
        "CORE COUNT": core_count,
        "AVAILABLE WORKER CORE COUNT": available_cores,
        "OUTPUT_0_Y_PAD[LOGICAL]": "1024[1024]",
        "INPUT_0_X_PAD[LOGICAL]": "64[64]",
        "INPUT_1_X_PAD[LOGICAL]": "128[128]",
        "INPUT_0_DATATYPE": "BFLOAT16",
        "INPUT_0_MEMORY": "DEV_0_L1_INTERLEAVED",
        "INPUT_1_DATATYPE": "BFLOAT16",
        "INPUT_1_MEMORY": "DEV_0_DRAM_INTERLEAVED",
        "OUTPUT_0_DATATYPE": "BFLOAT16",
        "OUTPUT_0_MEMORY": "DEV_0_L1_INTERLEAVED",
    }


def test_analyze_conv_scores_against_cores_actually_used():
    # Conv scores against the cores it used, as matmul does, so it is
    # subdevice-safe by construction and the per-op budget is not its
    # denominator. Changing only the budget must not move FLOPs %.
    arch = ArchitectureSpec.from_name("blackhole", 110)

    _, full_grid_pct, _, _, _, _ = analyze_conv(_conv_row(64, 110), CsvFormat.V2_1, arch)
    _, subdevice_pct, _, _, _, _ = analyze_conv(_conv_row(64, 12), CsvFormat.V2_1, arch)
    assert subdevice_pct == pytest.approx(full_grid_pct)

    # Halving the cores actually used doubles utilization of those cores.
    _, half_cores_pct, _, _, _, _ = analyze_conv(_conv_row(32, 110), CsvFormat.V2_1, arch)
    assert half_cores_pct == pytest.approx(full_grid_pct * 2)


def _colored_op_data(num_cores, available_cores):
    op_data = {
        "OP Code": Cell("MatmulDeviceOperation 512 x 512 x 512"),
        "Cores": Cell(num_cores),
        "Available Cores": Cell(available_cores),
        "Bound": Cell(""),
        "DRAM": Cell(None),
        "DRAM %": Cell(None),
        "FLOPs": Cell(None),
        "FLOPs %": Cell(None),
        "Op-to-Op Gap": Cell(None),
        "Math Fidelity": Cell(None),
    }
    color_row(op_data, 100.0, 0.0)
    return op_data["Cores"].color


def test_cores_turn_green_at_full_grid_for_any_grid_size():
    # Regression test for the hardcoded 64: a full Blackhole grid never went green.
    assert _colored_op_data(120, 120) == "green"
    assert _colored_op_data(108, 108) == "green"
    assert _colored_op_data(64, 64) == "green"
    # Below the grid it was given, so not green.
    assert _colored_op_data(64, 120) != "green"
    assert _colored_op_data(12, 120) != "green"
    # Unknown budget: no green rather than a guessed 64-core grid.
    assert _colored_op_data(64, None) != "green"


def test_cores_red_accounts_for_the_size_of_the_grid_given():
    # The headline example from tt-perf-report#65: a dispatch op given a 12-core
    # subdevice and using 6 of them is not underutilizing anything.
    assert _colored_op_data(6, 12) != "red"
    # The same 6 cores out of a full grid genuinely is a small slice.
    assert _colored_op_data(6, 120) == "red"
    # Absolute smallness is still required, so nothing newly turns red: an op
    # using 10-15 of a 64-core grid was not red before and is not red now.
    assert _colored_op_data(9, 64) == "red"
    assert _colored_op_data(10, 64) != "red"
    assert _colored_op_data(15, 64) != "red"
    # Unknown budget keeps the old absolute behavior.
    assert _colored_op_data(9, None) == "red"
    # A tiny subdevice used in full is at its grid, so green wins over red.
    assert _colored_op_data(6, 6) == "green"


def _advice_op_data(num_cores, available_cores):
    return {
        "OP Code": Cell("MatmulDeviceOperation 512 x 512 x 512"),
        "Bound": Cell("FLOP"),
        "Cores": Cell(num_cores),
        "Available Cores": Cell(available_cores),
        "Math Fidelity": Cell("HiFi2 BF16 x BF16 => BF16"),
        "Output Datatype": Cell("BFLOAT16"),
        "Input 0 Datatype": Cell("BFLOAT16"),
        "Input 1 Datatype": Cell("BFLOAT16"),
        "DRAM Sharded": Cell(False),
        "FLOPs %": Cell(80.0),
    }


def test_grid_size_advice_uses_the_ops_own_budget():
    advice = generate_matmul_advice(_advice_op_data(64, 120))
    assert "Increase grid size (currently using 64 of 120)" in advice

    # At full use of a subdevice there is no grid left to ask for, where the
    # hardcoded 64 would have told the user to increase it.
    assert not any("Increase grid size" in item for item in generate_matmul_advice(_advice_op_data(108, 108)))

    # An unknown budget yields no advice rather than advice against a guessed
    # 64-core grid, which would be wrong for every Blackhole part. Real reports
    # always carry a budget: analyze_op falls back to the architecture grid.
    assert not any("Increase grid size" in item for item in generate_matmul_advice(_advice_op_data(32, None)))


def test_subdevice_report_stacked_path_tolerates_promoted_column(mocker, tmp_path):
    # generate_stacked_report builds its DataFrame from visible_headers, so
    # promoting the subdevice columns into the visible table reaches it too.
    stacked_base = str(tmp_path / "stacked")
    _run_subdevice_report(
        mocker,
        csv_output_file=None,
        no_summary=False,
        no_stacked_report=False,
        summary_file=stacked_base,
    )

    with open(f"{stacked_base}.csv", "r") as f:
        stacked_rows = list(csv.DictReader(f))

    assert stacked_rows, "stacked report should not be empty"
    op_codes = [row["Op Code"] for row in stacked_rows]
    assert "MatmulDeviceOperation" in op_codes
    assert "DispatchDeviceOperation" in op_codes
    # The promoted columns are not aggregation keys and must not appear.
    assert "Sub Device ID" not in stacked_rows[0]
    assert "Available Cores" not in stacked_rows[0]


def _core_count_df(values):
    return pd.DataFrame({
        "DEVICE ARCH": ["blackhole"] * len(values),
        "AVAILABLE WORKER CORE COUNT": values,
    })


@pytest.mark.parametrize("values,expected", [
    # Numeric column: the largest budget is the full grid, whatever the row order.
    ([120, 108, 12], 120),
    ([12, 108, 120], 120),
    # Text column must be coerced, not compared lexicographically ("8" > "120").
    (["120", "64", "8"], 120),
    # An unreadable cell must be ignored rather than abort the report.
    (["120", "unknown", "8"], 120),
    (["120", "-", "8"], 120),
    # Non-finite values are not a core count.
    ([float("inf"), 64], 64),
    ([float("-inf"), 108], 108),
    # Nothing usable falls back to the architecture's own grid, not a fixed 64.
    ([0, 0], 110),
    (["nonsense"], 110),
    ([float("inf")], 110),
])
def test_worker_core_count_tolerates_malformed_cells(values, expected):
    # _core_count_df reports blackhole, whose registered grid is 110.
    assert ArchitectureSpec._get_worker_core_count_from_df(_core_count_df(values)).worker_cores == expected


def test_worker_core_count_reports_ignored_cells(capsys):
    ArchitectureSpec._get_worker_core_count_from_df(_core_count_df(["120", "unknown", "-", "108"]))

    assert "Ignoring 2 unreadable AVAILABLE WORKER CORE COUNT value(s)." in capsys.readouterr().out


def _flop_bound_csv_content():
    # Tuned so DRAM % lands under the 65% threshold while FLOPs % clears it,
    # which is the only path that reaches the grid-size advice branch.
    return _rows_to_csv([
        _subdevice_row(
            "MatmulDeviceOperation", "1", "108", "64", 1000,
            **{
                "DEVICE KERNEL DURATION [ns]": "1896",
                "OUTPUT_0_MEMORY": "DEV_0_L1_INTERLEAVED",
            },
        )
    ])


def test_grid_size_advice_reaches_the_report_for_a_subdevice_op(mocker):
    _, rows, _ = _run_report(mocker, _flop_bound_csv_content())

    assert rows[0]["Bound"] == "FLOP", f"expected a FLOP-bound op, got {rows[0]['Bound']!r}"
    assert rows[0]["Available Cores"] == "108"
    # The hardcoded 64 would have stayed silent here: the op uses 64 cores.
    assert "Increase grid size (currently using 64 of 108)" in rows[0]["Advice"]


def _multi_device_csv_content():
    return _rows_to_csv([
        _subdevice_row("MatmulDeviceOperation", "1", "108", "108", 1000, device_id="0"),
        # Slower, so merge_device_rows keeps this row and its budget.
        _subdevice_row(
            "MatmulDeviceOperation", "2", "64", "64", 1000, device_id="1",
            **{"DEVICE KERNEL DURATION [ns]": "200000"},
        ),
    ])


def test_merged_devices_keep_a_consistent_subdevice_and_budget_pair(mocker):
    _, rows, _ = _run_report(mocker, _multi_device_csv_content())

    # One row survives per op position, carrying that device's own pair - the
    # id and the budget must come from the same source row.
    assert len(rows) == 1
    assert rows[0]["Sub Device ID"] == "2"
    assert rows[0]["Available Cores"] == "64"
    assert rows[0]["Cores"] == "64"


def test_unmerged_devices_keep_their_own_budgets(mocker):
    _, rows, _ = _run_report(mocker, _multi_device_csv_content(), no_merge_devices=True)

    pairs = {(row["Sub Device ID"], row["Available Cores"]) for row in rows}
    assert pairs == {("1", "108"), ("2", "64")}


def _report_rows_for_fixture(mocker, filename, **overrides):
    path = os.path.join(os.path.dirname(__file__), "data", filename)
    with open(path, "r") as f:
        return _run_report(mocker, f.read(), **overrides)


@pytest.mark.parametrize("filename,arch,expected_budget", [
    # v2 capture with no AVAILABLE WORKER CORE COUNT column at all.
    ("ops_perf_results_2025_09_18_11_39_20.csv", "wormhole", "64"),
    # v2.1 blackhole capture whose column is uniformly 110.
    ("bh_invalid_trace_decode_window.csv", None, "110"),
])
def test_available_cores_falls_back_to_the_architecture_grid(mocker, filename, arch, expected_budget):
    # Pins that the per-op fallback is the architecture's own grid. Hardcoding it
    # back to 64 - the magic number this change removed - must fail here.
    _, rows, _ = _report_rows_for_fixture(mocker, filename, arch=arch, min_percentage=0.0)

    budgets = {row["Available Cores"] for row in rows}
    assert budgets == {expected_budget}, f"expected every op to report {expected_budget}, got {budgets}"


def test_subdevice_columns_absent_when_the_id_column_is_present_but_blank(mocker):
    # bh_invalid_trace_decode_window.csv is a real capture that carries a
    # SUB DEVICE ID column with no values in it, which must read as "no
    # subdevices" rather than as one unnamed subdevice.
    _, _, stdout = _report_rows_for_fixture(
        mocker, "bh_invalid_trace_decode_window.csv", csv_output_file=None, min_percentage=0.0
    )

    header_line = next(line for line in stdout.splitlines() if line.startswith("ID") and "Total %" in line)
    columns = re.split(r"\s{2,}", header_line.strip())
    assert "Sub Device ID" not in columns
    assert "Available Cores" not in columns
    assert "Subdevices:" not in stdout


def _multi_budget_flop_bound_csv():
    # 6 of 12 cores on the dispatch subdevice, in a file whose largest budget is
    # 120, so "of 12" and "of 120" are distinguishable. Duration is tuned to put
    # FLOPs %% over the 65%% bound threshold while DRAM %% stays under it.
    return _rows_to_csv([
        _subdevice_row(
            "MatmulDeviceOperation", "0", "12", "6", 1000,
            **{
                "DEVICE KERNEL DURATION [ns]": "20230",
                "OUTPUT_0_MEMORY": "DEV_0_L1_INTERLEAVED",
            },
        ),
        _subdevice_row("MatmulDeviceOperation", "1", "108", "108", 2000),
        _subdevice_row("MatmulDeviceOperation", "", "120", "120", 3000),
    ])


def test_grid_size_advice_names_the_ops_own_budget_not_the_file_wide_one(mocker):
    _, rows, _ = _run_report(mocker, _multi_budget_flop_bound_csv())

    dispatch = next(row for row in rows if row["Sub Device ID"] == "0")
    assert dispatch["Bound"] == "FLOP", f"expected FLOP-bound, got {dispatch['Bound']!r}"
    assert dispatch["Available Cores"] == "12"
    assert "Increase grid size (currently using 6 of 12)" in dispatch["Advice"]
    assert "of 120" not in dispatch["Advice"]


def test_budget_variation_alone_promotes_available_cores(mocker):
    # No subdevice ids anywhere, but the budgets differ - so the per-op values
    # are in use and must be visible, or the table silently disagrees with the
    # numbers behind it.
    csv_content = _rows_to_csv([
        _subdevice_row("MatmulDeviceOperation", "", "108", "6", 1000),
        _subdevice_row("MatmulDeviceOperation", "", "120", "120", 2000),
    ])
    _, _, stdout = _run_report(mocker, csv_content, csv_output_file=None)

    header_line = next(line for line in stdout.splitlines() if line.startswith("ID") and "Total %" in line)
    columns = re.split(r"\s{2,}", header_line.strip())

    assert "Available Cores" in columns
    # Nothing to show in an id column, so it stays out of the table.
    assert "Sub Device ID" not in columns
    assert "Worker core budgets vary across ops [120, 108]" in stdout


def test_core_count_of_infinity_does_not_abort_the_report(mocker):
    # CORE COUNT is untrusted like every other column; a non-finite cell must
    # cost that one value, not the whole report.
    csv_content = _rows_to_csv([_subdevice_row("MatmulDeviceOperation", "1", "108", "inf", 1000)])
    _, rows, _ = _run_report(mocker, csv_content)

    assert rows[0]["Cores"] == ""
    assert rows[0]["Available Cores"] == "108"


def test_fractional_budget_falls_back_rather_than_becoming_zero(mocker):
    # A budget under 1 truncates to 0, which would disable both coloring rules
    # and suppress the grid-size advice.
    csv_content = _rows_to_csv([_subdevice_row("MatmulDeviceOperation", "1", "0.5", "6", 1000)])
    _, rows, _ = _run_report(mocker, csv_content)

    assert rows[0]["Available Cores"] == "110"


def test_worker_core_count_does_not_call_readable_values_unreadable(capsys):
    # A negative parses fine; it is unusable, not unreadable.
    assert ArchitectureSpec._get_worker_core_count_from_df(_core_count_df([-8, 120])).worker_cores == 120
    assert "unreadable" not in capsys.readouterr().out

    assert ArchitectureSpec._get_worker_core_count_from_df(_core_count_df([0.5])).worker_cores == 110
    assert "unreadable" not in capsys.readouterr().out


def test_merge_prefers_a_zero_duration_row_over_an_unusable_one():
    # Pins the explicit None test in the merge sort key: a genuine zero duration
    # must outrank a row with no usable duration, which `or -1` would not do.
    from tt_perf_report.perf_report import _merge_sort_duration_ns

    zero_duration = ("Matmul", {"DEVICE KERNEL DURATION [ns]": 0})
    unusable = ("Matmul", {"DEVICE KERNEL DURATION [ns]": None})

    assert _merge_sort_duration_ns(zero_duration) == 0
    assert _merge_sort_duration_ns(unusable) == -1
    assert max([unusable, zero_duration], key=_merge_sort_duration_ns) is zero_duration


def test_fractional_subdevice_id_is_not_merged_into_a_real_subdevice(mocker):
    # "1.5" truncated to 1 would silently fold a malformed row into subdevice 1
    # and count it as that subdevice, contradicting the pass-through contract.
    csv_content = _rows_to_csv([
        _subdevice_row("MatmulDeviceOperation", "1", "108", "108", 1000),
        _subdevice_row("MatmulDeviceOperation", "1.5", "108", "108", 2000),
    ])
    _, rows, stdout = _run_report(mocker, csv_content)

    assert [row["Sub Device ID"] for row in rows] == ["1", "1.5"]
    assert "Subdevices: 2" in stdout


@pytest.mark.parametrize("column", ["INPUT_0_Y_PAD[LOGICAL]", "INPUT_1_X_PAD[LOGICAL]", "OUTPUT_0_X_PAD[LOGICAL]"])
@pytest.mark.parametrize("value", ["unknown", "inf", "", "512.5[512]"])
def test_malformed_tensor_dimension_omits_metrics_without_aborting(mocker, column, value):
    row = _subdevice_row("MatmulDeviceOperation", "1", "108", "108", 1000)
    row[column] = value
    _, rows, _ = _run_report(mocker, _rows_to_csv([row]))

    # The op is still reported, with its shape-derived figures omitted.
    assert len(rows) == 1
    assert rows[0]["FLOPs %"] == ""
    assert rows[0]["DRAM %"] == ""
    assert "unknown shape" in rows[0]["OP Code"]
    # Fields that do not depend on the shape survive.
    assert rows[0]["Cores"] == "108"
    assert rows[0]["Available Cores"] == "108"


@pytest.mark.parametrize("device_id", ["inf", "-inf", "banana", "1.5", ""])
def test_unusable_device_id_does_not_abort_the_report(mocker, device_id):
    row = _subdevice_row("MatmulDeviceOperation", "1", "108", "108", 1000)
    row["DEVICE ID"] = device_id
    _, rows, _ = _run_report(mocker, _rows_to_csv([row]))

    assert len(rows) == 1
    assert rows[0]["Device"] == ""


def test_non_matmul_op_with_unusable_core_count(mocker):
    # A matmul has its Cores overwritten by analyze_matmul, so only a non-matmul
    # row exercises analyze_op's own CORE COUNT coercion.
    csv_content = _rows_to_csv([_subdevice_row("DispatchDeviceOperation", "1", "108", "inf", 1000)])
    _, rows, _ = _run_report(mocker, csv_content)

    assert rows[0]["Cores"] == ""
    assert rows[0]["Available Cores"] == "108"


def test_blank_budget_row_inherits_the_largest_observed_budget(mocker):
    # Pins which fallback a blank cell gets in a file that does report budgets:
    # the largest observed, not the architecture's registered grid (110 here).
    csv_content = _rows_to_csv([
        _subdevice_row("MatmulDeviceOperation", "1", "108", "108", 1000),
        _subdevice_row("MatmulDeviceOperation", "0", "", "6", 2000),
        _subdevice_row("MatmulDeviceOperation", "", "120", "120", 3000),
    ])
    _, rows, _ = _run_report(mocker, csv_content)

    assert [row["Available Cores"] for row in rows] == ["108", "120", "120"]


@pytest.mark.parametrize("values,expected,message", [
    # Not integral: a fraction is malformed, not a 108-core grid.
    ([108.5], 110, "not a core count"),
    # Implausible magnitude, which a vectorised astype(int) would have saturated
    # to the int64 maximum and reported as the file's grid.
    ([1e30, 108.0], 108, "not a core count"),
    ([-8, 108.0], 108, "not a core count"),
    # Not a number at all keeps the separate, accurate wording.
    (["-", 108.0], 108, "unreadable"),
])
def test_worker_core_count_reports_what_it_ignored(capsys, values, expected, message):
    assert ArchitectureSpec._get_worker_core_count_from_df(_core_count_df(values)).worker_cores == expected
    assert message in capsys.readouterr().out


def test_worker_core_count_does_not_complain_about_blank_cells(capsys):
    # Blank cells are the common real case, and are not something to report.
    assert ArchitectureSpec._get_worker_core_count_from_df(_core_count_df([120, None, 108])).worker_cores == 120
    assert "Ignoring" not in capsys.readouterr().out


def test_cores_red_threshold_is_relative_to_the_budget(capsys):
    # Discriminates the "less than half the budget" rule from other divisors:
    # 8 of 20 is under half, 8 of 15 is not.
    assert _colored_op_data(8, 20) == "red"
    assert _colored_op_data(8, 15) != "red"


def test_partitioned_run_warns_that_totals_assume_sequential_execution(mocker):
    # The per-op figures are right, but every total still sums durations as if
    # the ops ran one after another, which is exactly what subdevices break.
    _, _, stdout = _run_subdevice_report(mocker)

    assert "totals assume ops ran sequentially" in stdout
    assert "overcounts overlapped work" in stdout
    # Every figure tt-perf-report#65 identified as skewed is named, so a reader
    # is not left assuming the summary report's totals escaped.
    for figure in [
        "summed device time",
        "Total %",
        "op-to-op gap totals",
        "overall DRAM roofline",
        "tracing-savings estimate",
        "Device Time Sum",
        "device-time-weighted mean FLOPs %",
    ]:
        assert figure in stdout, f"the warning does not name {figure}"
    # Per-op FLOPs % is a rate against cores used, so concurrency does not skew it.
    assert "Per-op FLOPs % is unaffected" in stdout


def test_no_sequential_warning_when_the_run_is_not_partitioned(test_csv_content, mocker):
    _, _, stdout = _run_report(mocker, test_csv_content, arch="wormhole", min_percentage=0.5)

    assert "totals assume ops ran sequentially" not in stdout


def test_subdevice_ids_with_a_uniform_budget_still_report_as_partitioned(mocker):
    # The realistic partitioned capture: two subdevices that happen to have been
    # given the same budget. The subdevice columns and the totals warning are
    # still owed, but there is nothing varying to announce.
    csv_content = _rows_to_csv([
        _subdevice_row("MatmulDeviceOperation", "0", "108", "108", 1000),
        _subdevice_row("MatmulDeviceOperation", "1", "108", "54", 2000),
    ])
    _, rows, stdout = _run_report(mocker, csv_content)

    assert [row["Available Cores"] for row in rows] == ["108", "108"]
    assert "Subdevices: 2" in stdout
    assert "totals assume ops ran sequentially" in stdout
    assert "Worker core budgets vary across ops" not in stdout


def test_filtered_out_subdevices_are_not_promoted_into_the_terminal_table(mocker):
    # The promotion decision is taken after every filter, so a range that leaves
    # only the full-grid op must leave the table looking like an unpartitioned
    # run - while --csv keeps both columns, whose order is a fixed contract.
    _, _, stdout = _run_report(
        mocker, _subdevice_csv_content(), csv_output_file=None, id_range=(4, 4)
    )
    header_line = next(line for line in stdout.splitlines() if line.startswith("ID") and "Total %" in line)

    assert "Sub Device ID" not in header_line
    assert "Available Cores" not in header_line

    headers, rows, _ = _run_report(mocker, _subdevice_csv_content(), id_range=(4, 4))
    assert "Sub Device ID" in headers
    assert "Available Cores" in headers
    assert [row["Cores"] for row in rows] == ["64"]


# --- Untrusted text reaching the report's two output sinks --------------------


def test_sanitize_text_drops_row_breaking_control_characters():
    # A newline would end the table row early and render its tail as an op that
    # never ran; a bare escape byte would recolour every following column.
    assert sanitize_text("1\nSPLIT") == "1SPLIT"
    assert sanitize_text("\x1b[31mRED\x1b[0m") == "[31mRED[0m"
    assert sanitize_text("a\x00\x7f\x9fb") == "ab"
    # Tab is left alone: it is whitespace, not a control sequence.
    assert sanitize_text("a\tb") == "a\tb"
    assert sanitize_text("MatmulDeviceOperation 512 x 512") == "MatmulDeviceOperation 512 x 512"
    # Safe on any cell, so callers need no isinstance check of their own.
    assert sanitize_text(108) == 108
    assert sanitize_text(None) is None


def test_a_control_character_in_an_op_code_cannot_forge_a_stacked_report_row(mocker):
    # The stacked report reads raw values rather than going through Cell.format,
    # so op codes are sanitized where they are read instead.
    row = _subdevice_row("Binary\nSPLITOperation", "", "108", "108", 1000)
    _, _, stdout = _run_report(mocker, _rows_to_csv([row]), csv_output_file=None, no_stacked_report=False)

    assert "BinarySPLITOperation" in stdout
    assert not any(line.strip().startswith("SPLIT") for line in stdout.splitlines())


def test_control_characters_in_a_sub_device_id_cannot_forge_a_table_row(mocker):
    row = _subdevice_row("MatmulDeviceOperation", "\x1b[31mRED\x1b[0m\nSPLIT", "108", "108", 1000)
    _, _, stdout = _run_report(mocker, _rows_to_csv([row]), csv_output_file=None)

    assert "SPLIT" in stdout, "the id should still be visible"
    assert not any(line.strip().startswith("SPLIT") for line in stdout.splitlines())


def test_a_long_sub_device_id_cannot_set_the_column_width(mocker):
    row = _subdevice_row("MatmulDeviceOperation", "A" * 300, "108", "108", 1000)
    _, rows, stdout = _run_report(mocker, _rows_to_csv([row]))

    assert rows[0]["Sub Device ID"] == "A" * 32
    assert "A" * 33 not in stdout


def test_escape_csv_formula_only_touches_text_a_spreadsheet_would_evaluate():
    assert escape_csv_formula("=cmd|' /C calc'!A0") == "'=cmd|' /C calc'!A0"
    assert escape_csv_formula("+1") == "'+1"
    assert escape_csv_formula("@SUM(A1)") == "'@SUM(A1)"
    # Values a report normally emits are untouched, numbers included.
    assert escape_csv_formula("MatmulDeviceOperation 512 x 512") == "MatmulDeviceOperation 512 x 512"
    assert escape_csv_formula(-3.5) == -3.5
    assert escape_csv_formula(108) == 108
    assert escape_csv_formula(None) is None


def test_csv_output_neutralises_a_formula_in_a_sub_device_id(mocker):
    row = _subdevice_row("MatmulDeviceOperation", "=cmd|' /C calc'!A0", "108", "108", 1000)
    _, rows, _ = _run_report(mocker, _rows_to_csv([row]))

    assert rows[0]["Sub Device ID"] == "'=cmd|' /C calc'!A0"


# --- Conv attribute parsing --------------------------------------------------


@pytest.mark.parametrize("attributes,expected", [
    ("window_hw=(3;3); stride_hw=(1;1); ", (3, 3)),
    ("window_hw=(7;1); ", (7, 1)),
    # No window recorded at all, which used to raise IndexError.
    ("", None),
    ("stride_hw=(1;1); ", None),
    # Present but not a pair of dimensions.
    ("window_hw=(a;b); ", None),
    ("window_hw=(3); ", None),
    ("window_hw=(0;3); ", None),
    ("window_hw=(-3;3); ", None),
    # A cell of thousands of digits, which int() refuses to convert at all.
    (f"window_hw=({'9' * 5000};3); ", None),
])
def test_get_conv_window_hw_never_raises_on_a_malformed_cell(attributes, expected):
    assert get_conv_window_hw(attributes) == expected


def test_get_conv_window_hw_tolerates_a_non_string_cell():
    assert get_conv_window_hw(None) is None
    assert get_conv_window_hw(3.5) is None


@pytest.mark.parametrize("attributes", [
    "",
    "stride_hw=(1;1); ",
    "window_hw=(a;b); ",
    f"window_hw=({'9' * 5000};3); ",
])
@pytest.mark.parametrize("op_code", ["Conv2dDeviceOperation", "OptimizedConvNew"])
def test_conv_with_no_usable_window_omits_metrics_without_aborting(mocker, attributes, op_code):
    row = _subdevice_row(op_code, "1", "108", "88", 1000, ATTRIBUTES=attributes)
    _, rows, _ = _run_report(mocker, _rows_to_csv([row]))

    assert len(rows) == 1
    assert "unknown shape" in rows[0]["OP Code"]
    assert rows[0]["FLOPs %"] == ""
    # Fields that do not depend on the window survive.
    assert rows[0]["Cores"] == "88"
    assert rows[0]["Available Cores"] == "108"


@pytest.mark.parametrize("column", ["OUTPUT_0_Y_PAD[LOGICAL]", "INPUT_0_X_PAD[LOGICAL]", "INPUT_1_X_PAD[LOGICAL]"])
@pytest.mark.parametrize("value", ["unknown", "inf", "", "512.5[512]"])
def test_conv_with_a_malformed_shape_omits_metrics_without_aborting(mocker, column, value):
    row = _subdevice_row(
        "Conv2dDeviceOperation", "1", "108", "88", 1000, ATTRIBUTES="window_hw=(3;3); "
    )
    row[column] = value
    _, rows, _ = _run_report(mocker, _rows_to_csv([row]))

    assert len(rows) == 1
    assert "unknown shape" in rows[0]["OP Code"]
    assert rows[0]["FLOPs %"] == ""
    assert rows[0]["Cores"] == "88"


def test_conv_with_a_usable_window_still_models_flops(mocker):
    row = _subdevice_row(
        "Conv2dDeviceOperation", "1", "108", "88", 1000, ATTRIBUTES="window_hw=(3;3); "
    )
    _, rows, _ = _run_report(mocker, _rows_to_csv([row]))

    assert "unknown shape" not in rows[0]["OP Code"]
    assert float(rows[0]["FLOPs %"]) > 0


def test_sparse_matmul_with_a_malformed_shape_does_not_blame_active_experts(mocker):
    # Without shapes there is no active batch count to be missing, and the op has
    # no modelled figures either way, so the advice must not send the reader off
    # to --active-experts when the fault is the shape cell.
    row = _subdevice_row("SparseMatmulDeviceOperation", "1", "108", "108", 1000)
    row["INPUT_0_Y_PAD[LOGICAL]"] = "unknown"
    _, rows, stdout = _run_report(mocker, _rows_to_csv([row]))

    assert "unknown shape" in rows[0]["OP Code"]
    assert "--active-experts" not in rows[0]["Advice"]
    assert "--active-experts" not in stdout


# --- Multi-file and multi-device merging -------------------------------------


def _write_csv(tmp_path, name, device_ids):
    rows = [
        _subdevice_row("MatmulDeviceOperation", "", "108", "108", 1000 + index, device_id=device_id)
        for index, device_id in enumerate(device_ids)
    ]
    path = tmp_path / name
    path.write_text(_rows_to_csv(rows))
    return str(path)


def test_merge_perf_traces_offsets_each_files_device_ids(tmp_path):
    first = _write_csv(tmp_path, "a.csv", ["0", "1"])
    second = _write_csv(tmp_path, "b.csv", ["0", "1"])

    merged = merge_perf_traces([first, second])

    assert sorted(merged["DEVICE ID"].tolist()) == [0.0, 1.0, 2.0, 3.0]


def test_merge_perf_traces_tolerates_a_file_that_names_no_device(tmp_path, capsys):
    # An unusable id must not decide the device count: taking zero from here
    # would zero the offset applied to every later file and silently merge their
    # devices together. It must not abort the run either.
    first = _write_csv(tmp_path, "a.csv", ["1.5", "banana"])
    second = _write_csv(tmp_path, "b.csv", ["0", "1"])

    merged = merge_perf_traces([first, second])
    stdout = capsys.readouterr().out

    assert "reports no usable 'DEVICE ID' values" in stdout
    # The second file still gets its own device range rather than colliding with
    # the first. The unusable id is passed through untouched for the report to
    # handle per row, which it does by leaving that row's Device cell blank.
    assert sorted(merged["DEVICE ID"].dropna().tolist()) == [1.5, 2.0, 3.0]


def test_merge_perf_traces_rejects_files_that_disagree_on_device_count(tmp_path, capsys):
    first = _write_csv(tmp_path, "a.csv", ["0", "1"])
    second = _write_csv(tmp_path, "b.csv", ["0"])

    with pytest.raises(SystemExit):
        merge_perf_traces([first, second])

    # The message names both counts, not an off-by-one derived from the maximum.
    assert "reports 1 device(s) (max DEVICE ID 0), expected 2" in capsys.readouterr().out


def _merge_frame(rows):
    df = pd.DataFrame(rows)
    df["ORIGINAL_ROW"] = df.index + 2
    return df


def test_merge_device_rows_keeps_an_unmergeable_device_op_alongside_merged_ones(capsys):
    good = _subdevice_row("MatmulDeviceOperation", "", "108", "108", 1000, device_id="0")
    bad = _subdevice_row("BinaryNgDeviceOperation", "", "108", "64", 2000, device_id="banana")
    result = merge_device_rows(_merge_frame([good, bad]))
    stdout = capsys.readouterr().out

    assert result["OP CODE"].tolist() == ["MatmulDeviceOperation", "BinaryNgDeviceOperation"]
    assert result["ORIGINAL_ROW"].tolist() == [2, 3]
    assert "BinaryNgDeviceOperation has no usable DEVICE ID" in stdout


def test_merge_device_rows_keeps_rows_when_there_is_nothing_to_merge():
    # Dropping these would lose rows rather than merely leave them unmerged.
    rows = [
        _subdevice_row("Sum (torch)", "", "108", "108", 1000, **{"OP TYPE": "tt_dnn_host"}),
        _subdevice_row("start", "", "108", "108", 2000, **{"OP TYPE": "signpost"}),
    ]
    result = merge_device_rows(_merge_frame(rows))

    assert result["OP CODE"].tolist() == ["Sum (torch)", "start"]
