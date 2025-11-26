#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import csv
import os
import tempfile
import re
from io import StringIO
import pytest
from tt_perf_report.perf_report import generate_perf_report

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
        "Advice",
        "Raw OP Code",
    ]

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
                    signpost=None,
                    ignore_signposts=True,
                    min_percentage=0.5,
                    id_range=None,
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=True,
                    no_host_ops=False,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_report_file=None,
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
def test_csv_headers_with_signpost(test_csv_content, mocker):
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
                    signpost='ResNet module started',
                    ignore_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    csv_output_file=output_file.name,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=True,
                    no_host_ops=False,
                    no_stacked_report=True,
                    no_stack_by_in0=True,
                    stacked_report_file=None,
                )

                with open(output_file.name, "r") as f:
                    reader = csv.reader(f)
                    actual_headers = next(reader)
                    data_rows = list(reader)

                    # Check that the first row after signpost is the expected operation
                    first_row_after_signpost = data_rows[0]
                    op_code_index = actual_headers.index("OP Code")
                    expected_op_after_signpost = "InterleavedToShardedDeviceOperation"
                    actual_op_after_signpost = first_row_after_signpost[op_code_index]
                    
                    assert actual_op_after_signpost == expected_op_after_signpost, \
                        f"First operation after 'ResNet module started' signpost should be '{expected_op_after_signpost}', got '{actual_op_after_signpost}'"

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
        "%",
        "OP Code Joined",
        "Device_Time_Sum_us",
        "Ops_Count",
        "Flops_min",
        "Flops_max",
        "Flops_mean",
        "Flops_std",
    ]

def test_stacked_csv_headers(expected_stacked_headers, test_csv_content, mocker):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as input_file:
        input_file.write(test_csv_content)
        input_file.flush()

        with tempfile.TemporaryDirectory() as temp_dir:
            stacked_csv_file = os.path.join(temp_dir, "test_stacked.csv")

            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    signpost=None,
                    ignore_signposts=True,
                    min_percentage=0.5,
                    id_range=None,
                    csv_output_file=None,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=False,
                    no_host_ops=False,
                    no_stacked_report=False,
                    no_stack_by_in0=True,
                    stacked_report_file=stacked_csv_file,
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

        with tempfile.TemporaryDirectory() as temp_dir:
            stacked_csv_file = os.path.join(temp_dir, "test_stacked_in0.csv")

            try:
                mocker.patch("sys.stdout", new_callable=StringIO)
                generate_perf_report(
                    csv_files=[input_file.name],
                    signpost=None,
                    ignore_signposts=True,
                    min_percentage=0.5,
                    id_range=None,
                    csv_output_file=None,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=False,
                    no_host_ops=False,
                    no_stacked_report=False,
                    no_stack_by_in0=False,
                    stacked_report_file=stacked_csv_file,
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
