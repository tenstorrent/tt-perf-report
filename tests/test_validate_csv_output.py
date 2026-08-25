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
    Cell,
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
    ("tests/data/bh_8xp150_deepseek_v3_d_p.csv", CsvFormat.V2_1, "blackhole", 110)
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
    detected_cores = ArchitectureSpec._get_worker_core_count_from_df(df)
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
