#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import csv
import os
import sys
import tempfile
import unittest
import re
from io import StringIO
from unittest.mock import patch
from tt_perf_report.perf_report import generate_perf_report

# Shared test data (sample output from TT-NN)
def read_test_csv_content():
    csv_file_path = os.path.join(os.path.dirname(__file__), "data", "ops_perf_results_2025_09_18_11_39_20.csv")
    
    try:
        with open(csv_file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Test CSV file not found at {csv_file_path}")
    
TEST_CSV_CONTENT = read_test_csv_content()

class TestCSVOutput(unittest.TestCase):
    def setUp(self):
        # Expected headers (visible_headers + additional_headers)
        self.expected_headers = [
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
    def test_csv_headers_with_all_options(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as input_file:
            input_file.write(TEST_CSV_CONTENT)
            input_file.flush()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as output_file:
                try:
                    with patch("sys.stdout", new_callable=StringIO):
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

                    self.assertTrue(
                        os.path.exists(output_file.name),
                        "Output CSV file should be created",
                    )

                    with open(output_file.name, "r") as f:
                        reader = csv.reader(f)
                        actual_headers = next(reader)
                        signposts = []

                        # Test that all expected headers are present and in the correct order
                        self.assertEqual(
                            len(actual_headers),
                            len(self.expected_headers),
                            f"Column count mismatch. Expected {len(self.expected_headers)}, got {len(actual_headers)}",
                        )
                        for i, (expected, actual) in enumerate(
                            zip(self.expected_headers, actual_headers)
                        ):
                            self.assertEqual(
                                actual,
                                expected,
                                f"Column {i} mismatch. Expected '{expected}', got '{actual}'",
                            )

                        with open(output_file.name, "r") as f:
                            reader = csv.DictReader(f)
                            input_0_memory_pattern = re.compile(r"DEV_(\d+)_(DRAM|L1)")

                            for row in reader:
                                input_0_memory = row.get("Input 0 Memory")
                                advice_field = row.get("Advice", "")

                                print(row.get("OP Code"))

                                if "(signpost)" in row.get("OP Code", ""):
                                    signposts.append(row)

                                # Test that advice can be split as TT-NN Visualizer expects
                                if advice_field and advice_field.strip():
                                    advice_items = advice_field.split(" • ")
                                    self.assertIsInstance(
                                        advice_items,
                                        list,
                                        "Advice should be splittable into a list",
                                    )
                                    for item in advice_items:
                                        self.assertIsInstance(
                                            item.strip(),
                                            str,
                                            f"Advice item '{item}' should be a string",
                                        )
                                        self.assertGreater(
                                            len(item.strip()),
                                            0,
                                            f"Advice item '{item}' should not be empty",
                                        )

                                # Test Input 0 Memory values
                                if input_0_memory and input_0_memory.strip():
                                    match = input_0_memory_pattern.match(input_0_memory)
                                    self.assertIsNotNone(
                                        match,
                                        f"Input 0 Memory value '{input_0_memory}' does not match pattern 'DEV_(\\d+)_(DRAM|L1)'",
                                    )

                                    device_id, memory_type = match.groups()
                                    self.assertTrue(
                                        device_id.isdigit(),
                                        f"Device ID '{device_id}' should be a digit",
                                    )
                                    self.assertIn(
                                        memory_type,
                                        ["DRAM", "L1"],
                                        f"Memory type '{memory_type}' should be DRAM or L1",
                                    )

                    # Ensure that signpost rows are captured
                    self.assertGreater(2, 0, "Two signpost rows should be present")

                # Clean up
                finally:
                    try:
                        os.unlink(input_file.name)
                        os.unlink(output_file.name)
                    except:
                        pass

    # TT-NN Visualizer request with signpost
    def test_csv_headers_with_signpost(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as input_file:
            input_file.write(TEST_CSV_CONTENT)
            input_file.flush()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as output_file:
                try:
                    with patch("sys.stdout", new_callable=StringIO):
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
                        # After "ResNet module started" the next operation is "InterleavedToShardedDeviceOperation"
                        first_row_after_signpost = data_rows[0]
                        op_code_index = actual_headers.index("OP Code")
                        expected_op_after_signpost = "InterleavedToShardedDeviceOperation"
                        actual_op_after_signpost = first_row_after_signpost[op_code_index]
                        
                        self.assertEqual(
                            actual_op_after_signpost,
                            expected_op_after_signpost,
                            f"First operation after 'ResNet module started' signpost should be '{expected_op_after_signpost}', got '{actual_op_after_signpost}'"
                        )

                # Clean up
                finally:
                    try:
                        os.unlink(input_file.name)
                        os.unlink(output_file.name)
                    except:
                        pass


class TestStackedCSVOutput(unittest.TestCase):
    def setUp(self):
        self.expected_stacked_headers = [
            "%",
            "OP Code Joined",
            "Device_Time_Sum_us",
            "Ops_Count",
            "Flops_min",
            "Flops_max",
            "Flops_mean",
            "Flops_std",
        ]

    def test_stacked_csv_headers(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as input_file:
            input_file.write(TEST_CSV_CONTENT)
            input_file.flush()

            with tempfile.TemporaryDirectory() as temp_dir:
                stacked_csv_file = os.path.join(temp_dir, "test_stacked.csv")

                try:
                    with patch("sys.stdout", new_callable=StringIO):
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

                    self.assertTrue(
                        os.path.exists(stacked_csv_file),
                        "Stacked CSV file should be created",
                    )

                    with open(stacked_csv_file, "r") as f:
                        reader = csv.reader(f)
                        actual_headers = next(reader)

                        # Test that all expected stacked headers are present and in the correct order
                        self.assertEqual(
                            actual_headers,
                            self.expected_stacked_headers,
                            "Stacked CSV headers do not match expected headers",
                        )
                        data_rows = list(reader)
                        self.assertGreater(
                            len(data_rows), 0, "Stacked CSV should contain data rows"
                        )
                        for i, (expected, actual) in enumerate(
                            zip(self.expected_stacked_headers, actual_headers)
                        ):
                            self.assertEqual(
                                actual,
                                expected,
                                f"Stacked column {i} mismatch. Expected '{expected}', got '{actual}'. This will break csv_queries.py column mapping in TT-NN Visualizer.",
                            )

                        # Ensure that no signpost rows are present
                        for row in reader:
                            op_code_joined = row.get("OP Code Joined", "")
                            self.assertNotIn(
                                "(signpost)",
                                op_code_joined,
                                f"Stacked CSV should not contain signpost rows, but found: {op_code_joined}",
                            )

                # Clean up
                finally:
                    try:
                        os.unlink(input_file.name)
                        os.unlink(stacked_csv_file)
                    except:
                        pass

    def test_stacked_csv_headers_with_input0_layout(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as input_file:
            input_file.write(TEST_CSV_CONTENT)
            input_file.flush()

            with tempfile.TemporaryDirectory() as temp_dir:
                stacked_csv_file = os.path.join(temp_dir, "test_stacked_in0.csv")

                try:
                    with patch("sys.stdout", new_callable=StringIO):
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
                        self.assertEqual(
                            actual_headers,
                            self.expected_stacked_headers,
                            "Stacked CSV headers should be the same regardless of input0 layout grouping",
                        )
                        data_rows = list(reader)
                        self.assertGreater(
                            len(data_rows), 0, "Stacked CSV should contain data rows"
                        )

                        # Test that OP Code Joined includes input 0 layout info
                        op_code_joined_values = [
                            row[1] for row in data_rows
                        ]  # Column 1 is OP Code Joined
                        has_layout_info = any(
                            "(in0:" in op_code for op_code in op_code_joined_values
                        )
                        self.assertTrue(
                            has_layout_info,
                            "OP Code Joined should include input 0 layout information",
                        )

                        # Ensure that no signpost rows are present
                        for row in reader:
                            op_code_joined = row.get("OP Code Joined", "")
                            self.assertNotIn(
                                "(signpost)",
                                op_code_joined,
                                f"Stacked CSV should not contain signpost rows, but found: {op_code_joined}",
                            )

                # Clean up
                finally:
                    try:
                        os.unlink(input_file.name)
                        os.unlink(stacked_csv_file)
                    except:
                        pass


if __name__ == "__main__":
    unittest.main()
