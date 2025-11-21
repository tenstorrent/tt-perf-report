#!/usr/bin/env python3
"""
Test suite for tt-perf-report CSV output functionality.
"""

import csv
import os
import sys
import tempfile
import unittest
import re
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tt_perf_report.perf_report import generate_perf_report

# Shared test data (sample output from TT-NN)
TEST_CSV_CONTENT = """OP TYPE,OP CODE,GLOBAL CALL COUNT,DEVICE KERNEL DURATION [ns],HOST START TS,DEVICE ID,CORE COUNT,INPUT_0_MEMORY,INPUT_1_MEMORY,OUTPUT_0_MEMORY,INPUT_0_DATATYPE,INPUT_1_DATATYPE,OUTPUT_0_DATATYPE,MATH FIDELITY,INPUT_0_W_PAD[LOGICAL],INPUT_0_Y_PAD[LOGICAL],INPUT_0_Z_PAD[LOGICAL],INPUT_0_X_PAD[LOGICAL],INPUT_1_W_PAD[LOGICAL],INPUT_1_Y_PAD[LOGICAL],INPUT_1_Z_PAD[LOGICAL],INPUT_1_X_PAD[LOGICAL],OUTPUT_0_W_PAD[LOGICAL],OUTPUT_0_Y_PAD[LOGICAL],OUTPUT_0_Z_PAD[LOGICAL],OUTPUT_0_X_PAD[LOGICAL],ATTRIBUTES,OP TO OP LATENCY [ns]
signpost,test_signpost,1,,1000,0,,,,,,,,,,,,,,,,,,,,,
tt_dnn_device,Matmul,2,5000000,2000,0,64,DEV_0_L1,DEV_0_L1,DEV_0_L1,BFLOAT16,BFLOAT16,BFLOAT16,HiFi4,1,1024,1,1024,1,1024,1,1024,1,1024,1,1024,,1000000
tt_dnn_device,LayerNorm,3,2000000,8000,0,32,DEV_0_L1,DEV_0_L1,DEV_0_L1,BFLOAT16,BFLOAT16,BFLOAT16,HiFi2,1,1024,1,1024,1,1024,1,1024,1,1024,1,1024,,500000
host_op,(torch) some_op,4,,,0,,,,,,,,,,,,,,,,,,,,1000000"""


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

    # TTNN-Visualizer default request
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

                    self.assertTrue(
                        os.path.exists(output_file.name),
                        "Output CSV file should be created",
                    )

                    with open(output_file.name, "r") as f:
                        reader = csv.reader(f)
                        actual_headers = next(reader)

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
                            ignore_signposts=False,
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
                            ignore_signposts=False,
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

                # Clean up
                finally:
                    try:
                        os.unlink(input_file.name)
                        os.unlink(stacked_csv_file)
                    except:
                        pass


if __name__ == "__main__":
    unittest.main()
