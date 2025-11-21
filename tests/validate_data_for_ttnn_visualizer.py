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
TEST_CSV_CONTENT = """OP CODE,OP TYPE,GLOBAL CALL COUNT,DEVICE ID,ATTRIBUTES,MATH FIDELITY,CORE COUNT,PARALLELIZATION STRATEGY,HOST START TS,HOST END TS,HOST DURATION [ns],DEVICE FW START CYCLE,DEVICE FW END CYCLE,OP TO OP LATENCY [ns],OP TO OP LATENCY BR/NRISC START [ns],DEVICE FW DURATION [ns],DEVICE KERNEL DURATION [ns],DEVICE KERNEL DURATION DM START [ns],DEVICE KERNEL DURATION PER CORE MIN [ns],DEVICE KERNEL DURATION PER CORE MAX [ns],DEVICE KERNEL DURATION PER CORE AVG [ns],DEVICE KERNEL FIRST TO LAST START [ns],DEVICE BRISC KERNEL DURATION [ns],DEVICE NCRISC KERNEL DURATION [ns],DEVICE TRISC0 KERNEL DURATION [ns],DEVICE TRISC1 KERNEL DURATION [ns],DEVICE TRISC2 KERNEL DURATION [ns],DEVICE ERISC KERNEL DURATION [ns],DEVICE COMPUTE CB WAIT FRONT [ns],DEVICE COMPUTE CB RESERVE BACK [ns],DISPATCH TOTAL CQ CMD OP TIME [ns],DISPATCH GO SEND WAIT TIME [ns],INPUT_0_W_PAD[LOGICAL],INPUT_0_Z_PAD[LOGICAL],INPUT_0_Y_PAD[LOGICAL],INPUT_0_X_PAD[LOGICAL],INPUT_0_LAYOUT,INPUT_0_DATATYPE,INPUT_0_MEMORY,INPUT_1_W_PAD[LOGICAL],INPUT_1_Z_PAD[LOGICAL],INPUT_1_Y_PAD[LOGICAL],INPUT_1_X_PAD[LOGICAL],INPUT_1_LAYOUT,INPUT_1_DATATYPE,INPUT_1_MEMORY,INPUT_2_W_PAD[LOGICAL],INPUT_2_Z_PAD[LOGICAL],INPUT_2_Y_PAD[LOGICAL],INPUT_2_X_PAD[LOGICAL],INPUT_2_LAYOUT,INPUT_2_DATATYPE,INPUT_2_MEMORY,INPUT_3_W_PAD[LOGICAL],INPUT_3_Z_PAD[LOGICAL],INPUT_3_Y_PAD[LOGICAL],INPUT_3_X_PAD[LOGICAL],INPUT_3_LAYOUT,INPUT_3_DATATYPE,INPUT_3_MEMORY,OUTPUT_0_W_PAD[LOGICAL],OUTPUT_0_Z_PAD[LOGICAL],OUTPUT_0_Y_PAD[LOGICAL],OUTPUT_0_X_PAD[LOGICAL],OUTPUT_0_LAYOUT,OUTPUT_0_DATATYPE,OUTPUT_0_MEMORY,METAL TRACE ID,METAL TRACE REPLAY SESSION ID,COMPUTE KERNEL SOURCE,COMPUTE KERNEL HASH,DATA MOVEMENT KERNEL SOURCE,DATA MOVEMENT KERNEL HASH,TENSIX DM 0 MAX KERNEL SIZE [B],TENSIX DM 1 MAX KERNEL SIZE [B],TENSIX COMPUTE 0 MAX KERNEL SIZE [B],TENSIX COMPUTE 1 MAX KERNEL SIZE [B],TENSIX COMPUTE 2 MAX KERNEL SIZE [B],ACTIVE ETH DM 0 MAX KERNEL SIZE [B],ACTIVE ETH DM 1 MAX KERNEL SIZE [B],IDLE ETH DM 0 MAX KERNEL SIZE [B],IDLE ETH DM 1 MAX KERNEL SIZE [B],PM IDEAL [ns],PM COMPUTE [ns],PM BANDWIDTH [ns],PM REQ I BW,PM REQ O BW,PM FPU UTIL (%),NOC UTIL (%),DRAM BW UTIL (%),NPE CONG IMPACT (%)
OftNet module started,signpost,,,,,,,11491655433,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
TilizeWithValPadding,tt_dnn_device,1024,0,{'enough_space_height': 'true'; 'enough_space_width': 'true'; 'output_dtype': 'DataType::BFLOAT16'; 'output_mem_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED;buffer_type=BufferType::DRAM;shard_spec=std::nullopt;nd_shard_spec=std::nullopt;created_with_nd_shard_spec=0)'; 'output_padded_shape': 'Shape([1; 384; 1280; 32])'; 'pad_value': '0'; 'use_multicore': 'true'},HiFi4,20,,12029758729,12029892888,134159,17397991088874,17398002833932,0,0,8700043,8699373,8699373,8412818,8699277,8561921,134,8699373,8698988,8698971,8665201,8688181,,,,,,1[1],384[384],1280[1280],3[3],ROW_MAJOR,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,,,,,,,,,,,,,,,,,,,,1[1],384[384],1280[1280],32[3],TILE,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,['ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp'],['tilize/13595278099964611939/'],['ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows_multicore.cpp'; 'ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp'],['reader_unary_pad_dims_split_rows_multicore/16155120703705794183/'; 'writer_unary_interleaved_start_id/1571380508649531845/'],1068,1644,1168,740,1576,0,0,0,0,1048496,1048496,305175,[2.8127145767211914],[30.002288818359375],12.053,,,
BinaryNgDeviceOperation,tt_dnn_device,2048,0,{'binary_op_type': 'BinaryOpType::SUB'; 'compute_kernel_config': 'std::nullopt'; 'dtype': 'DataType::BFLOAT16'; 'input_dtype': 'DataType::BFLOAT16'; 'is_quant_op': 'false'; 'is_sfpu': 'false'; 'lhs_activations': 'SmallVector([])'; 'memory_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED;buffer_type=BufferType::DRAM;shard_spec=std::nullopt;nd_shard_spec=std::nullopt;created_with_nd_shard_spec=0)'; 'post_activations': 'SmallVector([])'; 'rhs_activations': 'SmallVector([])'; 'scalar': 'std::nullopt'; 'subtile_broadcast_type': 'SubtileBroadcastType::ROW_B'; 'worker_grid': '{[(x=0;y=0) - (x=4;y=3)]}'},HiFi4,20,,12623710020,12623839429,129409,17398792909959,17398794593907,585242213,585242213,1247369,1246666,1246666,1245006,1246552,1245900,134,1246666,1246148,1246301,1246203,1246212,,,,,,1[1],384[384],1280[1280],32[3],TILE,BFLOAT16,DEV_1_DRAM_INTERLEAVED,1[1],1[1],32[1],32[3],TILE,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,,,,,,,,,,,,,1[1],384[384],1280[1280],32[3],TILE,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,['ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_row_bcast.cpp'],['eltwise_binary_row_bcast/10529549720625814295/'],['ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp'; 'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_row_bcast.cpp'],['writer_interleaved_no_bcast/14729214739779792163/'; 'reader_interleaved_row_bcast/14938709534218896793/'],1632,2532,1936,1080,2004,0,0,0,0,1,1,1,[],[],0.0,,,
BinaryNgDeviceOperation,tt_dnn_device,3072,0,{'binary_op_type': 'BinaryOpType::DIV'; 'compute_kernel_config': 'std::nullopt'; 'dtype': 'DataType::BFLOAT16'; 'input_dtype': 'DataType::BFLOAT16'; 'is_quant_op': 'false'; 'is_sfpu': 'false'; 'lhs_activations': 'SmallVector([])'; 'memory_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED;buffer_type=BufferType::DRAM;shard_spec=std::nullopt;nd_shard_spec=std::nullopt;created_with_nd_shard_spec=0)'; 'post_activations': 'SmallVector([])'; 'rhs_activations': 'SmallVector([])'; 'scalar': 'std::nullopt'; 'subtile_broadcast_type': 'SubtileBroadcastType::ROW_B'; 'worker_grid': '{[(x=0;y=0) - (x=4;y=3)]}'},HiFi4,20,,13194248623,13194338292,89669,17399563138349,17399564760303,569292890,569292890,1201447,1200734,1200734,1199462,1200633,1200031,133,1200734,1199349,1200323,1200244,1200237,,,,,,1[1],384[384],1280[1280],32[3],TILE,BFLOAT16,DEV_1_DRAM_INTERLEAVED,1[1],1[1],32[1],32[3],TILE,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,,,,,,,,,,,,,1[1],384[384],1280[1280],32[3],TILE,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,['ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_row_bcast.cpp'],['eltwise_binary_row_bcast/9636951885650971701/'],['ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp'; 'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_row_bcast.cpp'],['writer_interleaved_no_bcast/4795360211439300655/'; 'reader_interleaved_row_bcast/16408426625940580189/'],1632,2532,2224,3180,2256,0,0,0,0,1,1,1,[],[],0.0,,,
UntilizeWithUnpadding,tt_dnn_device,4096,0,{'enough_space_height': 'true'; 'enough_space_width': 'true'; 'fp32_dest_acc_en': 'false'; 'output_mem_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED;buffer_type=BufferType::DRAM;shard_spec=std::nullopt;nd_shard_spec=std::nullopt;created_with_nd_shard_spec=0)'; 'output_tensor_end': 'Shape([0; 383; 1279; 2])'; 'use_multicore': 'true'; 'use_pack_untilize': 'true'},HiFi4,20,,13762306936,13762395785,88849,17400330010141,17400339432541,566852451,566852453,6979556,6978839,6978837,6425929,6978756,6839884,127,6978836,6943312,6943353,6924669,6970089,,,,,,1[1],384[384],1280[1280],32[3],TILE,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,,,,,,,,,,,,,,,,,,,,1[1],384[384],1280[1280],3[3],ROW_MAJOR,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,['ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp'],['pack_untilize/7317391041484289947/'],['ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp'; 'ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp'],['reader_unary_interleaved_start_id/10624048636411831976/'; 'writer_unary_stick_layout_split_rows_multicore/10854064466706577341/'],1392,1000,1036,812,2012,0,0,0,0,1048496,1048496,305175,[30.002288818359375],[2.8127145767211914],15.024,,,
ResNet module started,signpost,,,,,,,13762437495,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
InterleavedToShardedDeviceOperation,tt_dnn_device,5120,0,{'keep_l1_aligned': 'false'; 'output_dtype': 'DataType::BFLOAT16'; 'output_mem_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED;buffer_type=BufferType::L1;shard_spec=ShardSpec(grid={[(x=0;y=0) - (x=4;y=3)]};shape={24576; 8};orientation=ShardOrientation::ROW_MAJOR;mode=ShardMode::LOGICAL;physical_shard_shape={24576; 8});nd_shard_spec=std::nullopt;created_with_nd_shard_spec=0)'},,20,,14305268524,14305355634,87110,17401063002958,17401076369659,535978805,535978805,9901260,9900521,9900521,9709019,9900446,9806816,127,9900521,9900509,,,,,,,,,1[1],1[1],491520[491520],3[3],ROW_MAJOR,BFLOAT16,DEV_1_DRAM_INTERLEAVED,,,,,,,,,,,,,,,,,,,,,,1[1],1[1],491520[491520],8[3],ROW_MAJOR,BFLOAT16,DEV_1_L1_HEIGHT_SHARDED,,,[],[],['ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp'; 'ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp'],['reader_unary_stick_layout_sharded_blocks_interleaved_start_id/14511290307302166168/'; 'writer_unary_sharded/7873680606836170382/'],412,1868,0,0,0,0,0,0,0,1016504,1016504,28610,[2.901237964630127],[7.736634731292725],10.267,,,
HaloDeviceOperation,tt_dnn_device,6144,0,{'config_': 'SlidingWindowConfig(batch_size=1; input_hw=(384;1280); window_hw=(7;7); stride_hw=(2;2); padding=((3; 3); (3; 3)); output_padding = (0; 0); dilation_hw=(1;1); num_cores_nhw=20; num_cores_c=1; core_range_set_={[(x=0;y=0) - (x=4;y=3)]})'; 'in_place_': 'false'; 'is_out_tiled_': 'true'; 'max_out_nsticks_per_core_': '32929'; 'output_memory_config_': 'MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED;buffer_type=BufferType::L1;shard_spec=ShardSpec(grid={[(x=0;y=0) - (x=4;y=3)]};shape={24576; 8};orientation=ShardOrientation::ROW_MAJOR;mode=ShardMode::LOGICAL;physical_shard_shape={24576; 8});nd_shard_spec=std::nullopt;created_with_nd_shard_spec=0)'; 'pad_val_': '0'; 'parallel_config_': 'ParallelConfig(grid={[(x=0;y=0) - (x=4;y=3)]}; shard_scheme=HEIGHT_SHARDED; shard_orientation=ROW_MAJOR)'; 'remote_read_': 'false'; 'transpose_mcast_': 'false'},,20,,14872836711,14872923050,86339,17401829212950,17401829224875,557662439,557662439,8833,8092,8092,6684,8019,7478,122,7991,8092,,,,,,,,,1[1],1[1],491520[491520],8[3],ROW_MAJOR,BFLOAT16,DEV_1_L1_HEIGHT_SHARDED,,,,,,,,,,,,,,,,,,,,,,1[1],1[1],658580[658580],8[8],ROW_MAJOR,BFLOAT16,DEV_1_L1_HEIGHT_SHARDED,,,[],[],['ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp'; 'ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp'],['halo_gather/14939481478949778718/'; 'halo_gather/3769533457188046372/'],1520,1412,0,0,0,0,0,0,0,1,1,1,[7864320.0],[10537280.0],0.012,,,
MoveDeviceOperation,tt_dnn_device,7168,0,{'move_op_parallelization_strategy': 'MoveOpParallelizationStrategy::MULTI_CORE_SHARDED'; 'output_mem_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED;buffer_type=BufferType::L1;shard_spec=ShardSpec(grid={[(x=0;y=0) - (x=4;y=3)]};shape={32929; 8};orientation=ShardOrientation::ROW_MAJOR;mode=ShardMode::LOGICAL;physical_shard_shape={32929; 8});nd_shard_spec=std::nullopt;created_with_nd_shard_spec=0)'},,20,,15368889276,15368974565,85289,17402498872596,17402498882765,496036074,496036074,7533,6807,6807,6652,6707,6678,118,,6807,,,,,,,,,1[1],1[1],658580[658580],8[8],ROW_MAJOR,BFLOAT16,DEV_1_L1_HEIGHT_SHARDED,1[1],1[1],658580[658580],8[8],ROW_MAJOR,BFLOAT16,DEV_1_L1_HEIGHT_SHARDED,,,,,,,,,,,,,,,1[1],1[1],658580[658580],8[8],ROW_MAJOR,BFLOAT16,DEV_1_L1_HEIGHT_SHARDED,,,[],[],['ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp'],['reader_unary_local_l1_copy_backwards/1054765016481697778/'],0,1228,0,0,0,0,0,0,0,11448,11448,1,[920.447265625; 920.447265625],[920.447265625],168.18,,,"""


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
