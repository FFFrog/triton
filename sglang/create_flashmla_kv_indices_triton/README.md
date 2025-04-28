Traceback (most recent call last):
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 77, in ttir_to_linalg
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/usr/local/python3.10.2/lib/python3.10/subprocess.py", line 524, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt', '/tmp/tmp0vfhoehx/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmp0vfhoehx/kernel.ttadapter.mlir']' returned non-zero exit status 1.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/y30064824/triton/SGL/triton/sglang/create_flashmla_kv_indices_triton/test_create_flashmla_kv_indices_triton.py", line 110, in <module>
    run_create_flashmla_kv_indices()
  File "/home/y30064824/triton/SGL/triton/sglang/create_flashmla_kv_indices_triton/test_create_flashmla_kv_indices_triton.py", line 92, in run_create_flashmla_kv_indices
    create_flashmla_kv_indices_triton[(grid_size,)](
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/compiler/compiler.py", line 287, in compile
    next_module = compile_ir(module, metadata)
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 328, in <lambda>
    stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 79, in ttir_to_linalg
    raise HuaweiCompilationError("ConvertTritonIRToLinalgIR", e.stderr.decode('utf-8'))
huawei.HuaweiCompilationError: 
///------------------[ERROR][TritonAscend][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
/home/y30064824/triton/SGL/triton/sglang/create_flashmla_kv_indices_triton/test_create_flashmla_kv_indices_triton.py:43:39: error: failed to legalize unresolved materialization from () to 'tensor<64xi1>' that remained live after conversion
        mask_out = paged_offset_out <= num_paged
                                      ^
/home/y30064824/triton/SGL/triton/sglang/create_flashmla_kv_indices_triton/test_create_flashmla_kv_indices_triton.py:43:39: note: see current operation: %43 = "builtin.unrealized_conversion_cast"() : () -> tensor<64xi1>
/home/y30064824/triton/SGL/triton/sglang/create_flashmla_kv_indices_triton/test_create_flashmla_kv_indices_triton.py:54:12: note: see existing live user here: tt.store %38, %39, %26 : tensor<64x!tt.ptr<i32>>
            data // PAGED_SIZE,
           ^
/home/y30064824/triton/SGL/triton/sglang/create_flashmla_kv_indices_triton/test_create_flashmla_kv_indices_triton.py:8:0: error: failed to apply Convertion Patterns
/home/y30064824/triton/SGL/triton/sglang/create_flashmla_kv_indices_triton/test_create_flashmla_kv_indices_triton.py:8:0: note: see current operation: 
"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}, {}, {}, {}], function_type = (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, i32, i32, i32, i32, i32, i32) -> (), sym_name = "create_flashmla_kv_indices_triton", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32):
    %0 = "builtin.unrealized_conversion_cast"(%arg3) : (memref<?xi32>) -> !tt.ptr<i32>
    %1 = "arith.constant"() <{value = 4095 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 4096 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 63 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 16 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 1024 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %8 = "tensor.empty"() : () -> tensor<64xi32>
    %9 = "linalg.fill"(%7, %8) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg12: i32, %arg13: i32):
      "linalg.yield"(%arg12) : (i32) -> ()
    }) : (i32, tensor<64xi32>) -> tensor<64xi32>
    %10 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %12 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %13 = "arith.index_cast"(%arg8) : (i32) -> index
    %14 = "memref.reinterpret_cast"(%arg1, %13) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<?xi32>, index) -> memref<1xi32, strided<[1], offset: ?>>
    %15 = "arith.constant"() <{value = 0 : index}> : () -> index
    %16 = "memref.load"(%14, %15) <{nontemporal = false}> : (memref<1xi32, strided<[1], offset: ?>>, index) -> i32
    %17 = "tt.ptr_to_int"(%0) : (!tt.ptr<i32>) -> i64
    %18 = "arith.cmpi"(%17, %11) <{predicate = 1 : i64}> : (i64, i64) -> i1
    %19:2 = "scf.if"(%18) ({
      %60 = "arith.index_cast"(%arg8) : (i32) -> index
      %61 = "memref.reinterpret_cast"(%arg3, %60) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<?xi32>, index) -> memref<1xi32, strided<[1], offset: ?>>
      %62 = "arith.constant"() <{value = 0 : index}> : () -> index
      %63 = "memref.load"(%61, %62) <{nontemporal = false}> : (memref<1xi32, strided<[1], offset: ?>>, index) -> i32
      "scf.yield"(%63, %63) : (i32, i32) -> ()
    }, {
      "scf.yield"(%12, %12) : (i32, i32) -> ()
    }) : (i1) -> (i32, i32)
    %20 = "arith.index_cast"(%arg8) : (i32) -> index
    %21 = "memref.reinterpret_cast"(%arg2, %20) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<?xi32>, index) -> memref<1xi32, strided<[1], offset: ?>>
    %22 = "arith.constant"() <{value = 0 : index}> : () -> index
    %23 = "memref.load"(%21, %22) <{nontemporal = false}> : (memref<1xi32, strided<[1], offset: ?>>, index) -> i32
    %24 = "arith.addi"(%19#1, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %25 = "arith.subi"(%24, %19#0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %26 = "arith.addi"(%25, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %27 = "arith.divsi"(%26, %10) : (i32, i32) -> i32
    %28 = "arith.addi"(%25, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %29 = "arith.divsi"(%28, %2) : (i32, i32) -> i32
    %30 = "arith.muli"(%27, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %31 = "arith.muli"(%16, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %32 = "arith.index_cast"(%31) : (i32) -> index
    %33 = "memref.reinterpret_cast"(%arg0, %32) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<?xi32>, index) -> memref<1xi32, strided<[1], offset: ?>>
    %34 = "arith.index_cast"(%31) : (i32) -> index
    %35 = "arith.index_cast"(%19#0) : (i32) -> index
    %36 = "arith.addi"(%34, %35) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %37 = "memref.reinterpret_cast"(%arg0, %36) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<?xi32>, index) -> memref<1xi32, strided<[1], offset: ?>>
    %38 = "arith.muli"(%arg8, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %39 = "arith.index_cast"(%38) : (i32) -> index
    %40 = "memref.reinterpret_cast"(%arg4, %39) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<?xi32>, index) -> memref<1xi32, strided<[1], offset: ?>>
    "scf.for"(%12, %29, %4) ({
    ^bb0(%arg11: i32):
      %41 = "arith.muli"(%arg11, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %42 = "builtin.unrealized_conversion_cast"() : () -> tensor<64xi1>
      %43 = "builtin.unrealized_conversion_cast"() : () -> tensor<64xi1>
      %44 = "arith.index_cast"(%31) : (i32) -> index
      %45 = "arith.index_cast"(%19#0) : (i32) -> index
      %46 = "arith.addi"(%44, %45) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %47 = "arith.index_cast"(%41) : (i32) -> index
      %48 = "arith.constant"() <{value = 64 : index}> : () -> index
      %49 = "arith.muli"(%47, %48) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %50 = "arith.addi"(%46, %49) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %51 = "memref.reinterpret_cast"(%arg0, %50) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64>, static_strides = array<i64: 64>}> : (memref<?xi32>, index) -> memref<64xi32, strided<[64], offset: ?>>
      %52 = "builtin.unrealized_conversion_cast"(%51) : (memref<64xi32, strided<[64], offset: ?>>) -> tensor<64x!tt.ptr<i32>>
      %53 = "tt.load"(%52, %42) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<64x!tt.ptr<i32>>, tensor<64xi1>) -> tensor<64xi32>
      %54 = "arith.index_cast"(%38) : (i32) -> index
      %55 = "arith.index_cast"(%41) : (i32) -> index
      %56 = "arith.addi"(%54, %55) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %57 = "memref.reinterpret_cast"(%arg4, %56) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64>, static_strides = array<i64: 1>}> : (memref<?xi32>, index) -> memref<64xi32, strided<[1], offset: ?>>
      %58 = "builtin.unrealized_conversion_cast"(%57) : (memref<64xi32, strided<[1], offset: ?>>) -> tensor<64x!tt.ptr<i32>>
      %59 = "arith.divsi"(%53, %9) : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi32>
      "tt.store"(%58, %59, %43) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (tensor<64x!tt.ptr<i32>>, tensor<64xi32>, tensor<64xi1>) -> ()
      "scf.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
    "tt.return"() : () -> ()
  }) {global_kernel = "local", noinline = false} : () -> ()
}) : () -> ()
///------------------[ERROR][TritonAscend][END]------------------

[ERROR] 2025-04-28-10:16:50 (PID:122111, Device:0, RankID:-1) ERR99999 UNKNOWN application exception