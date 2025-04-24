```Python
Traceback (most recent call last):
  File "/home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py", line 5, in <module>
    device = torch.device("npu:0")
RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: npu
root@d4c013b6e5eb:/home/y30064824/triton/SGL/sglang# python3 /home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py
Traceback (most recent call last):
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 77, in ttir_to_linalg
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/usr/local/python3.10.2/lib/python3.10/subprocess.py", line 524, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt', '/tmp/tmp5b01jh77/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmp5b01jh77/kernel.ttadapter.mlir']' returned non-zero exit status 1.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py", line 107, in <module>
    test_get_num_kv_splits_triton()
  File "/home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py", line 89, in test_get_num_kv_splits_triton
    get_num_kv_splits_triton[grid](
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
/home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py:64:30: error: failed to legalize unresolved materialization from () to 'tensor<16xi1>' that remained live after conversion
    mask_token = offs_token < num_seq * num_group
                             ^
/home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py:64:30: note: see current operation: %94 = "builtin.unrealized_conversion_cast"() : () -> tensor<16xi1>
/home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py:68:12: note: see existing live user here: tt.store %70, %64, %66 : tensor<16x!tt.ptr<i32>>
            num_kv_splits,
           ^
/home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py:10:0: error: failed to apply Convertion Patterns
/home/y30064824/triton/SGL/sglang/python/sglang/srt/layers/attention/test_triton_backend.py:10:0: note: see current operation:
"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {tt.divisibility = 16 : i32}, {}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}, {}, {}, {}], function_type = (memref<?xi32>, memref<?xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "get_num_kv_splits_triton", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32):
    %0 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %1 = "tensor.empty"() : () -> tensor<1xf32>
    %2 = "linalg.fill"(%0, %1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg43: f32, %arg44: f32):
      "linalg.yield"(%arg43) : (f32) -> ()
    }) : (f32, tensor<1xf32>) -> tensor<1xf32>
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    %4 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
    %5 = "tensor.empty"() : () -> tensor<1xf32>
    %6 = "linalg.fill"(%4, %5) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg41: f32, %arg42: f32):
      "linalg.yield"(%arg41) : (f32) -> ()
    }) : (f32, tensor<1xf32>) -> tensor<1xf32>
    %7 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %8 = "tensor.empty"() : () -> tensor<16xi32>
    %9 = "linalg.fill"(%7, %8) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg39: i32, %arg40: i32):
      "linalg.yield"(%arg39) : (i32) -> ()
    }) : (i32, tensor<16xi32>) -> tensor<16xi32>
    %10 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %12 = "arith.constant"() <{value = 16 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 10 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %15 = "memref.reinterpret_cast"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16>, static_strides = array<i64: 1>}> : (memref<?xi32>) -> memref<16xi32, strided<[1]>>
    %16 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xi32>
    %17 = "arith.index_cast"(%arg2) : (i32) -> index
    %18 = "arith.constant"() <{value = 0 : index}> : () -> index
    %19 = "arith.maxsi"(%18, %17) : (index, index) -> index
    %20 = "arith.constant"() <{value = 16 : index}> : () -> index
    %21 = "arith.minsi"(%20, %19) : (index, index) -> index
    %22 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %23 = "arith.constant"() <{value = false}> : () -> i1
    %24 = "arith.constant"() <{value = 16 : index}> : () -> index
    %25 = "arith.cmpi"(%21, %24) <{predicate = 2 : i64}> : (index, index) -> i1
    %26 = "arith.ori"(%23, %25) : (i1, i1) -> i1
    "scf.if"(%26) ({
      "linalg.fill"(%22, %16) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg37: i32, %arg38: i32):
        "linalg.yield"(%arg37) : (i32) -> ()
      }) : (i32, memref<16xi32>) -> ()
      "scf.yield"() : () -> ()
    }, {
    }) : (i1) -> ()
    %27 = "memref.subview"(%15, %21) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<16xi32, strided<[1]>>, index) -> memref<?xi32, strided<[1]>>
    %28 = "memref.subview"(%16, %21) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<16xi32>, index) -> memref<?xi32, strided<[1]>>
    "memref.copy"(%27, %28) : (memref<?xi32, strided<[1]>>, memref<?xi32, strided<[1]>>) -> ()
    %29 = "bufferization.to_tensor"(%16) <{restrict, writable}> : (memref<16xi32>) -> tensor<16xi32>
    %30 = "arith.constant"() <{value = -2147483648 : i32}> : () -> i32
    %31 = "bufferization.alloc_tensor"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> tensor<i32>
    %32 = "linalg.fill"(%30, %31) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg35: i32, %arg36: i32):
      "linalg.yield"(%arg35) : (i32) -> ()
    }) : (i32, tensor<i32>) -> tensor<i32>
    %33 = "linalg.reduce"(%29, %32) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg33: i32, %arg34: i32):
      %110 = "arith.maxsi"(%arg33, %arg34) : (i32, i32) -> i32
      "linalg.yield"(%110) : (i32) -> ()
    }) : (tensor<16xi32>, tensor<i32>) -> tensor<i32>
    %34 = "tensor.extract"(%33) : (tensor<i32>) -> i32
    %35 = "arith.constant"() <{value = 2147483647 : i32}> : () -> i32
    %36 = "bufferization.alloc_tensor"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> tensor<i32>
    %37 = "linalg.fill"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg31: i32, %arg32: i32):
      "linalg.yield"(%arg31) : (i32) -> ()
    }) : (i32, tensor<i32>) -> tensor<i32>
    %38 = "linalg.reduce"(%29, %37) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg29: i32, %arg30: i32):
      %109 = "arith.minsi"(%arg29, %arg30) : (i32, i32) -> i32
      "linalg.yield"(%109) : (i32) -> ()
    }) : (tensor<16xi32>, tensor<i32>) -> tensor<i32>
    %39 = "tensor.extract"(%38) : (tensor<i32>) -> i32
    %40 = "arith.muli"(%34, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %41 = "arith.muli"(%39, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %42 = "arith.cmpi"(%40, %41) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %43 = "arith.select"(%42, %34, %39) : (i1, i32, i32) -> i32
    %44 = "arith.addi"(%34, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %45 = "arith.subi"(%44, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %46 = "arith.divsi"(%45, %43) : (i32, i32) -> i32
    %47 = "arith.minsi"(%46, %arg6) : (i32, i32) -> i32
    %48 = "arith.addi"(%34, %47) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %49 = "arith.subi"(%48, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %50 = "arith.divsi"(%49, %47) : (i32, i32) -> i32
    %51 = "arith.sitofp"(%34) : (i32) -> f32
    %52 = "tensor.empty"() : () -> tensor<1xf32>
    %53 = "linalg.fill"(%51, %52) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg27: f32, %arg28: f32):
      "linalg.yield"(%arg27) : (f32) -> ()
    }) : (f32, tensor<1xf32>) -> tensor<1xf32>
    %54 = "arith.divf"(%53, %6) <{fastmath = #arith.fastmath<none>}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %55 = "tensor.extract"(%54, %3) : (tensor<1xf32>, index) -> f32
    %56 = "tensor.empty"() : () -> tensor<1xf32>
    %57 = "linalg.fill"(%55, %56) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg25: f32, %arg26: f32):
      "linalg.yield"(%arg25) : (f32) -> ()
    }) : (f32, tensor<1xf32>) -> tensor<1xf32>
    %58 = "math.log2"(%57) <{fastmath = #arith.fastmath<none>}> : (tensor<1xf32>) -> tensor<1xf32>
    %59 = "tensor.extract"(%58, %3) : (tensor<1xf32>, index) -> f32
    %60 = "tensor.empty"() : () -> tensor<1xf32>
    %61 = "linalg.fill"(%59, %60) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg23: f32, %arg24: f32):
      "linalg.yield"(%arg23) : (f32) -> ()
    }) : (f32, tensor<1xf32>) -> tensor<1xf32>
    %62 = "arith.maxnumf"(%61, %2) <{fastmath = #arith.fastmath<none>}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %63 = "tensor.extract"(%62, %3) : (tensor<1xf32>, index) -> f32
    %64 = "arith.sitofp"(%arg7) : (i32) -> f32
    %65 = "tensor.empty"() : () -> tensor<1xf32>
    %66 = "linalg.fill"(%64, %65) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg21: f32, %arg22: f32):
      "linalg.yield"(%arg21) : (f32) -> ()
    }) : (f32, tensor<1xf32>) -> tensor<1xf32>
    %67 = "tensor.empty"() : () -> tensor<1xf32>
    %68 = "linalg.fill"(%63, %67) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg19: f32, %arg20: f32):
      "linalg.yield"(%arg19) : (f32) -> ()
    }) : (f32, tensor<1xf32>) -> tensor<1xf32>
    %69 = "arith.mulf"(%66, %68) <{fastmath = #arith.fastmath<none>}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %70 = "tensor.extract"(%69, %3) : (tensor<1xf32>, index) -> f32
    %71 = "arith.fptosi"(%70) : (f32) -> i32
    %72 = "arith.divsi"(%arg4, %arg5) : (i32, i32) -> i32
    %73 = "arith.cmpi"(%72, %11) <{predicate = 0 : i64}> : (i32, i32) -> i1
    %74 = "scf.if"(%73) ({
      %107 = "arith.muli"(%arg2, %arg3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %108 = "arith.muli"(%107, %arg4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.yield"(%108) : (i32) -> ()
    }, {
      %101 = "arith.minsi"(%72, %12) : (i32, i32) -> i32
      %102 = "arith.muli"(%arg2, %arg3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %103 = "arith.addi"(%arg4, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %104 = "arith.subi"(%103, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %105 = "arith.divsi"(%104, %101) : (i32, i32) -> i32
      %106 = "arith.muli"(%102, %105) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.yield"(%106) : (i32) -> ()
    }) : (i1) -> i32
    %75 = "arith.addi"(%71, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %76 = "arith.subi"(%75, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %77 = "arith.divsi"(%76, %74) : (i32, i32) -> i32
    %78 = "arith.minsi"(%77, %arg6) : (i32, i32) -> i32
    %79 = "arith.addi"(%34, %78) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %80 = "arith.subi"(%79, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %81 = "arith.divsi"(%80, %78) : (i32, i32) -> i32
    %82 = "tensor.empty"() : () -> tensor<16xi32>
    %83 = "linalg.fill"(%50, %82) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg17: i32, %arg18: i32):
      "linalg.yield"(%arg17) : (i32) -> ()
    }) : (i32, tensor<16xi32>) -> tensor<16xi32>
    %84 = "arith.addi"(%29, %83) <{overflowFlags = #arith.overflow<none>}> : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %85 = "arith.subi"(%84, %9) <{overflowFlags = #arith.overflow<none>}> : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %86 = "arith.divsi"(%85, %83) : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %87 = "tensor.empty"() : () -> tensor<16xi32>
    %88 = "linalg.fill"(%81, %87) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg15: i32, %arg16: i32):
      "linalg.yield"(%arg15) : (i32) -> ()
    }) : (i32, tensor<16xi32>) -> tensor<16xi32>
    %89 = "arith.addi"(%29, %88) <{overflowFlags = #arith.overflow<none>}> : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %90 = "arith.subi"(%89, %9) <{overflowFlags = #arith.overflow<none>}> : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %91 = "arith.divsi"(%90, %88) : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %92 = "arith.maxsi"(%86, %91) : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %93 = "arith.muli"(%arg2, %arg3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %94 = "builtin.unrealized_conversion_cast"() : () -> tensor<16xi1>
    "scf.for"(%10, %arg3, %11) ({
    ^bb0(%arg14: i32):
      %95 = "arith.index_cast"(%arg14) : (i32) -> index
      %96 = "memref.reinterpret_cast"(%arg0, %95) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<?xi32>, index) -> memref<1xi32, strided<[1], offset: ?>>
      %97 = "arith.index_cast"(%arg14) : (i32) -> index
      %98 = "arith.index_cast"(%arg3) : (i32) -> index
      %99 = "memref.reinterpret_cast"(%arg0, %97, %98) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 16>, static_strides = array<i64: -9223372036854775808>}> : (memref<?xi32>, index, index) -> memref<16xi32, strided<[?], offset: ?>>
      %100 = "builtin.unrealized_conversion_cast"(%99) : (memref<16xi32, strided<[?], offset: ?>>) -> tensor<16x!tt.ptr<i32>>
      "tt.store"(%100, %92, %94) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>, tensor<16xi1>) -> ()
      "scf.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
    "tt.return"() : () -> ()
  }) {global_kernel = "local", noinline = false} : () -> ()
}) : () -> ()
///------------------[ERROR][TritonAscend][END]------------------
```

[ERROR] 2025-04-24-11:48:38 (PID:117301, Device:0, RankID:-1) ERR99999 UNKNOWN application exception

