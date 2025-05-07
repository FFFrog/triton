# w8a8_block_int8_matmul

## Quickly Start

```Python
python w8a8_block_int8_matmul.py
```

## Output

``Python
Traceback (most recent call last):
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 77, in ttir_to_linalg
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/usr/local/python3.10/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt', '/tmp/tmpr8jddcmo/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmpr8jddcmo/kernel.ttadapter.mlir']' died with <Signals.SIGSEGV: 11>.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/root/Git.d/pytorch/samples/northbound/triton/sglang/w8a8_block_int8_matmul/w8a8_block_int8_matmul.py", line 181, in <module>
    w8a8_block_int8_matmul(A, B, As, Bs, block_size)
  File "/root/Git.d/pytorch/samples/northbound/triton/sglang/w8a8_block_int8_matmul/w8a8_block_int8_matmul.py", line 148, in w8a8_block_int8_matmul
    _w8a8_block_int8_matmul[grid](
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/compiler/compiler.py", line 287, in compile
    next_module = compile_ir(module, metadata)
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 328, in <lambda>
    stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 79, in ttir_to_linalg
    raise HuaweiCompilationError("ConvertTritonIRToLinalgIR", e.stderr.decode('utf-8'))
huawei.HuaweiCompilationError:
///------------------[ERROR][TritonAscend][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.	Program arguments: /root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmpr8jddcmo/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmpr8jddcmo/kernel.ttadapter.mlir
///------------------[ERROR][TritonAscend][END]------------------

[ERROR] 2025-04-24-08:58:31 (PID:3523398, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
```

```
dtype: torch.bfloat16...Error: 

///------------------[ERROR][Triton][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmpqxcixwgd/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmpqxcixwgd/kernel.ttadapter.mlir
#0 0x0000aaaaafa5ef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
#1 0x0000aaaaafa5ca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
#2 0x0000aaaaafa5cb78 SignalHandler(int) Signals.cpp:0:0
#3 0x0000ffff8ee057c0 (linux-vdso.so.1+0x7c0)
#4 0x0000ffff8e8e7d5c ./string/../sysdeps/aarch64/multiarch/../memcpy.S:259:0
#5 0x0000aaaaaec1f6f8 mlir::triton::BlockDataParser::parseExpandDims(mlir::triton::ExpandDimsOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f6f8)
#6 0x0000ffffe712a600 
///------------------[ERROR][Triton][END]------------------


dtype: torch.float16...Error: 

///------------------[ERROR][Triton][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp0fuzpeii/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp0fuzpeii/kernel.ttadapter.mlir
#0 0x0000aaaac59cef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
#1 0x0000aaaac59cca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
#2 0x0000aaaac59ccb78 SignalHandler(int) Signals.cpp:0:0
#3 0x0000ffff969217c0 (linux-vdso.so.1+0x7c0)
#4 0x0000ffff96407d6c ./string/../sysdeps/aarch64/multiarch/../memcpy.S:263:0
#5 0x0000aaaac4b8f6f8 mlir::triton::BlockDataParser::parseExpandDims(mlir::triton::ExpandDimsOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f6f8)
#6 0x0000fffff7feee20 
///------------------[ERROR][Triton][END]------------------


dtype: torch.float32...Error: 

///------------------[ERROR][Triton][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmpc_ggstr9/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmpc_ggstr9/kernel.ttadapter.mlir
#0 0x0000aaaae87bef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
#1 0x0000aaaae87bca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
#2 0x0000aaaae87bcb78 SignalHandler(int) Signals.cpp:0:0
#3 0x0000ffff9e9667c0 (linux-vdso.so.1+0x7c0)
#4 0x0000ffff9e447d64 ./string/../sysdeps/aarch64/multiarch/../memcpy.S:261:0
#5 0x0000aaaae797f6f8 mlir::triton::BlockDataParser::parseExpandDims(mlir::triton::ExpandDimsOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData> > const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f6f8)
#6 0x0000ffffe2126490 
///------------------[ERROR][Triton][END]------------------
```
