# decode_kernel

## Quick Start

```Python
python decode_kernel.py
```

## Output

```Python
Traceback (most recent call last):
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 200, in linalg_to_bin_enable_npu_compile
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/usr/local/python3.10/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/root/Git.d/tools/npuc', '/tmp/tmpu8bohsyt/kernel.ttadapter.mlir', '--enable-auto-multi-buffer=true', '-o', '/tmp/tmpu8bohsyt/kernel']' returned non-zero exit status 1.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/root/Git.d/pytorch/samples/northbound/_decode_kernel.py", line 114, in <module>
    decode_kernel()
  File "/root/Git.d/pytorch/samples/northbound/_decode_kernel.py", line 90, in decode_kernel
    _decode_kernel[grid](
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/compiler/compiler.py", line 287, in compile
    next_module = compile_ir(module, metadata)
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 329, in <lambda>
    stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile(src, metadata, options)
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 202, in linalg_to_bin_enable_npu_compile
    raise HuaweiCompilationError("ConvertLinalgRToBinary", e.stderr.decode('utf-8'))
huawei.HuaweiCompilationError:
///------------------[ERROR][TritonAscend][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc("/tmp/tmpu8bohsyt/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmpu8bohsyt/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmpu8bohsyt/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmpu8bohsyt/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmpu8bohsyt/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

Error run BiShengIR pipeline pipeline
///------------------[ERROR][TritonAscend][END]------------------

[ERROR] 2025-04-24-03:17:50 (PID:3496068, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
```

```
dtype: torch.bfloat16...Error: 

///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc("/tmp/tmpwxjug5e8/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmpwxjug5e8/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmpwxjug5e8/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmpwxjug5e8/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmpwxjug5e8/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

Error run BiShengIR pipeline pipeline 
///------------------[ERROR][Triton][END]------------------


dtype: torch.float16...Success
dtype: torch.float32...Error: 

///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc("/tmp/tmp1dx5q6x7/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmp1dx5q6x7/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmp1dx5q6x7/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmp1dx5q6x7/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

loc("/tmp/tmp1dx5q6x7/kernel.ttadapter.mlir":1:1): error: run BiShengHIR pipeline failed

Error run BiShengIR pipeline pipeline 
///------------------[ERROR][Triton][END]------------------
```
