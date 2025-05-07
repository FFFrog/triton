# per_token_quant_int8

## Quickly Start

```Python
python per_token_quant_int8.py
```

## Output

```Python
Traceback (most recent call last):
  File "/root/Git.d/pytorch/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 200, in linalg_to_bin_enable_npu_compile
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/usr/local/python3.10/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/root/Git.d/tools/npuc', '/tmp/tmpv3svrise/kernel.ttadapter.mlir', '--enable-auto-multi-buffer=true', '-o', '/tmp/tmpv3svrise/kernel']' returned non-zero exit status 1.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/root/Git.d/pytorch/samples/northbound/triton/sglang/per_token_quant_int8/per_token_quant_int8.py", line 62, in <module>
    per_token_quant_int8(x)
  File "/root/Git.d/pytorch/samples/northbound/triton/sglang/per_token_quant_int8/per_token_quant_int8.py", line 45, in per_token_quant_int8
    _per_token_quant_int8[(M,)](
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
loc("/tmp/tmpv3svrise/kernel.ttadapter.mlir":47:11): error: Dialect `tt' not found for custom op 'tt.extern_elementwise'
Error opening /tmp/tmpv3svrise/kernel.ttadapter.mlir
///------------------[ERROR][TritonAscend][END]------------------

[ERROR] 2025-04-24-03:46:30 (PID:3499025, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
```

```
dtype: torch.bfloat16...Error: 

///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc("/tmp/tmpxuynfzq1/kernel.ttadapter.mlir":48:11): error: Dialect `tt' not found for custom op 'tt.extern_elementwise' 
Error opening /tmp/tmpxuynfzq1/kernel.ttadapter.mlir
///------------------[ERROR][Triton][END]------------------


dtype: torch.float16...Error: 

///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc("/tmp/tmp56ml6n25/kernel.ttadapter.mlir":48:11): error: Dialect `tt' not found for custom op 'tt.extern_elementwise' 
Error opening /tmp/tmp56ml6n25/kernel.ttadapter.mlir
///------------------[ERROR][Triton][END]------------------


dtype: torch.float32...Error: 

///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc("/tmp/tmpjt_a7z_2/kernel.ttadapter.mlir":47:11): error: Dialect `tt' not found for custom op 'tt.extern_elementwise' 
Error opening /tmp/tmpjt_a7z_2/kernel.ttadapter.mlir
///------------------[ERROR][Triton][END]------------------
```
