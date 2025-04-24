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
