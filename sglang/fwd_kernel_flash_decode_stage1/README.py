Traceback (most recent call last):
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 77, in ttir_to_linalg
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/usr/local/python3.10.2/lib/python3.10/subprocess.py", line 524, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt', '/tmp/tmp2m5i5pyz/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmp2m5i5pyz/kernel.ttadapter.mlir']' died with <Signals.SIGSEGV: 11>.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/y30064824/triton/SGL/triton/sglang/fwd_kernel_flash_decode_stage1/test_fwd_kernel_flash_decode_stage1.py", line 198, in <module>
    flash_decode_stage1(
  File "/usr/local/python3.10.2/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/y30064824/triton/SGL/triton/sglang/fwd_kernel_flash_decode_stage1/test_fwd_kernel_flash_decode_stage1.py", line 142, in flash_decode_stage1
    _fwd_kernel_flash_decode_stage1[grid](
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
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/y30064824/triton/SGL/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp2m5i5pyz/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp2m5i5pyz/kernel.ttadapter.mlir
///------------------[ERROR][TritonAscend][END]------------------

[ERROR] 2025-04-28-11:12:18 (PID:122861, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
