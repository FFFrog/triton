Traceback (most recent call last):
  File "/home/y30064824/triton/SGL/triton/sglang/write_zeros_to_output/test_write_zeros_to_output.py", line 69, in <module>
    test_write_zeros_to_output()
  File "/home/y30064824/triton/SGL/triton/sglang/write_zeros_to_output/test_write_zeros_to_output.py", line 50, in test_write_zeros_to_output
    write_zeros_to_output[grid](
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/runtime/jit.py", line 580, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
  File "<string>", line 2, in dynamic_func
  File "/home/y30064824/triton/SGL/triton-ascend/triton/python/triton/runtime/jit.py", line 313, in mangle_type
    dsk = (arg.dtype, is_const)
AttributeError: 'dtype' object has no attribute 'dtype'
[ERROR] 2025-04-29-16:43:35 (PID:136248, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
