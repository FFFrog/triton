import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def launch_write_zeros_to_output(
    output: torch.Tensor,
    token_ids: torch.Tensor,
    valid_tokens_mask: torch.Tensor,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
):
    """
    Args:
        output: Tensor[M, N], 输出张量，将被部分置零。
        token_ids: Tensor[TOKENS], 有效 token 的行索引（如 [0, 1, 2] 表示前三行）。
        valid_tokens_mask: Tensor[TOKENS], 每个 token 是否有效的掩码（bool 或 int 类型）。
        BLOCK_SIZE_M/N: 块大小，需编译时确定。
    """
    M, N = output.shape
    grid = (1,)  # 当前只处理第一个 n-block，可根据需要扩展

    if output.dtype == torch.float16:
        compute_type = tl.float16
    elif output.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif output.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {output.dtype}")

    write_zeros_to_output[grid](
        c_ptr=output,
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        pid_n=0,
        N=N,
        offs_token=token_ids,
        token_mask=valid_tokens_mask,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        compute_type=compute_type,
    )

if __name__ == "__main__":
    # 构造参数
    M = 128   # 所有 token 总数
    N = 64    # 每个 token 的维度
    TOKENS = 8  # 本次操作的有效 token 数量

    output = torch.randn(M, N, device="npu", dtype=torch.float16)
    print("Before zeroing:\n", output[:TOKENS, :16])

    token_ids = torch.tensor([i for i in range(TOKENS)], device="npu", dtype=torch.int32)
    valid_tokens_mask = torch.tensor([True] * TOKENS, device="npu")

    # 调用 launcher
    launch_write_zeros_to_output(output, token_ids, valid_tokens_mask)

    # torch.cuda.synchronize()

    print("\nAfter zeroing:\n", output[:TOKENS, :16])  # 应该变为 0