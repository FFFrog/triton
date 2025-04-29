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


def test_write_zeros_to_output():
    device = 'npu'
    dtype = torch.bfloat16
    M, N = 64, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 64
    pid_n = 0

    # 创建输出张量
    C = torch.randn(M, N, device=device, dtype=dtype)

    # 构造参数
    c_ptr = C.data_ptr()
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # 构造 token id 和 mask
    offs_token = torch.arange(M, device=device)
    token_mask = (offs_token < 32)  # 前32个是有效的token

    # 使用 grid 来启动 kernel
    def grid(meta):
        return (1,)  # 只需要一个 block 测试即可

    # 启动 kernel
    write_zeros_to_output[grid](
        c_ptr=c_ptr,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        pid_n=pid_n,
        N=N,
        offs_token=offs_token,
        token_mask=token_mask,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        compute_type=tl.bfloat16,
    )

    # 检查结果
    print("Output tensor after zero writing:")
    print(C[:8, :8].cpu().to(torch.float32))


if __name__ == "__main__":
    test_write_zeros_to_output()