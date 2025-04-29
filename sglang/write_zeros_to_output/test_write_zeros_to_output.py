import torch
import triton
import triton.language as tl

# 你提供的 write_zeros_to_output kernel
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
    # 设置参数
    device = "cuda"
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    M = 64  # 总共的 token 数量
    N = 128  # 输出维度
    num_valid_tokens = 48  # 前面的有效 token 数量

    # 创建输出张量（初始化为随机值，我们之后会写零进去）
    C = torch.randn((M, N), device=device, dtype=torch.bfloat16)

    # 构造输入参数
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # 假设当前处理第 pid_n = 0 个 block column
    pid_n = 0

    # token id：假设前 num_valid_tokens 是有效的
    offs_token = torch.arange(M, device=device)
    token_mask = (offs_token < num_valid_tokens)

    # 启动 kernel
    grid = lambda META: (1,)  # 只需要一个 block 来做测试
    write_zeros_to_output[grid](
        c_ptr=C.data_ptr(),
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

    # 打印部分结果确认是否成功写入 0
    print("First few rows of output matrix C:")
    print(C[:8, :8].cpu().to(torch.float32))  # 转成 float 显示更清晰

if __name__ == "__main__":
    test_write_zeros_to_output()