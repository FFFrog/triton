import torch
import torch_npu
import triton
import triton.language as tl

# 定义 fused_softcap_kernel
@triton.jit
def fused_softcap_kernel(
    full_logits_ptr,
    softcapping_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(full_logits_ptr + offsets, mask=mask)

    # Perform operations in-place
    x = x / softcapping_value

    # Manual tanh implementation using exp
    exp2x = tl.exp(2 * x)
    x = (exp2x - 1) / (exp2x + 1)

    x = x * softcapping_value

    # Store result
    tl.store(full_logits_ptr + offsets, x, mask=mask)


# 主函数
def run_fused_softcap_kernel():
    # 定义输入张量和参数
    device = torch.device("npu")  # 确保有 GPU 设备
    n_elements = 1024  # 数据元素数量
    softcapping_value = 5.0  # Softcapping 值

    # 初始化 logits 张量
    full_logits_tensor = torch.randn(n_elements, dtype=torch.float32, device=device)

    # 打印初始 logits
    print("Initial Logits:")
    print(full_logits_tensor.cpu().numpy())

    # 配置参数
    BLOCK_SIZE = 256  # 每个线程块处理的数据大小
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)  # 计算网格大小

    # 调用 Triton 内核
    fused_softcap_kernel[(grid_size,)](
        full_logits_tensor,  # 输入/输出 logits 指针
        softcapping_value,  # Softcapping 值
        n_elements,  # 元素总数
        BLOCK_SIZE=BLOCK_SIZE,  # 每个线程块的大小
    )

    # 打印结果
    print("\nSoftcapped Logits (after kernel):")
    print(full_logits_tensor.cpu().numpy())


if __name__ == "__main__":
    run_fused_softcap_kernel()