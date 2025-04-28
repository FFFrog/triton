import torch
import torch_npu
import triton
import triton.language as tl

# 定义 create_flashmla_kv_indices_triton 内核
@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
):
    PAGED_SIZE: tl.constexpr = 64
    BLOCK_SIZE: tl.constexpr = 4096
    NUM_PAGE_PER_BLOCK: tl.constexpr = 64
    pid = tl.program_id(axis=0)

    # 找到请求池索引
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)

    for i in range(num_pages_loop):
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset <= num_paged * PAGED_SIZE
        mask_out = paged_offset_out <= num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )


# 主函数
def run_create_flashmla_kv_indices():
    # 定义输入和输出张量
    device = torch.device("npu")  # 确保有 GPU 设备
    max_batch = 8
    max_context_len = 1024
    num_requests = 4

    # 初始化输入张量
    req_to_token_tensor = torch.randint(
        0, 10000, (max_batch, max_context_len), dtype=torch.int32, device=device
    )  # 请求到标记的映射
    req_pool_indices_tensor = torch.randint(
        0, max_batch, (num_requests,), dtype=torch.int32, device=device
    )  # 请求池索引
    page_kernel_lens_tensor = torch.randint(
        1, 512, (num_requests,), dtype=torch.int32, device=device
    )  # 每个请求的页面长度
    kv_start_idx_tensor = torch.randint(
        0, 512, (num_requests,), dtype=torch.int32, device=device
    )  # KV 开始索引

    # 输出张量
    kv_indices_tensor = torch.zeros(
        (num_requests, max_context_len // 64), dtype=torch.int32, device=device
    )

    # 配置参数
    grid_size = num_requests  # 每个请求对应一个线程块
    req_to_token_ptr_stride = req_to_token_tensor.stride(0)
    kv_indices_ptr_stride = kv_indices_tensor.stride(0)

    # 调用 Triton 内核
    create_flashmla_kv_indices_triton[(grid_size,)](
        req_to_token_tensor,  # 请求到标记的映射
        req_pool_indices_tensor,  # 请求池索引
        page_kernel_lens_tensor,  # 页面长度
        kv_start_idx_tensor,  # KV 开始索引
        kv_indices_tensor,  # 输出 KV 索引
        req_to_token_ptr_stride=req_to_token_ptr_stride,  # 请求到标记的步幅
        kv_indices_ptr_stride=kv_indices_ptr_stride,  # KV 索引的步幅
    )

    # 打印结果
    print("Request to Token Tensor:")
    print(req_to_token_tensor.cpu().numpy())
    print("\nKV Indices Tensor (after kernel):")
    print(kv_indices_tensor.cpu().numpy())


if __name__ == "__main__":
    run_create_flashmla_kv_indices()