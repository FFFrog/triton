import torch
import torch_npu
import triton
import triton.language as tl

# 定义 memcpy_triton_kernel
@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    offset_src,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


# 主函数
def run_memcpy_kernel():
    # 定义输入和输出张量
    device = torch.device("npu")
    src_tensor = torch.arange(1024, dtype=torch.int32, device=device)  # 源张量
    dst_tensor = torch.zeros_like(src_tensor)  # 目标张量

    # 定义偏移量和大小
    offset_tensor = torch.tensor([0], dtype=torch.int32, device=device)  # 偏移量
    size_tensor = torch.tensor([1024], dtype=torch.int32, device=device)  # 数据大小

    # 配置参数
    BLOCK_SIZE = 256  # 每个线程块处理的数据大小
    grid_size = triton.cdiv(1024, BLOCK_SIZE)  # 计算网格大小

    # 调用 Triton 内核
    memcpy_triton_kernel[(grid_size,)](
        dst_tensor,  # 目标指针
        src_tensor,  # 源指针
        offset_tensor,  # 偏移量指针
        size_tensor,  # 大小指针
        offset_src=False,  # 是否对源数据应用偏移
        chunk_size=1,  # 块大小倍数
        BLOCK_SIZE=BLOCK_SIZE,  # 每个线程块的大小
    )

    # 打印结果
    print("Source Tensor:")
    print(src_tensor.cpu().numpy())
    print("\nDestination Tensor (after memcpy):")
    print(dst_tensor.cpu().numpy())


if __name__ == "__main__":
    run_memcpy_kernel()