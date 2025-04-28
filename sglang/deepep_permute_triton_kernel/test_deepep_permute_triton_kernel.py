import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def deepep_permute_triton_kernel(
    input_ptr,         # input tensor (src_len, hidden_size)
    gateup_input_ptr,  # output tensor (dst_len, hidden_size)
    src2dst_ptr,       # mapping from source to destination indices (src_len, topk)
    topk_ids_ptr,      # top-k expert ids (src_len, topk)
    a1_scales_ptr,     # optional scaling factors (if needed)
    topk: tl.constexpr, 
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    # Get the current source index
    src_idx = tl.program_id(0)

    # Compute pointers for src2dst and topk_ids
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    # Compute pointer for the source data
    src_ptr = input_ptr + src_idx * hidden_size

    # Process the hidden_size dimension in blocks
    for start_offset in range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size

        # Load input data for the current block
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)

        # Iterate over the top-k experts
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)  # Load destination index
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


def deepep_permute_impl(
    input: torch.Tensor,          # (src_len, hidden_size)
    gateup_input: torch.Tensor,   # (dst_len, hidden_size)
    src2dst: torch.Tensor,        # (src_len, topk)
    topk_ids: torch.Tensor,       # (src_len, topk)
    a1_scales: torch.Tensor,      # Optional (src_len,)
    topk: int,
    hidden_size: int,
    BLOCK_SIZE: int = 512,
):
    """
    Perform permutation of input data based on src2dst mapping.

    Args:
        input: Input tensor (src_len, hidden_size).
        gateup_input: Output tensor (dst_len, hidden_size).
        src2dst: Mapping from source to destination indices (src_len, topk).
        topk_ids: Top-k expert ids (src_len, topk).
        a1_scales: Optional scaling factors (src_len,).
        topk: Number of top-k experts.
        hidden_size: Hidden size dimension.
        BLOCK_SIZE: Block size for Triton kernel.
    """
    assert input.shape[1] == hidden_size
    assert gateup_input.shape[1] == hidden_size
    assert src2dst.shape[1] == topk
    assert topk_ids.shape[1] == topk

    grid = lambda meta: (input.shape[0],)

    # Launch the Triton kernel
    deepep_permute_triton_kernel[grid](
        input_ptr=input,
        gateup_input_ptr=gateup_input,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        a1_scales_ptr=a1_scales,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return gateup_input

# 初始化输入张量
src_len = 8
dst_len = 16
hidden_size = 128
topk = 2
BLOCK_SIZE = 64

input = torch.randn(src_len, hidden_size, device="npu", dtype=torch.float16)
gateup_input = torch.zeros(dst_len, hidden_size, device="npu", dtype=torch.float16)
src2dst = torch.randint(0, dst_len, (src_len, topk), device="npu", dtype=torch.int32)
topk_ids = torch.randint(0, 10, (src_len, topk), device="npu", dtype=torch.int32)
a1_scales = torch.rand(src_len, device="npu", dtype=torch.float16)

# 调用函数
output = deepep_permute_impl(
    input=input,
    gateup_input=gateup_input,
    src2dst=src2dst,
    topk_ids=topk_ids,
    a1_scales=a1_scales,
    topk=topk,
    hidden_size=hidden_size,
    BLOCK_SIZE=BLOCK_SIZE,
)

# 打印结果
print("Output Tensor:")
print(output)