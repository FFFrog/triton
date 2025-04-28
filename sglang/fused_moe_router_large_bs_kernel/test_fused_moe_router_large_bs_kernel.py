import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def fused_moe_router_large_bs_kernel(
    a_ptr,  # input (bs, hidden_dim)
    b_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
    bs,
    num_experts: tl.constexpr,
    topk: tl.constexpr,  # only support topk == 1
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # not supported
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
):
    # 1. Get block ID
    pid = tl.program_id(axis=0)

    # 2. Create pointers for the first block of A and B
    # 2.1. Setup a_ptrs with offsets in m and k
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    bs_mask = offs_m < bs
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    a_ptrs = a_ptr + (offs_m * stride_am + offs_k)

    # 2.2. Setup b_ptrs with offsets in k and n.
    #      Note: b matrix is k-major.
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    offs_n = tl.arange(0, BLOCK_SIZE_N)[:, None]
    expert_mask = offs_n < num_experts
    b_ptrs = b_ptr + (offs_n * stride_bn + offs_k)

    # 3. Create an accumulator of float32 of size [BLOCK_SIZE_M, BLOCK_SIZE_N]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_SIZE_K):  # hidden_dim % BLOCK_SIZE_K == 0
        a = tl.load(a_ptrs, mask=bs_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=expert_mask, other=0.0).to(tl.float32).T
        acc += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    # 4. Logit softcap
    logits_scaled = acc / moe_softcapping
    exped = tl.exp(2 * logits_scaled)
    logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping

    # 5. Top-1 selection
    cond = tl.arange(0, BLOCK_SIZE_N)[None, :] < num_experts
    top1 = tl.argmax(tl.where(cond, logits_softcapped, float("-inf")), axis=1)
    top1_v = tl.max(
        tl.where(cond, logits_softcapped, float("-inf")), axis=1, keep_dims=True
    )
    invsumexp = 1.0 / tl.sum(
        tl.where(cond, tl.exp(logits_softcapped - top1_v), 0.0), axis=1
    )

    # 6. Store to output
    offs_topk = pid * topk * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    topk_mask = offs_topk < bs
    tl.store(topk_ids_ptr + offs_topk, top1, mask=topk_mask)
    tl.store(
        topk_weights_ptr + offs_topk,
        invsumexp,
        mask=topk_mask,
    )


def fused_moe_router_large_bs_impl(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    assert num_experts <= BLOCK_SIZE_N
    assert hidden_dim % BLOCK_SIZE_K == 0
    assert topk == 1

    # Allocate output tensors
    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)

    # Compute grid size
    grid = (triton.cdiv(bs, BLOCK_SIZE_M) * triton.cdiv(num_experts, BLOCK_SIZE_N),)

    # Launch kernel
    fused_moe_router_large_bs_kernel[grid](
        a_ptr=x,
        b_ptr=router_weight,
        topk_weights_ptr=topk_weights,
        topk_ids_ptr=topk_ids,
        bs=bs,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        K=hidden_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        stride_am=hidden_dim,
        stride_bn=hidden_dim,
    )

    return topk_weights, topk_ids

# 初始化输入张量
batch_size = 16
hidden_dim = 128
num_experts = 8
topk = 1
moe_softcapping = 1.0
BLOCK_SIZE_M = 4
BLOCK_SIZE_N = 8
BLOCK_SIZE_K = 32

x = torch.randn(batch_size, hidden_dim, device="npu", dtype=torch.float16)
router_weight = torch.randn(num_experts, hidden_dim, device="npu", dtype=torch.float16)

# 调用函数
topk_weights, topk_ids = fused_moe_router_large_bs_impl(
    x,
    router_weight,
    topk,
    moe_softcapping,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
)
