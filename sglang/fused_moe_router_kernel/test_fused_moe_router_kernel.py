import torch
import torch_npu
import triton
import triton.language as tl

def is_hip() -> bool:
    return torch.version.hip is not None

_is_hip = is_hip()

@triton.jit
def fused_moe_router_kernel(
    input_ptr,  # input (bs, hidden_dim)
    moe_router_weight_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # not supported
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Load router weights (num_experts, hidden_dim)
    expert_offsets = tl.arange(0, num_experts)[:, None]
    router_mask = mask[None, :]
    w_router = tl.load(
        moe_router_weight_ptr + expert_offsets * hidden_dim + offsets[None, :],
        mask=router_mask,
        other=0.0,
    )

    # Load input vector (hidden_dim,)
    x = tl.load(input_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0)

    # Compute logits: dot product between router weights and input
    logits = tl.sum((w_router.to(tl.float32) * x[None, :].to(tl.float32)), axis=-1)

    # Softcap logits
    logits_scaled = logits / moe_softcapping
    exped = tl.exp(2 * logits_scaled)
    top = exped - 1
    bottom = exped + 1
    logits_softcapped = top / bottom * moe_softcapping

    # Top-1 selection
    top1 = tl.argmax(logits_softcapped, axis=0)
    tl.store(topk_ids_ptr + pid * topk + 0, top1)

    top1_v = tl.max(logits_softcapped, axis=0)
    invsumexp = 1.0 / tl.sum(tl.exp(logits_softcapped - top1_v), axis=0)
    tl.store(topk_weights_ptr + pid * topk + 0, invsumexp)

    # Top-2 selection
    if topk >= 2:
        top2 = tl.argmax(
            tl.where(
                tl.arange(0, num_experts) != top1, logits_softcapped, float("-inf")
            ),
            axis=0,
        )
        tl.store(topk_ids_ptr + pid * topk + 1, top2)
        top2_v = tl.sum(
            logits_softcapped * (tl.arange(0, num_experts) == top2), axis=0
        )
        tl.store(
            topk_weights_ptr + pid * topk + 1,
            tl.exp(top2_v - top1_v) * invsumexp,
        )

    # Top-k selection for k > 2
    if topk > 2:
        topk_mask = tl.full(logits_softcapped.shape, 1.0, dtype=logits_softcapped.dtype)
        topk_mask = tl.where(
            tl.arange(0, num_experts) != top1, topk_mask, float("-inf")
        )
        topk_mask = tl.where(
            tl.arange(0, num_experts) != top2, topk_mask, float("-inf")
        )
        for i in range(2, topk):
            topi = tl.argmax(logits_softcapped + topk_mask, axis=0)
            topk_mask = tl.where(
                tl.arange(0, num_experts) != topi, topk_mask, float("-inf")
            )
            tl.store(topk_ids_ptr + pid * topk + i, topi)
            topi_v = tl.sum(
                logits_softcapped * (tl.arange(0, num_experts) == topi), axis=0
            )
            tl.store(
                topk_weights_ptr + pid * topk + i,
                tl.exp(topi_v - top1_v) * invsumexp,
            )


def fused_moe_router_impl(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    # Allocate output tensors
    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)

    grid = lambda meta: (bs,)

    min_num_warps = 16 if _is_hip else 32

    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), min_num_warps), 4
        ),
    }

    fused_moe_router_kernel[grid](
        x,
        router_weight,
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        hidden_dim=hidden_dim,
        **config,
    )

    return topk_weights, topk_ids

# 初始化输入张量
batch_size = 4
hidden_dim = 64
num_experts = 16
topk = 2
moe_softcapping = 1.0

x = torch.randn(batch_size, hidden_dim, device="npu", dtype=torch.float16)
router_weight = torch.randn(num_experts, hidden_dim, device="npu", dtype=torch.float16)

# 调用函数
topk_weights, topk_ids = fused_moe_router_impl(x, router_weight, topk, moe_softcapping)

