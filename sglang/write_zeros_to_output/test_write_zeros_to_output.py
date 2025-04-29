import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional, Dict, Any, List

@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    将指定 token 的输出置零。
    """
    # 创建零矩阵
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)

    # 列偏移
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 将 stride 转换为 Triton tensor
    stride_m = tl.full((1,), stride_cm, dtype=tl.int32)
    stride_n = tl.full((1,), stride_cn, dtype=tl.int32)

    # 指针计算
    c_ptrs = (
        c_ptr
        + stride_m * (offs_token[:, None])
        + stride_n * (offs_cn[None, :])
    )

    # mask 计算
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    # 存储零值
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    MoE 内核，支持 AWQ/GPTQ 量化加速。
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # 获取当前 block 对应的 token id
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # 获取专家 ID
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # 如果专家无效，则直接写零
        write_zeros_to_output(
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
        )
        return

    # 初始化 offsets
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A 矩阵指针
    a_ptrs = a_ptr + (
        (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    )

    # B 矩阵指针
    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    # 加载 scale / zero point
    b_scale_ptrs = (
        b_scale_ptr
        + off_experts * stride_bse
        + offs_bn[None, :] * stride_bsn
        + ((offs_k[:, None]) // group_size) * stride_bsk
    )

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not even_Ks:
            k_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K)
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        # Load A 和 B
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs)

        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other).to(tl.float32)

        if has_zp and use_int4_w4a16:
            b_zp_shifter = (offs_bn[None, :] % 2) * 4
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
        elif has_zp and use_int8_w8a16:
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)

        # Apply quantization
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = (b.to(tl.float32) * b_scale).to(compute_type)

        # Dot product
        accumulator = tl.dot(a, b, acc=accumulator)

        # 移动指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # 权重乘法
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # 写入输出
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    stride_m = tl.full((1,), stride_cm, dtype=tl.int32)
    stride_n = tl.full((1,), stride_cn, dtype=tl.int32)
    c_ptrs = c_ptr + stride_m * (offs_token[:, None]) + stride_n * (offs_cn[None, :])
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
) -> None:
    """
    调用 Triton 实现的 MoE 内核。
    """
    M = sorted_token_ids.shape[0]
    N, K = B.shape[1], B.shape[2]
    EM = triton.cdiv(M, config["BLOCK_SIZE_M"])
    grid = (triton.cdiv(EM, config["GROUP_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]), )

    even_Ks = (K % config["BLOCK_SIZE_K"] == 0)

    fused_moe_kernel_gptq_awq[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        M,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        B_scale.stride(0),
        0,
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        group_size=block_shape[1] if block_shape else 128,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        even_Ks=even_Ks,
        **config,
    )

def main():
    # 配置参数
    config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
    }

    # 输入数据维度
    M = 128   # 总token数
    N = 64    # 每个 token 的输出维度
    K = 128   # 权重矩阵 B 的隐藏层大小
    EM = 4    # 专家数量
    top_k = 2 # top-k 选择的专家数量

    # 创建输入张量
    A = torch.randn(M, K, device="npu", dtype=torch.float16)
    B = torch.randn(EM, K, N, device="npu", dtype=torch.float16)
    C = torch.empty(M, N, device="npu", dtype=torch.float16)

    # Scale 和 Zero Point 张量（假设使用 int8_w8a16）
    B_scale = torch.ones((EM, N), device="npu", dtype=torch.float16) * 0.1
    B_zp = None  # 如果不使用零点，则为 None

    # Top-K 权重和 IDs
    topk_weights = torch.ones(M, device="npu", dtype=torch.float16)
    topk_ids = torch.randint(0, EM, (M,), device="npu")

    # Sorted Token IDs 和 Expert IDs
    sorted_token_ids = torch.arange(M, device="npu")
    expert_ids = torch.randint(-1, EM, (triton.cdiv(M, config["BLOCK_SIZE_M"]),), device="npu")

    # Post-padded Tokens 数量
    num_tokens_post_padded = torch.tensor(M, device="npu")

    # 调用 invoke_fused_moe_kernel 函数
    invoke_fused_moe_kernel(
        A=A,
        B=B,
        C=C,
        A_scale=None,
        B_scale=B_scale,
        B_zp=B_zp,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=top_k,
        config=config,
        compute_type=tl.float16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=[config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]],
        no_combine=False,
    )

    print("Kernel execution completed.")

if __name__ == "__main__":
    main()