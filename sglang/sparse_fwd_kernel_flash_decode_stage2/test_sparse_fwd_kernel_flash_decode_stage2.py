import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _sparse_fwd_kernel_flash_decode_stage2(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,  # shape: [B, S]
    Topk_token_indices,  # shape: [H, B, k]
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Heavy_token_num,  # NOTE: This can be used as constexpr but we may support dynamic heavy token number in the future
    stride_req_to_tokens_b,
    stride_topk_token_indices_h,
    stride_topk_token_indices_b,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_o_eb,
    stride_mid_o_eh,
    gqa_group_size,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(Heavy_token_num, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(cur_batch_start_index, cur_batch_end_index, BLOCK_N):
        offs_n_new = start_n + offs_n
        topk_token_indices = tl.load(
            Topk_token_indices
            + stride_topk_token_indices_h * cur_head
            + stride_topk_token_indices_b * cur_batch
            + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch + topk_token_indices,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(
            K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))
        v = tl.load(
            V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    need_store = 1
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d
        )
        off_mid_o_logexpsum = (
            cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        )
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))


@torch.no_grad()
def sparse_flash_decode_stage2(
    q,
    k,
    v,
    Req_to_tokens,
    Topk_token_indices,
    heavy_token_num,
    mid_out,
    mid_out_logsumexp,
    block_seq,
    sm_scale,
):
    BLOCK_SEQ = block_seq
    BLOCK_N = 16
    assert BLOCK_SEQ % BLOCK_N == 0
    # Shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    assert heavy_token_num == Topk_token_indices.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]
    grid = (batch, head_num, triton.cdiv(heavy_token_num, BLOCK_SEQ))

    gqa_group_size = q.shape[1] // k.shape[1]

    _sparse_fwd_kernel_flash_decode_stage2[grid](
        q,
        k,
        v,
        sm_scale,
        Req_to_tokens,
        Topk_token_indices,
        mid_out,
        mid_out_logsumexp,
        heavy_token_num,
        Req_to_tokens.stride(0),
        Topk_token_indices.stride(0),
        Topk_token_indices.stride(1),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        gqa_group_size,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        num_stages=2,
    )

# 初始化输入张量
batch_size = 4
head_num = 8
seq_len = 128
d_model = 64
block_seq = 16
heavy_token_num = 32

# 查询、键和值张量
q = torch.randn(batch_size, head_num, seq_len, d_model, device="cuda", dtype=torch.float16)
k = torch.randn(batch_size, head_num, seq_len, d_model, device="cuda", dtype=torch.float16)
v = torch.randn(batch_size, head_num, seq_len, d_model, device="cuda", dtype=torch.float16)

# 请求到标记的映射和 Top-k 标记索引
Req_to_tokens = torch.randint(0, seq_len, (batch_size, seq_len), device="cuda", dtype=torch.int32)
Topk_token_indices = torch.randint(0, heavy_token_num, (head_num, batch_size, heavy_token_num), device="cuda", dtype=torch.int32)

# 中间输出张量
mid_out = torch.zeros(batch_size, head_num, seq_len // block_seq, d_model, device="cuda", dtype=torch.float16)
mid_out_logsumexp = torch.zeros(batch_size, head_num, seq_len // block_seq, device="cuda", dtype=torch.float16)

# 缩放因子
sm_scale = 1.0 / (d_model**0.5)

# 调用函数
sparse_flash_decode_stage2(
    q,
    k,
    v,
    Req_to_tokens,
    Topk_token_indices,
    heavy_token_num,
    mid_out,
    mid_out_logsumexp,
    block_seq,
    sm_scale,
)

