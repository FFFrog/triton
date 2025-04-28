import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _sparse_fwd_kernel_flash_decode_stage1(  # Double Sparsity's approximate attention
    Q_Label,
    K_Label_Buffer,
    sm_scale,
    Req_to_tokens,  # shape: [B, S]
    B_Seqlen,
    Att_Out,  # shape: [H, B, S] easier for topk
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    att_stride_h,
    att_stride_b,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    logit_cap: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    min_val = -float("inf")
    att_value = tl.full([BLOCK_N], min_val, dtype=tl.float32)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_index = start_n * BLOCK_N
    block_mask = tl.where(block_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q_Label + off_q + start_mark).to(tl.float32)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        offs_buf_k = (
            k_loc[:, None] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[None, :]
        )
        k = tl.load(
            K_Label_Buffer + offs_buf_k,
            mask=offs_n_new[:, None] < cur_batch_end_index,
            other=0.0,
        ).to(tl.float32)

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale

        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)

    att_value = tl.where(offs_n < cur_batch_end_index, att_value, min_val)
    off_o = cur_head * att_stride_h + (cur_batch * att_stride_b + offs_n)
    tl.store(Att_Out + off_o, att_value)


@torch.no_grad()
def sparse_flash_decode_stage1(
    q_label,
    k_label_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q_label.shape[-1], k_label_buffer.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256, 576}

    BLOCK_DMODEL = Lk

    batch, head_num = q_label.shape[0], q_label.shape[1]

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
    kv_group_num = q_label.shape[1] // k_label_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    _sparse_fwd_kernel_flash_decode_stage1[grid](
        q_label,
        k_label_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q_label.stride(0),
        q_label.stride(1),
        k_label_buffer.stride(0),
        k_label_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        kv_group_num,
        BLOCK_DMODEL,
        BLOCK,
        logit_cap,
        num_warps=num_warps,
        num_stages=1,
    )

# 初始化输入张量
batch_size = 4
head_num = 8
seq_len = 128
d_model = 64
block_size = 32
logit_cap = 50.0

# 查询、键和注意力输出张量
q_label = torch.randn(batch_size, head_num, seq_len, d_model, device="npu", dtype=torch.float16)
k_label_buffer = torch.randn(batch_size, head_num, seq_len, d_model, device="npu", dtype=torch.float16)
att_out = torch.zeros(head_num, batch_size, seq_len, device="npu", dtype=torch.float16)

# 请求到标记的映射和序列长度
Req_to_tokens = torch.randint(0, seq_len, (batch_size, seq_len), device="npu", dtype=torch.int32)
B_Seqlen = torch.full((batch_size,), seq_len, device="npu", dtype=torch.int32)

# 缩放因子
sm_scale = 1.0 / (d_model**0.5)

# 调用函数
sparse_flash_decode_stage1(
    q_label,
    k_label_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    seq_len,
    sm_scale,
    logit_cap,
)

# 打印结果
print("Attention Output:")
print(att_out)
