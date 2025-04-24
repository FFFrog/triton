import torch
import torch_npu
import triton
import triton.language as tl

device = torch.device("npu:0")


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # 计算每个序列的偏移量
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    # 加载序列长度，并处理越界情况
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    min_seq_len = tl.min(seq_lens)

    # 如果最大序列长度与最小序列长度的比例满足条件，则调整最小序列长度
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len

    # 计算第一种分割方式的最大分割数和块大小
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # 基于设备核心数计算第二种分割方式的最大分割数和块大小
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head

    # 根据注意力头分组情况调整 token 网格大小
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)

    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    # 计算最终的 KV 分割数
    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    # 将结果写回全局内存
    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(
            num_kv_splits_ptr + i + offs_token,
            num_kv_splits,
            mask=mask_token,
        )


def test_get_num_kv_splits_triton():
    # 定义输入参数
    num_seq = 8
    num_group = 4
    num_head = 16
    num_kv_head = 4
    max_kv_splits = 16
    device_core_count = 128
    MAX_NUM_SEQ = 16  # 设置为 2 的幂次方以优化性能

    # 初始化输入张量
    seq_lens = torch.randint(1, 128, (num_seq,), dtype=torch.int32, device=device)
    num_kv_splits_ptr = torch.empty((num_seq * num_group,), dtype=torch.int32, device=device)

    # 调用 Triton 内核
    grid = (1,)
    get_num_kv_splits_triton[grid](
        num_kv_splits_ptr,
        seq_lens,
        num_seq,
        num_group,
        num_head,
        num_kv_head,
        max_kv_splits,
        device_core_count,
        MAX_NUM_SEQ=MAX_NUM_SEQ,
    )


if __name__ == "__main__":
    test_get_num_kv_splits_triton()
