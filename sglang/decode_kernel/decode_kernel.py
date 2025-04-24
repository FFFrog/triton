import torch
import torch_npu
import triton
import triton.language as tl

device = torch.device("npu:0")


@triton.jit
def _decode_kernel(
    Q,
    K,
    V,
    KV,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    d_original: tl.constexpr,
    e: tl.constexpr,
    e_original: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    s = tl.load(S + off_h)
    ratio = tl.exp(-s)

    d_idx = tl.arange(0, d)
    e_idx = tl.arange(0, e)

    # Create masks for original dimensions
    d_mask = d_idx < d_original
    e_mask = e_idx < e_original

    # Load with masking
    q = tl.load(Q + qk_offset + d_idx, mask=d_mask, other=0.0)
    k = tl.load(K + qk_offset + d_idx, mask=d_mask, other=0.0)
    v = tl.load(V + v_offset + e_idx, mask=e_mask, other=0.0)

    # Load KV with 2D masking
    kv = tl.load(
        KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
        mask=(d_mask[:, None] & e_mask[None, :]),
        other=0.0,
    )

    # Compute outer product using element-wise operations
    k_v_prod = k[:, None] * v[None, :]
    kv = ratio * kv + k_v_prod

    # Store KV with 2D masking
    tl.store(
        KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
        kv.to(KV.dtype.element_ty),
        mask=(d_mask[:, None] & e_mask[None, :]),
    )

    # Compute matrix-vector multiplication using element-wise operations and reduction
    o = tl.sum(q[:, None] * kv, axis=0)

    # Store output with masking
    tl.store(Out + o_offset + e_idx, o.to(Out.dtype.element_ty), mask=e_mask)


def decode_kernel():
    q_padded = torch.randn([64, 64, 1, 128], dtype=torch.bfloat16, device=device)
    k_padded = torch.randn([64, 64, 1, 128], dtype=torch.bfloat16, device=device)
    v_padded = torch.randn([64, 64, 1, 128], dtype=torch.bfloat16, device=device)
    kv_padded = torch.randn([64, 64, 128, 128], dtype=torch.float32, device=device)
    o_padded = torch.randn([64, 64, 1, 128], dtype=torch.bfloat16, device=device)
    s = torch.randn([64, 1, 1], dtype=torch.float32, device=device)
    b = 64
    h = 64
    n = 1
    d_padded = 128
    d = 96
    e_padded = 128
    e = 96

    grid = (b * h, 1)

    _decode_kernel[grid](
        q_padded,
        k_padded,
        v_padded,
        kv_padded,
        o_padded,
        s,
        b=b,
        h=h,
        n=n,
        d=d_padded,
        d_original=d,
        e=e_padded,
        e_original=e,
    )

    # Get unpadded outputs
    o = o_padded[..., :e]
    kv_out = kv_padded[..., :d, :e]

    return o, kv_out


if __name__ == "__main__":
    decode_kernel()
