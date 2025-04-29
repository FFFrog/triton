import torch
import torch_npu
import triton
import triton.language as tl

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
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def call_write_zeros(
    output: torch.Tensor,
    token_mask: torch.Tensor,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
):
    """
    Call the write_zeros_to_output kernel to initialize output tensor with zeros.
    
    Args:
        output: Output tensor to be zeroed (M x N)
        token_mask: Mask indicating which tokens to process (M,)
        BLOCK_SIZE_M: Tile size along M dimension
        BLOCK_SIZE_N: Tile size along N dimension
    """
    assert output.is_cuda, "Output tensor must be on GPU/NPU"
    assert token_mask.is_cuda, "Token mask must be on GPU/NPU"
    
    M, N = output.shape
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    # Convert token_mask to the right format
    offs_token = torch.arange(0, M, device=output.device)
    token_mask = token_mask.to(torch.bool)
    
    write_zeros_to_output[grid](
        output,
        output.stride(0),
        output.stride(1),
        0,  # pid_n will be handled by the grid
        N,
        offs_token,
        token_mask,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        compute_type=output.dtype,  # 使用输出张量的dtype
    )

# Example usage
if __name__ == "__main__":
    # Set device - this will automatically use NPU if available
    device = torch.device('npu')
    
    # Create sample data
    M, N = 1024, 2048
    output = torch.randn(M, N, device=device)
    token_mask = torch.ones(M, device=device, dtype=torch.bool)  # Process all tokens
    
    print("Original output (first 5x5):")
    print(output[:5, :5])
    
    # Call the zeroing function
    call_write_zeros(output, token_mask)
    
    print("\nOutput after zeroing (first 5x5):")
    print(output[:5, :5])
    
    # Verify the result
    assert torch.allclose(output, torch.zeros_like(output)), "Output was not properly zeroed"
    print("\nVerification passed - output is all zeros!")