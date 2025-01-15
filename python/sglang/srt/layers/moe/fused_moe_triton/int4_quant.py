import torch
import triton 
import triton.language as tl

@triton.jit
def int4_to_fp8_dequant(
        qweights_ptr,  # quantized matrix, K/8 x N
        scales_ptr,  # scales, per channel (N,)
        K: tl.constexpr,
        N: tl.constexpr
):
    #tl.device_print("in_qweights", qweights)
    offs_out = tl.arange(0, K*2)[:, None] + tl.arange(0,N)[None, :]
    qweights = tl.interleave(qweights, qweights)
    qweights = tl.interleave(qweights, qweights)
    qweights = tl.interleave(qweights, qweights).reshape(K*N,8).trans(1,0)

    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
                                tl.arange(0, 4)[:, None]).reshape(8)
    
    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (K*N, 8))
    shifts = tl.reshape(shifts, (1, K*N 8))
    tl.device_print("shifts", shifts)
    # Unpack and reorder: shift out the correct 4-bit value and mask.
    #qweights = (qweights >> shifts) & 0xF
    tl.device_print("out_qweights", qweights)

    return qweights

    #qweights = tl.interleave(qweights, qweights)
    #weights = tl.interleave(qweights, qweights)


    #tl.store(out + offs_out, qweights)
'''
    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (K/8*N, 8))
    shifts = tl.reshape(shifts, (1, K*N))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    qweights = (qweights >> shifts) & 0xF

    # Compute scale offsets and masks.
    scales = tl.broadcast_to(scales, (1, K))

    # Dequantize.
    qweights = qweights  * scales
    out = qweights.to(tl.fp8e4b8).reshape(K/8, N)

    return out
'''

@triton.jit
def int4_to_fp8_dequant_kernel(
        output,
        qweights_ptr,  # quantized matrix, K/8 x N
        scales_ptr,  # scales, per channel (N,)
        K: tl.constexpr,
        N: tl.constexpr
):
    offs_qweights = tl.arange(0, K)[:, None] + tl.arange(0,N)[None, :]
    qweights = tl.load(qweights_ptr + offs_qweights)

    dweights = int4_to_fp8_dequant(qweights, scales, K, N)


qweights = torch.tensor([[1,8,286331153,305419896]], dtype=torch.int32, device='cuda') #(K/8, N) with K=8, N=4
#print(qweights)
scales = torch.tensor([1,2,3,4], dtype=torch.float32, device='cuda') #(N,) N=4

out = torch.zeros(qweights.shape[0]*8, qweights.shape[1], device="cuda")
#out = torch.zeros(8, device="cuda")
print(f"out={out}")
grid = (1,)
int4_to_fp8_dequant_kernel[grid](
                        out,
                        qweights,
                        scales,
                        qweights.shape[0],
                        qweights.shape[1]
)

print(f"out={out}")