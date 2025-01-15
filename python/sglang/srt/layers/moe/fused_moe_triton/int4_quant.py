import torch
import triton 
import triton.language as tl

@triton.jit
def int4_to_fp8_dequant(
        qweights,  # quantized matrix, K/8 x N
        scales,  # scales, per channel (N,)
        K: tl.constexpr,
        N: tl.constexpr
):
    tl.device_print("in_qweights", qweights)
    qweights = qweights.trans(1,0)
    qweights = tl.interleave(qweights, qweights)
    qweights = tl.interleave(qweights, qweights)
    weights = tl.interleave(qweights, qweights).trans(1,0)

    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
                                tl.arange(0, 4)[:, None]).reshape(8)
    
    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_order_tensor * 4
    #shifts = tl.join(shifts,shifts)
    shifts = tl.broadcast_to(shifts[None, :], (K*N, 8))
    #shifts = tl.broadcast_to(shifts, (8*K, 1))
    #shifts = tl.broadcast_to(shifts, (8*K, N))
    shifts = tl.reshape(shifts, (N, K*8)).trans(1,0)
    tl.device_print("shifts", shifts)
    # Unpack and reorder: shift out the correct 4-bit value and mask.
    weights = (weights >> shifts) & 0xF
    tl.device_print("out_weights", weights)

    return weights

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
        stride_qw_k,
        stride_qw_n,
        K: tl.constexpr,
        N: tl.constexpr
):
    offs_qweights = tl.arange(0, K)[:, None]*stride_qw_k + tl.arange(0,N)[None, :]
    qweights = tl.load(qweights_ptr + offs_qweights)

    dweights = int4_to_fp8_dequant(qweights, scales_ptr, K, N)

#286331153 = 0x11111111 and 19088743=0x01234567
#2309737967 =0x89abcdef and 4275878552 = 0xfedcba98 
qweights = torch.tensor([[1,8,286331153,19088743],[2,7,2309737967,4275878552] ], dtype=torch.int32, device='cuda') #(K/8, N) with K=8, N=4
#print(qweights)
scales = torch.tensor([1,2,3,4], dtype=torch.float32, device='cuda') #(N,) N=4

out = torch.zeros(qweights.shape[0]*8, qweights.shape[1], device="cuda")
#out = torch.zeros(8, device="cuda")
#print(f"out={out}")
grid = (1,)
int4_to_fp8_dequant_kernel[grid](
                        out,
                        qweights,
                        scales,
                        qweights.stride(0),
                        qweights.stride(1),
                        qweights.shape[0],
                        qweights.shape[1]
)

#print(f"out={out}")