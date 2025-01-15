import torch
import triton 
import triton.language as tl

@triton.jit
def int4_to_fp8_dequant(
        qweights,  # quantized matrix, K/8 x N
        scales,  # scales, per channel (N,)
        K8: tl.constexpr, #K/8
        N: tl.constexpr
):
    #tl.device_print("in_qweights", qweights)
    qweights = qweights.trans(1,0) #(N,K8)
    qweights = tl.interleave(qweights, qweights) 
    qweights = tl.interleave(qweights, qweights)
    weights = tl.interleave(qweights, qweights).trans(1,0) #(K,N)

    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
                                tl.arange(0, 4)[:, None]).reshape(8)
    
    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in weights
    shifts = reverse_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (K8*N, 8)) #(K8*N,8)
    shifts = tl.reshape(shifts, (N, K8*8)).trans(1,0) #(K,N)
    #tl.device_print("shifts", shifts)

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    weights = (weights >> shifts) & 0xF #(K,N)

    scales = tl.broadcast_to(scales[None, :], (K8*8,N))
    #tl.device_print("scales", scales)
    dweights = weights * scales
    dweights = dweights.to(tl.float8e4b8)
    #tl.device_print("dweights", dweights)

    return dweights


@triton.jit
def int4_to_fp8_dequant_kernel(
        output,
        qweights_ptr,  # quantized matrix, K/8 x N
        scales_ptr,  # scales, per channel (N,)
        stride_qw_k,
        stride_qw_n,
        stride_dw_k,
        stride_dw_n,
        K8: tl.constexpr,
        N: tl.constexpr
):
    offs_qweights = tl.arange(0, K8)[:, None]*stride_qw_k + tl.arange(0,N)[None, :]
    qweights = tl.load(qweights_ptr + offs_qweights)

    offs_scales = tl.arange(0, N)
    scales = tl.load(scales_ptr + offs_scales)

    dweights = int4_to_fp8_dequant(qweights, scales, K8, N)

    offs_dweights = tl.arange(0, K8*8)[:, None]*stride_dw_k + tl.arange(0,N)[None, :]
    tl.store(output + offs_dweights, dweights)


#Big endian
#286331153  = 0x11111111
#68494903   = 0x04152637(wo reorder 0x01234567) 
#2359144127 = 0x8c9daebf(wo reorder 0x89abcdef)
#4226472392 = 0xfbead9c8(wo reorder 0xfedcba98)

#Little endian
#16777216 = 0x01000000 
#33554432 = 0x02000000 
#117440512 = 0x07000000 
#134217728 = 0x08000000 
#925242628  = 0x37261504 (wo reorder 0x76543210)
#3215891852 = 0xbfae9d8c(wo reorder 0xfedcba98)
#3369724667 = 0xc8d9eafb (wo reorder 0x89abcdef)


weights = torch.tensor([[1,8,1,0],
                        [0,0,1,1],
                        [0,0,1,2],
                        [0,0,1,3],
                        [0,0,1,4],
                        [0,0,1,5],
                        [0,0,1,6],
                        [0,0,1,7],
                        [2,7,8,15],
                        [0,0,9,14],
                        [0,0,10,13],
                        [0,0,11,12],
                        [0,0,12,11],
                        [0,0,13,10],
                        [0,0,14,9],
                        [0,0,15,8]])
qweights_big_endian = torch.tensor([[1,8,286331153,68494903],[2,7,2359144127,4226472392] ], dtype=torch.int32, device='cuda') #(K/8, N) with K=8, N=4
qweights_little_endian = torch.tensor([[16777216,134217728,286331153,925242628],[33554432,117440512,3215891852,3369724667] ], dtype=torch.int32, device='cuda') #(K/8, N) with K=8, N=4

qweights = qweights_big_endian
#qweights = qweights_little_endian
print(f"quantized_unpacked_weights_wo_scale={weights}")
print(f"quantized_packed_weights_wo_scale={qweights}")

scales = torch.tensor([1.0,2.5,3.0,4.0], dtype=torch.float32, device='cuda') #(N,) N=4
print(f"scales={scales}")

out = torch.zeros(qweights.shape[0]*8, qweights.shape[1], dtype=torch.float8_e4m3fnuz, device="cuda")
grid = (1,)
int4_to_fp8_dequant_kernel[grid](
                        out,
                        qweights,
                        scales,
                        qweights.stride(0),
                        qweights.stride(1),
                        out.stride(0),
                        out.stride(1),
                        qweights.shape[0],
                        qweights.shape[1]
)

print(f"dequantized_weights={out}")