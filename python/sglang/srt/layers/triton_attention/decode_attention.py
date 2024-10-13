"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/token_attention_softmax_and_reducev.py
import triton
import triton.language as tl
import torch


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _fwd_kernel_stage1(
    Q,#[num_seqs/batch,num_q_heads, D=head_sz/head_dim]
    K_Buffer,#[total_tokens,num_kv_heads, D=head_sz/head_dim], # total_tokens=sum of seq_len for each seq
    V_Buffer,#[total_tokens,num_kv_heads, D=head_sz/head_dim], # total_tokens=sum of seq_len for each seq
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    Att_Out, #(num_q_heads, total_tokens)->(num_seqs, num_q_heads, max_len_in_batch/BLOCK, head_sz)
    exp_sums, #(num_seqs, num_q_heads, max_len_in_batch/BLOCK)
    max_logits, #(num_seqs, num_q_heads, max_len_in_batch/BLOCK)
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_attn_logits_h,
    stride_exp_sums_bs,
    stride_exp_sums_h,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    logit_cap: tl.constexpr,    
):
    cur_batch = tl.program_id() #cur_seq
    cur_head = tl.program_id(1) #cur head
    cur_token_blk_in_batch = tl.program_id(2) #token block within seq_len

    reduce_dtype = Att_Out.dtype.element_ty

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(b_req_idx + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)


    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)
    offs_n = cur_token_blk_in_batch*BLOCK_N +  tl.arange(0, BLOCK_N)

    #load q ([1xhead_sz/BLOCK_DMODEL])    
    offs_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q).to_reduce_dtype

    #load k_loc from req_to_tokens ie do logical to physical block translation
    k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n,
        mask=offs_n < cur_batch_end_index,
        other=0
    )
    #offs_k
    offs_buf_k = (
        k_loc[:, None] * stride_buf_kbs
        + cur_kv_head * stride_buf_kh
        + offs_d[None, :]
    )
    #load k [BLOCK_N, BLOCK_DMODEL]
    k = tl.load(
        K_Buffer + offs_buf_k,
        mask=(offs_n[:, None] < cur_batch_end_index) * (offs_d[None, :] < BLOCK_DMODEL),
        other=0.0
    ).to(reduce_dtype)
    
    #calculate qk [BLOCK_N]
    qk = tl.sum((q[None, :] * k), axis=1)
    qk *= sm_scale
    
    if logit_cap > 0:
        qk = logit_cap * tanh(att_value / logit_cap)
    
    #find max
    m = tl.max(qk, 0)

    #calculate p  [BLOCK_N]
    p = tl.exp(qk - m)
    
    #sum p
    exp_sum = tl.sum(p, 0)

    #load v_index from req_to_tokens ie do logical to physical block translation 
    v_loc = tl.load(
        Req_to_tokens + cur_batch_req_idx * stride_req_to_tokens_b
        + offs_n,
        mask=offs_n < cur_batch_seq_len,
        other=0
    )
    #load v [BLOCK_N, BLOCK_DMODEL]
    offs_buf_v = v_log[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_d[None, :]
    v = tl.load(
        V_Buffer + offs_buf_v,
        mask=(off_n[:, None] < cur_batch_seq_len) & (offs_d[None, :] < BLOCK_DMODEL),
        other=0.0
    )
    #Calculate qkv[1, BLOCK_DMODEL]. elementwise multiply p[:, None] * v. 
    qkv = tl.sum(p[:, none] * v, 0)

    att_value = qkv / exp_sum

    #store exp_sum and max_logits
    offs_exp = cur_batch* stride_exp_sums_bs + cur_head * stride_exph + cur_token_blk_in_batch 
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, m)

    #store logits
    off_o = cur_head * stride_attn_logits_h + (cur_batch_start_loc + offs_n)
    tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)

@triton.jit
def _fwd_kernel_stage2(
    out,
    exp_sums,
    max_logits,
    attn_logits,
    stride_obs,
    stride_oh,
    stride_exp_sums_bs,
    stride_exp_sums_h,
    stride_attn_logits_h,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr, 
    NUM_TOKEN_BLKS: tl.constexpr
):

    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    #load max_logits for each blk.
    offs_exp = cur_batch* stride_exp_sums_bs + cur_head * stride_exph 
    max_logits = tl.load(
        max_logits + offs_logit + tl.arange(0, NUM_TOKEN_BLKS),
        other=float("-inf")
    ) #TODO add masking

    #calculate global_max_logit
    max_logit = tl.max(max_logits, axis=0)

    #load exp_sums
    exp_sum = tl.load(
        exp_sums + offs_logits + tl.arange(0, NUM_TOKEN_BLKS),
        other=0.0
    )#TODO add masking

    #rescale each sum and calculate global exp sum
    rescale_factor = tl.exp(max_logits - max_logit)
    rescaled_exp_sum = exp_sum * tl.exp(max_logits - max_logit)
    global_exp_sum = tl.sum(rescaled_exp_sum, axis=0)

    #load attn_logits
    attn_logits_start_ptr  = attn_logits + cur_head * stride_attn_logits_h + (cur_batch_start_loc)
    #rescale logits
    #add all attn_logits
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    for token_blk in tl.range(0, NUM_TOKEN_BLKS):
        offs_n = token_blk*BLOCK_N +  tl.arange(0, BLOCK_N)
        attn_logit = tl.load(attn_logits_start_ptr + off_n, att_value, mask=offs_n < cur_batch_seq_len)
        attn_logit = attn_logit * rescaled_factor[token_blk]
        acc += attn_logit 


    #divide by global_sum
    acc /= (global_exp_sum + 1e-6)

    #write out output
    off_o = cur_batch * stride_obs + cur_head * stride_oh + tl.arange(0, BLOCK_DMODEL_POW2)
    tl.store(out + off_o, acc, mask=(offs_d < BLOCK_DMODEL))

def _decode_reduce_fwd(
    o,
    attn_logits,
    exp_sums,
    max_logits,
    req_to_token,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    BLOCK_N: tl.constexpr,
    NUM_TOKEN_BLKS: tl.constexpr
):
    #BLOCK = 64
    batch, head = b_seq_len.shape[0], logits.shape[0]

    BLOCK_DMODEL = v_buffer.shape[-1]
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(BLOCK_DMODEL)

    grid = (batch, head, 1)
    _fwd_kernel_stage2[grid](
        out,
        exp_sums,
        max_logits,
        o.stride(0),
        o.stride(1),
        exp_sums.stride(0),
        exp_sums.stride(1),
        attn_logits.stride(0),
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        BLOCK_N=BLOCK_N,
        NUM_TOKEN_BLKS=NUM_TOKEN_BLKS
    )

def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    exp_sums,
    max_logits,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
    BLOCK_N,
    NUM_TOKEN_BLKS,
):
    BLOCK_DMODEL = k_buffer.shape[-1] #head_sz/head_dim
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(BLOCK_DMODEL)

    #set up grid (batch, num_q_heads, max_len_in_match/BLOCK)
    num_batch, num_q_heads = B_req_idx.shape[0], q.shape[1]
    #num_token_blks = triton.cdiv(max_len_in_batch, BLOCK_N)

    grid=(num_batch, num_q_heads, NUM_TOKEN_BLKS)

    _fwd_kernel_stage1[grid] (
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        exp_sums,
        max_logits,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        att_out.stride(0),
        exp_sums.stride(0),
        exp_sums.stride(1),
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        BLOCK_N=BLOCK_N,
        logit_cap=logit_cap
    )

@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_buffer,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    att_out,
    exp_sums,
    max_logits,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_attn_logits_h,
    stride_exp_sums_bs,
    stride_exp_sums_h,
    kv_group_num=kv_group_num,
    num_q_heads=num_q_heads,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    logit_cap: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_token_blk_in_batch = tl.program_id(2)

    reduce_dtype= att_out.dtype.element_ty
    cur_q_heads = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)
    offs_n = cur_token_blk_in_batch*BLOCK_N + tl.arange(0, BLOCK_N)

    #load q [BLOCK_H, BLOCK_DMODEL]
    offs_q = cur_batch * stride_qbs + cur_q_heads[:, None] + stride_qh + offs_d[None, :]
    mask_q = cur_q_heads < q_head_num
    q = tl.load(Q + offs_q, mask=(mask_q[:, None]) & (offs_d[None,:] < BLOCK_DMODEL)).to(reduce_dtype)

    #load k_loc from req_to_tokens ie do logical to physical block translation
    k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n,
        mask=offs_n < cur_batch_end_index,
        other=0
    )

    #offs_k
    offs_buf_k = (
        k_loc[None, :] * stride_buf_kbs
        + cur_kv_head * stride_buf_kh
        + offs_d[:, None]
    )
    #load kt [BLOCK_DMODEL, BLOCK_N]
    k = tl.load(
        K_Buffer + offs_buf_k,
        mask=(offs_n[None, :] < cur_batch_end_index) * (offs_d[:, None] < BLOCK_DMODEL),
        other=0.0
    ).to(reduce_dtype)

    #qk using tl.dot[BLOCK_H, BLOCK_N]
    qk = tl.dot(q, k)
    qk *= sm_scale
    if logit_cap > 0:
        qk = logit_cap * tanh(qk/ logit_cap)

    #find max [BLOCK_H]
    m = tl.max(qk, 1)

    #calculate p [BLOCK_H, BLOCK_N]
    p = tl.exp(qk - m)

    #sum p
    exp_sum = tl.sum(p, 1)

    #load v_index from req_to_tokens ie do logical to physical block translation 
    v_loc = tl.load(
        Req_to_tokens + cur_batch_req_idx * stride_req_to_tokens_b
        + offs_n,
        mask=offs_n < cur_batch_seq_len,
        other=0
    )
    #load v [BLOCK_N, BLOCK_DMODEL]
    offs_buf_v = v_log[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_d[None, :]
    v = tl.load(
        V_Buffer + offs_buf_v,
        mask=(off_n[:, None] < cur_batch_seq_len) & (offs_d[None, :] < BLOCK_DMODEL),
        other=0.0
    )

    #calculate qkv and attn [BLOCK_H, BLOCK_DMODEL]
    qkv = tl.dot(p, v)
    attn = qkv / exp_sum

    #store exp_sum and max_logits
    offs_exp = cur_batch* stride_exp_sums_bs + cur_head * stride_exph + cur_token_blk_in_batch 
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, m)

    #store logits
    off_o = cur_head * stride_attn_logits_h + (cur_batch_start_loc + offs_n)
    tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)


@triton.jit
def _fwd_grouped_kernel_stage2(
    out,
    exp_sums,
    max_logits,
    attn_logits,
    stride_obs,
    stride_oh,
    stride_exp_sums_bs,
    stride_exp_sums_h,
    stride_attn_logits_h,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr, 
    NUM_TOKEN_BLKS: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    #load max_logits for each blk.
    offs_exp = cur_batch* stride_exp_sums_bs + cur_head * stride_exph 
    max_logits = tl.load(
        max_logits + offs_logit + tl.arange(0, NUM_TOKEN_BLKS),
        other=float("-inf")
    ) #TODO add masking

    #calculate global_max_logit
    max_logit = tl.max(max_logits, axis=0)

    #load exp_sums
    exp_sum = tl.load(
        exp_sums + offs_logits + tl.arange(0, NUM_TOKEN_BLKS),
        other=0.0
    )#TODO add masking

    #rescale each sum and calculate global exp sum
    rescale_factor = tl.exp(max_logits - max_logit)
    rescaled_exp_sum = exp_sum * tl.exp(max_logits - max_logit)
    global_exp_sum = tl.sum(rescaled_exp_sum, axis=0)

    #load attn_logits
    attn_logits_start_ptr  = attn_logits + cur_head * stride_attn_logits_h + (cur_batch_start_loc)
    #rescale logits
    #add all attn_logits
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    for token_blk in tl.range(0, NUM_TOKEN_BLKS):
        offs_n = token_blk*BLOCK_N +  tl.arange(0, BLOCK_N)
        attn_logit = tl.load(attn_logits_start_ptr + off_n, att_value, mask=offs_n < cur_batch_seq_len)
        attn_logit = attn_logit * rescaled_factor[token_blk]
        acc += attn_logit 


    #divide by global_sum
    acc /= (global_exp_sum + 1e-6)

    #write out output
    off_o = cur_batch * stride_obs + cur_head * stride_oh + tl.arange(0, BLOCK_DMODEL_POW2)
    tl.store(out + off_o, acc, mask=(offs_d < BLOCK_DMODEL))

def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    exp_sums,
    max_logits,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
    BLOCK_N,
    NUM_TOKEN_BLKS,
):
    BLOCK_DMODEL = k_buffer.shape[-1] #head_sz/head_dim
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(BLOCK_DMODEL)

    #set up grid (batch, num_q_heads, max_len_in_match/BLOCK)
    num_batch, num_q_heads = B_req_idx.shape[0], q.shape[1]
    #num_token_blks = triton.cdiv(max_len_in_batch, BLOCK_N)

    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = max(16, triton.next_power_of_2(kv_group_num))
    grid = (
        num_batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_TOKEN_BLKS,
    )

    grid=(num_batch, num_q_heads, NUM_TOKEN_BLKS)

    _fwd_grouped_kernel_stage1[grid] (
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        exp_sums,
        max_logits,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        att_out.stride(0),
        exp_sums.stride(0),
        exp_sums.stride(1),
        kv_group_num=kv_group_num,
        num_q_heads=num_q_heads,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        logit_cap=logit_cap
    )

def _decode_grouped_reduce_fwd(
    o,
    attn_logits,
    exp_sums,
    max_logits,
    req_to_token,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    BLOCK_N,
    NUM_TOKEN_BLKS
):
    batch, head = b_seq_len.shape[0], logits.shape[0]

    BLOCK_DMODEL = v_buffer.shape[-1]
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(BLOCK_DMODEL)

    grid = (batch, head, 1)
    _fwd_kernel_stage2[grid](
        out,
        exp_sums,
        max_logits,
        o.stride(0),
        o.stride(1),
        exp_sums.stride(0),
        exp_sums.stride(1),
        attn_logits.stride(0),
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        BLOCK_N=BLOCK_N,
        NUM_TOKEN_BLKS=NUM_TOKEN_BLKS
    )


def decode_attention_fwd(
    q, #[num_seqs/batch,num_q_heads, D=head_sz/head_dim]
    k_buffer, #[total_tokens,num_kv_heads, D=head_sz/head_dim], # total_tokens=sum of seq_len for each seq
    v_buffer, #[total_tokens,num_kv_heads, D=head_sz/head_dim] 
    o, #[, num_q_heads, D=head_sz/head_dim]
    req_to_token, #[num_seq/batch, seq_len for each seq]
    b_req_idx, #[num_seq], 
    b_start_loc, #[num_seq], start token location in k/v_buffer for this seq/batch
    b_seq_len, #[num_seq], seq length for this batch/seq
    attn_logits, # #[num_head, total_num_tokens]
    max_len_in_batch,
    sm_scale,
    logit_cap=0.0,
):
    kv_group_num = q.shape[1] // v_buffer.shape[1]

    BLOCK_N = 32
    num_batch, num_q_heads = b_req_idx.shape[0], q.shape[1]
    NUM_TOKEN_BLKS = triton.cdiv(max_len_in_batch, BLOCK_N)
    exp_sums = torch.empty((num_batch, num_q_heads, NUM_TOKEN_BLKS))
    max_logits = torch.empty((num_batch, num_q_heads, NUM_TOKEN_BLKS))

    if kv_group_num == 1:
        # MHA
        _decode_att_m_fwd(
            q,
            k_buffer,
            v_buffer,
            attn_logits,
            exp_sums,
            max_logits,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            sm_scale,
            logit_cap,
            BLOCK_N,
            NUM_TOKEN_BLKS, 
        )
        _decode_reduce_fwd(
            o,
            attn_logits,
            exp_sums,
            max_logits,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            BLOCK_N,
            NUM_TOKEN_BLKS
        )
    # GQA
    else:
        _decode_grouped_att_m_fwd(
            q,
            k_buffer,
            v_buffer,
            attn_logits,
            exp_sums,
            max_logits,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            sm_scale,
            logit_cap,
            BLOCK_N,
            NUM_TOKEN_BLKS,           
        )
        _decode_grouped_reduce_fwd(
            o,
            attn_logits,
            exp_sums,
            max_logits,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            BLOCK_N,
            NUM_TOKEN_BLKS
        )

