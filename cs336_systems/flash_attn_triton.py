import torch

import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )


    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides = (stride_oq, stride_od),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (query_tile_index * Q_TILE_SIZE, ),
        block_shape = (Q_TILE_SIZE, ),
        order = (0,),
    )


    # Initialize a buffer to write to
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    x = tl.cdiv(N_QUERIES, K_TILE_SIZE)
    if is_causal:
        x = tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
    i = tl.arange(0, Q_TILE_SIZE)[:, None]
    j = tl.arange(0, K_TILE_SIZE)[None, :]
    mask = i < j

    for i in range(x):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s = tl.dot(q, tl.trans(k)) * scale
        if is_causal and i == x-1:
            s += mask * (-1000000)
        m2 = tl.maximum(tl.max(s, axis=1), m)
        p = tl.exp(s - m2[:, None])
        di = tl.exp(m - m2)
        l = di * l + tl.sum(p, axis=1)
        o = di[:, None] * o + tl.dot(p, v)
        m = m2
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    o = o / l[:, None]
    l = m + tl.log(l)

    tl.store(O_block_ptr, o, boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))


class FlashAttnTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_queries, D = Q.shape
        _, n_keys, _ = K.shape
        scale = 1 / (D ** 0.5)

        ctx.Q_TILE_SIZE = 64
        ctx.K_TILE_SIZE = 64
        ctx.D = D
        ctx.is_causal = is_causal
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, n_queries, device=Q.device)

        flash_attn_fwd_kernel[(triton.cdiv(n_queries, ctx.Q_TILE_SIZE), batch_size)](Q, K, V,
                              O, L,
                              Q.stride(-3), Q.stride(-2), Q.stride(-1),
                              K.stride(-3), K.stride(-2), K.stride(-1),
                              V.stride(-3), V.stride(-2), V.stride(-1),
                              O.stride(-3), O.stride(-2), O.stride(-1),
                              L.stride(-2), L.stride(-1),
                              n_queries, n_keys,
                              scale,
                              ctx.D, ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE,
                              ctx.is_causal,
                              )
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError