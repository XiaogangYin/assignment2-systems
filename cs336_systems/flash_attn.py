from einops import rearrange, einsum
import math

import torch

class FlashAttnPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        bq = 64
        bk = 64
        i = torch.arange(0, bq)[:, None]
        j = torch.arange(0, bk)[None, :]
        mask = i < j

        n_queries, D = Q.shape[-2:]
        sqrt_d = math.sqrt(D)
        old_Q = Q
        old_K = K
        old_V = V
        Q = rearrange(Q, "... (bl bq) d -> (...) bl bq d", bq=bq)
        K = rearrange(K, "... (bl bk) d -> (...) bl bk d", bk=bk)
        V = rearrange(V, "... (bl bk) d -> (...) bl bk d", bk=bk)

        O = []
        L = []

        for b in range(Q.shape[0]):
            ob = []
            lb = []
            for i in range(Q.shape[1]):
                qi = Q[b, i]
                oi = torch.zeros(bq, D, device=Q.device, requires_grad=True)
                li = torch.zeros(bq, device=Q.device, requires_grad=True)
                mi = torch.full((bq,), -torch.inf, device=Q.device, requires_grad=True)

                x = K.shape[1]
                if is_causal:
                    x = math.ceil((i + 1) * bq / bk)

                for j in range(x):
                    kj = K[b, j]
                    vj = V[b, j]
                    sij = einsum(qi, kj, "bq d, bk d -> bq bk") / sqrt_d
                    if is_causal and j == x - 1:
                        sij += mask * (-1000_000)
                    mij = torch.max(mi, torch.max(sij, dim=1).values)
                    pij = torch.exp(sij - mij[:, None])
                    li = torch.exp(mi - mij) * li + torch.sum(pij, dim=1)
                    oi = torch.exp(mi - mij)[:, None] * oi + pij @ vj
                    mi = mij
                oi = oi / li[:, None]
                li = mi + torch.log(li)
                ob.append(oi)
                lb.append(li)
            ob = torch.cat(ob)
            lb = torch.cat(lb)
            O.append(ob)
            L.append(lb)
        O = torch.stack(O).view(old_Q.shape)
        L = torch.stack(L).view(old_Q.shape[:-1])
        ctx.save_for_backward(L, old_Q, old_K, old_V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        d = Q.size(-1)
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]

        scale = 1 / (d ** 0.5)
        S = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") * scale
        if ctx.is_causal:
            S = torch.where(
                torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[
                    None, None, :],
                S,
                -1e6
            )

        P = torch.exp(S - L[..., None])
        D = torch.sum(O * grad_output, dim=-1)
        dV = P.transpose(1, 2) @ grad_output
        dP = grad_output @ V.transpose(1, 2)
        dS = P * (dP - D[..., None])
        dQ = dS @ K * scale
        dK = dS.transpose(1, 2) @ Q * scale

        return dQ, dK, dV, None
