from einops import rearrange, einsum
import math

import torch

class FlashAttnPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        bq = 32
        bk = 32
        batch_size, n_queries, D = Q.shape
        sqrt_d = math.sqrt(D)
        old_Q = Q
        old_K = K
        old_V = V
        Q = rearrange(Q, "batch (bl bq) d -> batch bl bq d", bq=bq)
        K = rearrange(K, "batch (bl bk) d -> batch bl bk d", bk=bk)
        V = rearrange(V, "batch (bl bk) d -> batch bl bk d", bk=bk)

        O = []
        L = []

        for b in range(batch_size):
            ob = []
            lb = []
            for i in range(Q.shape[1]):
                qi = Q[b, i]
                oi = torch.zeros(bq, D, device=Q.device, requires_grad=True)
                li = torch.zeros(bq, device=Q.device, requires_grad=True)
                mi = torch.full((bq,), -torch.inf, device=Q.device, requires_grad=True)

                for j in range(K.shape[1]):
                    kj = K[b, j]
                    vj = V[b, j]
                    sij = einsum(qi, kj, "bq d, bk d -> bq bk") / sqrt_d
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
        O = torch.stack(O)
        L = torch.stack(L)
        ctx.save_for_backward(L, old_Q, old_K, old_V, O)

        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        d = Q.size(-1)

        scale = 1 / (d ** 0.5)
        S = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") * scale

        P = torch.exp(S - L[..., None])
        D = torch.sum(O * grad_output, dim=-1)
        dV = P.transpose(1, 2) @ grad_output
        dP = grad_output @ V.transpose(1, 2)
        dS = P * (dP - D[..., None])
        dQ = dS @ K * scale
        dK = dS.transpose(1, 2) @ Q * scale

        return dQ, dK, dV, None
