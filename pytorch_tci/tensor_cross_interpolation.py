import torch
from typing import List, Tuple

def _err_element_3d(A, ps, cs, rs, ss, i, j, k):
    # A[i,j,k] - sum_t (cs[t,i] * rs[t,j] * ss[t,k]) / ps[t]
    if ps.numel() == 0:
        return A[i, j, k]
    w = (cs[:, i] * rs[:, j] * ss[:, k]) / ps.squeeze(-1)  # [t]
    return A[i, j, k] - w.sum()

def _err_fiber_mode1(A, ps, cs, rs, ss, j, k):
    # E(:, j, k) = A(:, j, k) - C^T * w, where w_t = (R[t,j]*S[t,k])/p_t
    vec = A[:, j, k]
    if ps.numel() == 0:
        return vec.clone()
    w = (rs[:, j] * ss[:, k]) / ps.squeeze(-1)            # [t]
    approx = cs.T @ w                                      # [m]
    return vec - approx

def _err_fiber_mode2(A, ps, cs, rs, ss, i, k):
    # E(i, :, k) = A(i, :, k) - R^T * w, where w_t = (C[t,i]*S[t,k])/p_t
    vec = A[i, :, k]
    if ps.numel() == 0:
        return vec.clone()
    w = (cs[:, i] * ss[:, k]) / ps.squeeze(-1)            # [t]
    approx = rs.T @ w                                      # [n]
    return vec - approx

def _err_fiber_mode3(A, ps, cs, rs, ss, i, j):
    # E(i, j, :) = A(i, j, :) - S^T * w, where w_t = (C[t,i]*R[t,j])/p_t
    vec = A[i, j, :]
    if ps.numel() == 0:
        return vec.clone()
    w = (cs[:, i] * rs[:, j]) / ps.squeeze(-1)            # [t]
    approx = ss.T @ w                                      # [p]
    return vec - approx

def _rook_search_3d(A, ps, cs, rs, ss, max_iter: int = 6):
    m, n, p = A.shape
    # random seed pivot
    i = torch.randint(0, m, (1,)).item()
    j = torch.randint(0, n, (1,)).item()
    k = torch.randint(0, p, (1,)).item()

    # one full rook cycle tries all three modes
    for _ in range(max_iter):
        # mode-1 move (optimize i | j,k)
        v1 = _err_fiber_mode1(A, ps, cs, rs, ss, j, k)
        new_i = torch.argmax(v1.abs()).item()

        # mode-2 move (optimize j | i,k)
        v2 = _err_fiber_mode2(A, ps, cs, rs, ss, new_i, k)
        new_j = torch.argmax(v2.abs()).item()

        # mode-3 move (optimize k | i,j)
        v3 = _err_fiber_mode3(A, ps, cs, rs, ss, new_i, new_j)
        new_k = torch.argmax(v3.abs()).item()

        changed = (new_i != i) or (new_j != j) or (new_k != k)
        i, j, k = new_i, new_j, new_k
        if not changed:
            break

    # final fibers and pivot value at the rook point
    c = _err_fiber_mode1(A, ps, cs, rs, ss, j, k)      # [m]
    r = _err_fiber_mode2(A, ps, cs, rs, ss, i, k)      # [n]
    s = _err_fiber_mode3(A, ps, cs, rs, ss, i, j)      # [p]
    pv = _err_element_3d(A, ps, cs, rs, ss, i, j, k)   # scalar tensor
    return i, j, k, pv, c, r, s

def reconstruct_from_cross3d(ps, cs, rs, ss):
    """
    Given ps:[t,1], cs:[t,m], rs:[t,n], ss:[t,p], returns A_tilde:[m,n,p].
    """
    if ps.numel() == 0:
        raise ValueError("No terms to reconstruct from.")
    w = (1.0 / ps.squeeze(-1))                           # [t]
    # einsum: sum_t w[t]*cs[t,:]⊗rs[t,:]⊗ss[t,:]
    A_tilde = torch.einsum('t,tm,tn,tp->mnp', w, cs, rs, ss)
    return A_tilde

def tensor_cross_interpolation_3d(
    tensor: torch.Tensor,
    method: str = "rook",
    error_threshold: float = 1e-3,
):
    """
    Hard-coded 3D tensor cross interpolation (ACA style, CP rank-1 updates).
    - Greedy pivoting via 3D rook search.
    - Never materializes the full residual.
    Returns:
        (I, J, K, (ps, cs, rs, ss))
        where:
          I,J,K: lists of chosen indices per mode,
          ps: [t,1], cs: [t,m], rs: [t,n], ss: [t,p].
    """
    assert tensor.dim() == 3, "Only 3D tensor is supported."
    assert method == "rook", "Only 'rook' is implemented for 3D."

    m, n, p = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    ps = torch.empty((0, 1), device=device, dtype=dtype)   # [t,1]
    cs = torch.empty((0, m), device=device, dtype=dtype)   # [t,m]
    rs = torch.empty((0, n), device=device, dtype=dtype)   # [t,n]
    ss = torch.empty((0, p), device=device, dtype=dtype)   # [t,p]

    I: List[int] = []
    J: List[int] = []
    K: List[int] = []
    seen = set()

    # In the worst case, rank cannot exceed min(n1,n2,n3)
    max_terms = min(m, n, p)

    while len(I) < max_terms:
        with torch.no_grad():
            i_star, j_star, k_star, pv, c, r, s = _rook_search_3d(
                tensor, ps, cs, rs, ss, max_iter=6
            )

        if (i_star, j_star, k_star) in seen:
            # try a new random seed by continuing; if residual is really zero we'll exit on threshold
            continue

        if pv.abs() < error_threshold:
            break

        # accept pivot and update banks
        seen.add((i_star, j_star, k_star))
        I.append(i_star); J.append(j_star); K.append(k_star)

        ps = torch.cat([ps, pv.view(1, 1)], dim=0)        # [t+1,1]
        cs = torch.cat([cs, c.view(1, -1)], dim=0)        # [t+1,m]
        rs = torch.cat([rs, r.view(1, -1)], dim=0)        # [t+1,n]
        ss = torch.cat([ss, s.view(1, -1)], dim=0)        # [t+1,p]

    return I, J, K, (ps, cs, rs, ss)

# --- example usage ---
if __name__ == "__main__":
    torch.manual_seed(0)
    # build a synthetic low-CP-rank 3D tensor: sum_{t=1}^r a_t ∘ b_t ∘ c_t
    m, n, p, r = 80, 70, 60, 10
    A1 = torch.randn(r, m, device='cuda')
    A2 = torch.randn(r, n, device='cuda')
    A3 = torch.randn(r, p, device='cuda')
    w  = torch.randn(r, device='cuda')
    T  = torch.einsum('t,tm,tn,tp->mnp', w, A1, A2, A3)

    I, J, K, (ps, cs, rs, ss) = tensor_cross_interpolation_3d(
        tensor=T, method="rook", error_threshold=1e-6
    )
    T_tilde = reconstruct_from_cross3d(ps, cs, rs, ss)

    rel_err = (T - T_tilde).norm() / T.norm()
    print(f"rank terms selected: {len(I)}")
    print(f"relative error: {rel_err.item():.3e}")
