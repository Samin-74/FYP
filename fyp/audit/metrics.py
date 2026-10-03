"""Manipulability audit metrics (plan section 3.5).

Three per-item, per-level quantities:

- **Boundary margin** (computed in fyp.common.generate_semantic_ids): d2 - d1,
  the gap between the two nearest codeword distances in residual space.

- **Sensitivity**: smallest L2 perturbation of the *input embedding* that
  flips the assigned code. Estimated white-box: *descend* the margin
  d2 - d1 via its gradient w.r.t. the input embedding (the encoder and all
  upstream levels are recomputed each step; argmin selections are piecewise
  constant, so the gradient is exact within the current cell) until the argmin
  changes. (An earlier version ascended the margin, i.e. moved items *away*
  from their nearest boundary; numbers produced before that fix are invalid.)

- **Steerability**: composite score `exp(-sensitivity / tau)` in [0, 1] with
  `tau` = catalogue median sensitivity. Highly steerable items need only a tiny
  perturbation to flip. (Margin distributions are exported alongside; the
  composite is deliberately simple and documented.)

Perturbation budget: each step moves ||delta|| by at most ``step_size``, so the
largest reachable perturbation is ``min(norm_cap, max_steps * step_size)``
(= 10 with the defaults), see :func:`effective_budget`. The perturbed embedding
is kept on the original norm sphere (``keep_norm``) and, optionally, within a
cosine budget (``cosine_threshold``) after every step.
"""

import torch

from fyp.attack.constraints import project_to_cosine, project_to_norm
from fyp.common.generate_semantic_ids import DEVICE, codebook_dist


def effective_budget(max_steps: int, step_size: float, norm_cap: float) -> float:
    """Largest ||delta|| the flip search can actually reach."""
    return min(norm_cap, max_steps * step_size)


def _constrain(x0, x, cosine_threshold, keep_norm):
    if cosine_threshold is not None:
        x = project_to_cosine(x0, x, cosine_threshold)
    if keep_norm:
        x = project_to_norm(x0, x)
    return x


def margin_at_level(model, x, level: int) -> torch.Tensor:
    """Differentiable boundary margin d2 - d1 at the given level, for input x."""
    r = model.encode(x)
    for k, layer in enumerate(model.layers):
        d = codebook_dist(layer, r)
        if k == level:
            top2 = torch.topk(d, k=2, dim=1, largest=False)
            return top2.values[:, 1] - top2.values[:, 0]
        ids = d.detach().argmin(axis=1)
        r = r - layer.get_item_embeddings(ids)
    raise ValueError(f"level {level} out of range")


def assigned_id_at_level(model, x, level: int) -> torch.Tensor:
    """Current code index at the given level (no grad)."""
    with torch.no_grad():
        r = model.encode(x)
        out = None
        for k, layer in enumerate(model.layers):
            d = codebook_dist(layer, r)
            ids = d.argmin(axis=1)
            if k == level:
                out = ids
            r = r - layer.get_item_embeddings(ids)
    return out


def min_flip_perturbation(
    model,
    x0: torch.Tensor,
    level: int = 0,
    max_steps: int = 200,
    step_size: float = 0.05,
    norm_cap: float = 50.0,
    cosine_threshold: float | None = None,
    keep_norm: bool = True,
) -> dict:
    """Estimate the smallest input-space L2 perturbation that flips the code.

    Follows the negative normalized gradient of the level-k margin until the argmin at
    that level changes, or the norm cap / step budget is exhausted.

    Returns dict with keys: flipped (bool), perturbation_norm, steps,
    original_id, flipped_id, level.
    """
    model.eval()
    x = x0.detach().clone().requires_grad_(True)
    orig_id = assigned_id_at_level(model, x0, level)

    for step in range(max_steps):
        if x.grad is not None:
            x.grad.zero_()
        m = margin_at_level(model, x, level).sum()
        m.backward()
        with torch.no_grad():
            g = x.grad
            gn = g.norm()
            if gn < 1e-12 or torch.isnan(gn):
                break
            x -= step_size * g / gn
            delta = x - x0
            dn = delta.norm()
            if dn > norm_cap:
                x.copy_(x0 + norm_cap * delta / dn)
            x.copy_(_constrain(x0, x, cosine_threshold, keep_norm))
            dn = (x - x0).norm()
            new_id = assigned_id_at_level(model, x, level)
            if bool((new_id != orig_id).item()):
                return {
                    "flipped": True,
                    "perturbation_norm": float(dn),
                    "steps": step + 1,
                    "original_id": int(orig_id),
                    "flipped_id": int(new_id),
                    "level": level,
                }

    delta = (x - x0).norm()
    return {
        "flipped": False,
        "perturbation_norm": float(delta.detach()),
        "steps": max_steps,
        "original_id": int(orig_id),
        "flipped_id": int(orig_id),
        "level": level,
    }


def min_flip_perturbation_batch(
    model,
    X0: torch.Tensor,
    level: int = 0,
    max_steps: int = 200,
    step_size: float = 0.05,
    norm_cap: float = 50.0,
    batch: int = 512,
    cosine_threshold: float | None = None,
    keep_norm: bool = True,
    record_pre: bool = False,
) -> dict:
    """Batched version of :func:`min_flip_perturbation` over many items.

    All items in a batch descend their own margin gradient simultaneously;
    items converge (flip) or hit the cap at different steps and are frozen
    out of further updates via an active mask.

    Returns dict of (N,) tensors: flipped, perturbation_norm, steps,
    original_id, flipped_id, and final embeddings ``x``. With
    ``record_pre=True`` also returns ``x_pre``: for flipped items the last
    iterate whose code still equalled the original (the input for
    :func:`bisect_flip_distance`).
    """
    model.eval()
    out = {
        "flipped": [],
        "perturbation_norm": [],
        "steps": [],
        "original_id": [],
        "flipped_id": [],
        "x": [],
    }
    if record_pre:
        out["x_pre"] = []
    for i in range(0, X0.shape[0], batch):
        res = _flip_batch(
            model, X0[i : i + batch].to(DEVICE).float(), level,
            max_steps, step_size, norm_cap, cosine_threshold, keep_norm,
            record_pre,
        )
        for k, v in res.items():
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def _flip_batch(
    model, x0, level, max_steps, step_size, norm_cap, cosine_threshold, keep_norm,
    record_pre: bool = False,
) -> dict:
    x = x0.detach().clone().requires_grad_(True)
    orig_id = assigned_id_at_level(model, x0, level)
    n = x0.shape[0]
    active = torch.ones(n, dtype=torch.bool, device=x0.device)
    flipped = torch.zeros(n, dtype=torch.bool, device=x0.device)
    pnorm = torch.full((n,), float(norm_cap), device=x0.device)
    steps_t = torch.full((n,), float(max_steps), device=x0.device)
    flipped_id = orig_id.clone()
    pre_flip = x0.detach().clone() if record_pre else None

    for step in range(max_steps):
        if x.grad is not None:
            x.grad.zero_()
        m = margin_at_level(model, x, level)
        m.sum().backward()
        with torch.no_grad():
            if record_pre:
                x_before = x.detach().clone()
            g = x.grad
            gn = g.norm(dim=1, keepdim=True)
            gn = torch.where(gn < 1e-12, torch.ones_like(gn), gn)
            step_dir = step_size * g / gn
            upd = active[:, None] * step_dir
            x -= upd
            delta = x - x0
            dn = delta.norm(dim=1, keepdim=True)
            over = (dn.squeeze(1) > norm_cap) & active
            if over.any():
                scale = torch.where(
                    over[:, None],
                    norm_cap / torch.clamp(dn, min=1e-12),
                    torch.ones_like(dn),
                )
                x.copy_(x0 + delta * scale)
            # frozen rows have delta already inside the constraints, so the
            # projections leave them unchanged
            x.copy_(_constrain(x0, x, cosine_threshold, keep_norm))
            dn = (x - x0).norm(dim=1, keepdim=True)
            new_id = assigned_id_at_level(model, x, level)
            just_flipped = active & (new_id != orig_id)
            if record_pre:
                pre_flip[just_flipped] = x_before[just_flipped]
            flipped |= just_flipped
            pnorm[just_flipped] = dn.squeeze(1)[just_flipped]
            steps_t[just_flipped] = float(step + 1)
            flipped_id[just_flipped] = new_id[just_flipped]
            active &= ~just_flipped
            if not active.any():
                break

    out = {
        "flipped": flipped,
        "perturbation_norm": pnorm,
        "steps": steps_t,
        "original_id": orig_id,
        "flipped_id": flipped_id,
        "x": x.detach(),
    }
    if record_pre:
        out["x_pre"] = pre_flip
    return out


@torch.no_grad()
def bisect_flip_distance(
    model,
    x0: torch.Tensor,
    x_pre: torch.Tensor,
    x_post: torch.Tensor,
    orig_id: torch.Tensor,
    level: int = 0,
    n_bisect: int = 25,
    keep_norm: bool = True,
    batch: int = 2048,
) -> torch.Tensor:
    """Measured flip distance: bisect the segment on which each item flipped.

    ``x_pre`` is the last search iterate whose level code still equals the
    original and ``x_post`` the first that differs (both from
    :func:`min_flip_perturbation_batch` with ``record_pre=True``). Bisection
    on that segment (midpoints re-projected to the original norm) brackets the
    flip to ~||x_post - x_pre|| / 2**n_bisect. The returned distance is the L2
    norm of the smallest bracketed *flipping* point found, so it is a measured
    flip distance rather than a step-size artefact. It remains an upper bound
    on the true minimum (the descent direction is heuristic), and it is the
    distance along the search path, not the geodesic to the nearest boundary.
    """
    dists = torch.full((x0.shape[0],), float("inf"))
    for i in range(0, x0.shape[0], batch):
        xb0 = x0[i : i + batch].to(DEVICE).float()
        lo = x_pre[i : i + batch].to(DEVICE).float()
        hi = x_post[i : i + batch].to(DEVICE).float()
        orig = orig_id[i : i + batch].to(DEVICE)
        for _ in range(n_bisect):
            mid = (lo + hi) / 2
            if keep_norm:
                mid = project_to_norm(xb0, mid)
            flips = (assigned_id_at_level(model, mid, level) != orig)[:, None]
            hi = torch.where(flips, mid, hi)
            lo = torch.where(flips, lo, mid)
        dists[i : i + batch] = (hi - xb0).norm(dim=1).cpu()
    return dists


def steerability(sensitivity: torch.Tensor, tau: float) -> torch.Tensor:
    """Composite steerability score in (0, 1]; 1 = flips with ~zero perturbation.

    Items with sensitivity >= norm_cap (non-flippable within budget) get 0.
    """
    out = torch.exp(-sensitivity / max(tau, 1e-12))
    out = torch.where(torch.isinf(sensitivity), torch.zeros_like(out), out)
    return out
