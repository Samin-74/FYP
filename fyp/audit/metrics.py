"""Manipulability audit metrics (plan section 3.5).

Three per-item, per-level quantities:

- **Boundary margin** (computed in fyp.common.generate_semantic_ids): d2 - d1,
  the gap between the two nearest codeword distances in residual space.

- **Sensitivity**: smallest L2 perturbation of the *input embedding* that
  flips the assigned code. Estimated white-box: ascend the margin via its
  gradient w.r.t. the input embedding (the encoder and all upstream levels are
  recomputed each step; argmin selections are piecewise constant, so the
  gradient is exact within the current cell) until the argmin changes.

- **Steerability**: composite score `exp(-sensitivity / tau)` in [0, 1] with
  `tau` = catalogue median sensitivity. Highly steerable items need only a tiny
  perturbation to flip. (Margin distributions are exported alongside; the
  composite is deliberately simple and documented.)
"""

import torch

from fyp.common.generate_semantic_ids import DEVICE, codebook_dist


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
) -> dict:
    """Estimate the smallest input-space L2 perturbation that flips the code.

    Follows the normalized gradient of the level-k margin until the argmin at
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
            x += step_size * g / gn
            delta = x - x0
            dn = delta.norm()
            if dn > norm_cap:
                x.copy_(x0 + norm_cap * delta / dn)
                dn = torch.tensor(norm_cap, device=dn.device)
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
) -> dict:
    """Batched version of :func:`min_flip_perturbation` over many items.

    All items in a batch ascend their own margin gradient simultaneously;
    items converge (flip) or hit the cap at different steps and are frozen
    out of further updates via an active mask.

    Returns dict of (N,) tensors: flipped, perturbation_norm, steps,
    original_id, flipped_id, and final embeddings ``x``.
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
    for i in range(0, X0.shape[0], batch):
        res = _flip_batch(
            model, X0[i : i + batch].to(DEVICE).float(), level,
            max_steps, step_size, norm_cap,
        )
        for k, v in res.items():
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def _flip_batch(
    model, x0, level, max_steps, step_size, norm_cap
) -> dict:
    x = x0.detach().clone().requires_grad_(True)
    orig_id = assigned_id_at_level(model, x0, level)
    n = x0.shape[0]
    active = torch.ones(n, dtype=torch.bool, device=x0.device)
    flipped = torch.zeros(n, dtype=torch.bool, device=x0.device)
    pnorm = torch.full((n,), float(norm_cap), device=x0.device)
    steps_t = torch.full((n,), float(max_steps), device=x0.device)
    flipped_id = orig_id.clone()

    for step in range(max_steps):
        if x.grad is not None:
            x.grad.zero_()
        m = margin_at_level(model, x, level)
        m.sum().backward()
        with torch.no_grad():
            g = x.grad
            gn = g.norm(dim=1, keepdim=True)
            gn = torch.where(gn < 1e-12, torch.ones_like(gn), gn)
            step_dir = step_size * g / gn
            upd = active[:, None] * step_dir
            x += upd
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
                dn = (x - x0).norm(dim=1, keepdim=True)
            new_id = assigned_id_at_level(model, x, level)
            just_flipped = active & (new_id != orig_id)
            flipped |= just_flipped
            pnorm[just_flipped] = dn.squeeze(1)[just_flipped]
            steps_t[just_flipped] = float(step + 1)
            flipped_id[just_flipped] = new_id[just_flipped]
            active &= ~just_flipped
            if not active.any():
                break

    return {
        "flipped": flipped,
        "perturbation_norm": pnorm,
        "steps": steps_t,
        "original_id": orig_id,
        "flipped_id": flipped_id,
        "x": x.detach(),
    }


def steerability(sensitivity: torch.Tensor, tau: float) -> torch.Tensor:
    """Composite steerability score in (0, 1]; 1 = flips with ~zero perturbation.

    Items with sensitivity >= norm_cap (non-flippable within budget) get 0.
    """
    out = torch.exp(-sensitivity / max(tau, 1e-12))
    out = torch.where(torch.isinf(sensitivity), torch.zeros_like(out), out)
    return out
