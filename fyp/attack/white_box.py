"""White-box targeted steering attack, embedding space (plan section 3.6).

The attacker knows the RQ-VAE and Sentence-T5 encoder (white box) and perturbs
the item's input embedding to force a *target* code at a chosen level — e.g.
collide with a bestseller's semantic ID, or land on a high-traffic level-1
prefix. Text realisation of the perturbation (HotFlip / LLM rewrite ranked by
embedding progress) reuses this optimiser unchanged.

Objective per step: descend the distance from the level-k residual to the
target codeword, `dist(r_k, c_target)`, and stop as soon as the target wins the
argmin. A semantic-preservation constraint (cosine threshold) is *enforced
inside* the loop: after each step the perturbation is projected back onto the
constraint boundary, so successful attacks are guaranteed to satisfy it.
"""

import torch

from fyp.common.generate_semantic_ids import codebook_dist


def _project_to_cosine(x0: torch.Tensor, x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Shrink ``delta = x - x0`` so that cos(x0, x) >= threshold (bisection)."""
    delta = x - x0
    lo = torch.zeros(x.shape[0], device=x.device)
    hi = torch.ones(x.shape[0], device=x.device)
    for _ in range(25):
        mid = (lo + hi) / 2
        xm = x0 + mid[:, None] * delta
        cos = torch.nn.functional.cosine_similarity(x0, xm, dim=-1)
        better = cos >= threshold
        lo = torch.where(better, mid, lo)
        hi = torch.where(better, mid, hi)
    return x0 + lo[:, None] * delta


def _residual_to_level(model, x, level: int):
    """Residual r_k at `level`, differentiable w.r.t. x."""
    r = model.encode(x)
    for k, layer in enumerate(model.layers):
        if k == level:
            return r, layer
        ids = codebook_dist(layer, r).detach().argmin(axis=1)
        r = r - layer.get_item_embeddings(ids)
    raise ValueError(f"level {level} out of range")


def targeted_flip(
    model,
    x0: torch.Tensor,
    target_code: int,
    level: int = 0,
    cosine_threshold: float | None = 0.95,
    constraint=None,
    max_steps: int = 300,
    step_size: float = 0.05,
) -> dict:
    """Steer x0 so that level-`level` argmin becomes `target_code`.

    ``cosine_threshold`` (default 0.95) is enforced by projection after every
    step. ``constraint`` (fyp.attack.constraints) is additionally *checked* at
    the end for reporting compatibility.

    Returns dict: success, perturbation_norm, cosine_to_original, steps,
    original_id, final_id, level, target_code, constraint_ok.
    """
    model.eval()
    x0 = x0.detach()
    x = x0.clone().requires_grad_(True)

    with torch.no_grad():
        orig_id = codebook_dist(
            model.layers[level], model.encode(x0)
        ).argmin(axis=1)

    new_id = orig_id
    for step in range(max_steps):
        if x.grad is not None:
            x.grad.zero_()
        r, layer = _residual_to_level(model, x, level)
        d_target = codebook_dist(layer, r)[:, target_code].sum()
        d_target.backward()
        with torch.no_grad():
            g = x.grad
            gn = g.norm(dim=1, keepdim=True)
            gn = torch.where(gn < 1e-12, torch.ones_like(gn), gn)
            x -= step_size * g / gn
            if cosine_threshold is not None:
                x.copy_(_project_to_cosine(x0, x, cosine_threshold))
            new_id = codebook_dist(
                model.layers[level], model.encode(x)
            ).argmin(axis=1)
            if bool((new_id == target_code).all()):
                break

    delta = (x - x0).norm(dim=1)
    cos_val = torch.nn.functional.cosine_similarity(x0, x, dim=-1)
    constraint_ok = constraint.satisfied(x0, x) if constraint is not None else True
    return {
        "success": bool((new_id == target_code).all()),
        "perturbation_norm": float(delta.mean().detach()),
        "cosine_to_original": float(cos_val.mean().detach()),
        "steps": step + 1,
        "original_id": int(orig_id[0]),
        "final_id": int(new_id[0]),
        "level": level,
        "target_code": target_code,
        "constraint_ok": constraint_ok,
        "x": x.detach(),
    }
