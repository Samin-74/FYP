"""White-box targeted steering attack, embedding space (plan section 3.6).

The attacker knows the RQ-VAE and Sentence-T5 encoder (white box) and perturbs
the item's input embedding to force a *target* code — either a single level
(e.g. land on a high-traffic level-0 prefix) or the full semantic ID (collide
with a bestseller on every level). Text realisation of the perturbation
(HotFlip / LLM rewrite ranked by embedding progress) reuses these optimisers
unchanged.

Objective per step: descend the distance from the residual(s) to the target
codeword(s) and stop as soon as the target wins the argmin. Two projections are
*enforced inside* the loop after every step, so successful attacks are
guaranteed to satisfy them:

  - cosine(original, perturbed) >= threshold (semantic budget);
  - ||perturbed|| = ||original|| (stay on the Sentence-T5 output sphere; see
    ``fyp.attack.constraints.project_to_norm``).
"""

import torch

from fyp.attack.constraints import project_to_cosine, project_to_norm
from fyp.common.generate_semantic_ids import codebook, codebook_dist


def _project(x0, x, cosine_threshold, keep_norm):
    if cosine_threshold is not None:
        x = project_to_cosine(x0, x, cosine_threshold)
    if keep_norm:
        x = project_to_norm(x0, x)
    return x


def assigned_ids(model, x: torch.Tensor) -> torch.Tensor:
    """Full semantic ID (B, L) under the upstream assignment rule (no grad)."""
    with torch.no_grad():
        r = model.encode(x)
        out = []
        for layer in model.layers:
            ids = codebook_dist(layer, r).argmin(axis=1)
            out.append(ids)
            r = r - layer.get_item_embeddings(ids)
    return torch.stack(out, dim=1)


def _residual_to_level(model, x, level: int):
    """Residual r_k at `level`, differentiable w.r.t. x."""
    r = model.encode(x)
    for k, layer in enumerate(model.layers):
        if k == level:
            return r, layer
        ids = codebook_dist(layer, r).detach().argmin(axis=1)
        r = r - layer.get_item_embeddings(ids)
    raise ValueError(f"level {level} out of range")


def _result(x0, x, success, steps, orig, final, target, constraint, level):
    delta = (x - x0).norm(dim=1)
    cos_val = torch.nn.functional.cosine_similarity(x0, x, dim=-1)
    constraint_ok = constraint.satisfied(x0, x) if constraint is not None else True
    return {
        "success": success,
        "perturbation_norm": float(delta.mean().detach()),
        "cosine_to_original": float(cos_val.mean().detach()),
        "steps": steps,
        "original_id": orig,
        "final_id": final,
        "level": level,
        "target_code": target,
        "constraint_ok": constraint_ok,
        "x": x.detach(),
    }


def targeted_flip(
    model,
    x0: torch.Tensor,
    target_code: int,
    level: int = 0,
    cosine_threshold: float | None = 0.95,
    constraint=None,
    max_steps: int = 300,
    step_size: float = 0.05,
    keep_norm: bool = True,
) -> dict:
    """Steer x0 so that the level-`level` argmin becomes `target_code`.

    ``cosine_threshold`` (default 0.95) and the norm projection are enforced
    after every step. ``constraint`` (fyp.attack.constraints) is additionally
    *checked* at the end for reporting compatibility.

    Returns dict: success, perturbation_norm, cosine_to_original, steps,
    original_id, final_id, level, target_code, constraint_ok, x.
    """
    model.eval()
    x0 = x0.detach()
    x = x0.clone().requires_grad_(True)
    orig_id = assigned_ids(model, x0)[:, level]

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
            x.copy_(_project(x0, x, cosine_threshold, keep_norm))
            new_id = assigned_ids(model, x)[:, level]
            if bool((new_id == target_code).all()):
                break

    return _result(
        x0, x, bool((new_id == target_code).all()), step + 1,
        int(orig_id[0]), int(new_id[0]), target_code, constraint, level,
    )


def targeted_collision(
    model,
    x0: torch.Tensor,
    target_ids,
    cosine_threshold: float | None = 0.95,
    constraint=None,
    max_steps: int = 300,
    step_size: float = 0.05,
    keep_norm: bool = True,
) -> dict:
    """Steer x0 so that its *full* semantic ID equals ``target_ids`` (all levels).

    Loss = sum_k dist(r_k, c_k[t_k]) where the residual chain is built with the
    target codes (r_{k+1} = r_k - c_k[t_k]), i.e. the distance from the encoded
    item to the point the target ID reconstructs, decomposed per level. Success
    requires every level's argmin to match — a genuine semantic-ID collision,
    not just a shared prefix.
    """
    model.eval()
    x0 = x0.detach()
    x = x0.clone().requires_grad_(True)
    target = torch.as_tensor(list(target_ids), device=x0.device)
    orig = assigned_ids(model, x0)[0]

    cur = orig
    for step in range(max_steps):
        if x.grad is not None:
            x.grad.zero_()
        r = model.encode(x)
        loss = 0.0
        for k, layer in enumerate(model.layers):
            loss = loss + codebook_dist(layer, r)[:, target[k]].sum()
            r = r - codebook(layer)[target[k]]
        loss.backward()
        with torch.no_grad():
            g = x.grad
            gn = g.norm(dim=1, keepdim=True)
            gn = torch.where(gn < 1e-12, torch.ones_like(gn), gn)
            x -= step_size * g / gn
            x.copy_(_project(x0, x, cosine_threshold, keep_norm))
            cur = assigned_ids(model, x)[0]
            if bool((cur == target).all()):
                break

    return _result(
        x0, x, bool((cur == target).all()), step + 1,
        tuple(int(v) for v in orig), tuple(int(v) for v in cur),
        tuple(int(v) for v in target), constraint, None,
    )
