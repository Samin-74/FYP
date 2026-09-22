"""Black-box query-only steering stub (plan section 3.6).

Threat model: the seller queries only the tokenizer (item text in, semantic ID
out) — no gradients, no codebook access. This stub searches in embedding space
with random/local perturbations using only ``get_semantic_ids`` as the oracle,
to be replaced by an LLM-rewrite evolutionary/bandit search that proposes
text edits. The oracle interface is identical either way.
"""

import torch

from fyp.common.generate_semantic_ids import DEVICE
from fyp.audit.metrics import assigned_id_at_level


def oracle_id(model, x: torch.Tensor, level: int) -> int:
    """The only capability the black-box attacker is assumed to have."""
    return int(assigned_id_at_level(model, x.to(DEVICE), level))


def query_only_flip(
    model,
    x0: torch.Tensor,
    target_code: int,
    level: int = 0,
    constraint=None,
    budget: int = 200,
    sigma: float = 0.5,
    seed: int = 0,
) -> dict:
    """Random-search stub: sample perturbations, keep the first that flips to target.

    Returns dict with the same keys as white_box.targeted_flip plus queries.
    """
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0c = x0.detach().to(DEVICE)
    orig_id = oracle_id(model, x0c, level)
    best = x0c.clone()
    queries = 0

    for q in range(budget):
        delta = sigma * torch.randn(x0c.shape, generator=g).to(DEVICE)
        x = x0c + delta
        queries += 1
        if oracle_id(model, x, level) == target_code:
            if constraint is None or constraint.satisfied(x0c, x):
                best = x
                break
    else:
        best = x0c  # never found a constrained flip

    success = oracle_id(model, best, level) == target_code and not torch.equal(best, x0c)
    return {
        "success": bool(success),
        "perturbation_norm": float((best - x0c).norm()),
        "cosine_to_original": float(
            torch.nn.functional.cosine_similarity(x0c, best, dim=-1)
        ),
        "steps": queries,
        "queries": queries,
        "original_id": orig_id,
        "final_id": oracle_id(model, best, level),
        "level": level,
        "target_code": target_code,
        "constraint_ok": True if constraint is None else constraint.satisfied(x0c, best),
    }
