"""Semantic-preservation constraints for steering attacks (plan section 3.6).

Stubs operate in embedding space: the attacker perturbs the Sentence-T5
embedding of a listing; realising the perturbation as actual text (HotFlip /
LLM rewrite) is Phase-3 work and plugs into the same constraint interface.

All embedding-space optimisers apply ``project_to_cosine`` (semantic budget)
followed by ``project_to_norm`` (stay on the encoder's output sphere) after
every step.
"""

import torch


def project_to_cosine(x0: torch.Tensor, x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Shrink ``delta = x - x0`` so that cos(x0, x) >= threshold (bisection).

    Rows that already satisfy the threshold are returned unchanged.
    """
    delta = x - x0
    ok = torch.nn.functional.cosine_similarity(x0, x, dim=-1) >= threshold
    lo = torch.zeros(x.shape[0], device=x.device)
    hi = torch.ones(x.shape[0], device=x.device)
    for _ in range(25):
        mid = (lo + hi) / 2
        xm = x0 + mid[:, None] * delta
        cos = torch.nn.functional.cosine_similarity(x0, xm, dim=-1)
        better = cos >= threshold
        lo = torch.where(better, mid, lo)
        hi = torch.where(better, hi, mid)
    scale = torch.where(ok, torch.ones_like(lo), lo)
    return x0 + scale[:, None] * delta


def project_to_norm(x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Rescale each row of ``x`` to the L2 norm of the matching row of ``x0``.

    Sentence-T5 emits unit-norm embeddings, so no text can produce a vector
    off that sphere. Cosine is scale-invariant, so without this projection an
    optimiser can move radially at cos = 1 and exploit the (non scale-invariant)
    RQ-VAE encoder with an unrealisable embedding. Projecting after a cosine
    projection leaves the cosine unchanged.
    """
    target = x0.norm(dim=-1, keepdim=True)
    return x * (target / torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-12))


class EmbeddingCosineConstraint:
    """Require cosine(original, perturbed) >= threshold."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def similarity(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(x0, x, dim=-1)

    def satisfied(self, x0: torch.Tensor, x: torch.Tensor) -> bool:
        return bool((self.similarity(x0, x) >= self.threshold).all())

    def describe(self) -> str:
        return f"cosine>={self.threshold}"


class NormBudgetConstraint:
    """Require ||x - x0||_2 <= budget (seller rewrite-budget analogue)."""

    def __init__(self, budget: float):
        self.budget = budget

    def satisfied(self, x0: torch.Tensor, x: torch.Tensor) -> bool:
        return bool(((x - x0).norm(dim=-1) <= self.budget).all())

    def describe(self) -> str:
        return f"L2<={self.budget}"


class AndConstraint:
    def __init__(self, *constraints):
        self.constraints = constraints

    def satisfied(self, x0: torch.Tensor, x: torch.Tensor) -> bool:
        return all(c.satisfied(x0, x) for c in self.constraints)

    def describe(self) -> str:
        return " & ".join(c.describe() for c in self.constraints)
