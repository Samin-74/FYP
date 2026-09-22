"""Semantic-preservation constraints for steering attacks (plan section 3.6).

Stubs operate in embedding space: the attacker perturbs the Sentence-T5
embedding of a listing; realising the perturbation as actual text (HotFlip /
LLM rewrite) is Phase-3 work and plugs into the same constraint interface.
"""

import torch


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
