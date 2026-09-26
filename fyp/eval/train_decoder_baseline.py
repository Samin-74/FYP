"""Launch upstream TIGER decoder (retriever) training on Windows.

Upstream ``RQ-VAE-Recommender-main/train_decoder.py`` calls
``torch.compile(model)``, which requires triton — unavailable on Windows — so
compilation is patched to eager mode (same numerics, somewhat slower). The
upstream tree is never edited; this is a runtime monkeypatch, per the repo
rules in fyp/README.md.

Usage:
  python -m fyp.eval.train_decoder_baseline configs/decoder_beauty_fyp.gin
"""

import runpy
import sys
from pathlib import Path

import torch

UPSTREAM_SCRIPT = (
    Path(__file__).resolve().parents[2] / "RQ-VAE-Recommender-main" / "train_decoder.py"
)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/decoder_beauty_fyp.gin"
    # triton has no Windows build; compile to eager. Handles both
    # torch.compile(model) and @torch.compile(mode=...) decorator usage.
    def _eager_compile(model=None, *args, **kwargs):
        return (lambda m: m) if model is None else model

    torch.compile = _eager_compile  # type: ignore[assignment]
    sys.argv = [str(UPSTREAM_SCRIPT), config_path]
    runpy.run_path(str(UPSTREAM_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
