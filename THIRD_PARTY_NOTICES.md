# Third-Party Notices

This project builds on the following open-source software and research. The
original code and publications remain the property of their respective authors.

## RQ-VAE-Recommender (vendored, unmodified)

- **Location:** `RQ-VAE-Recommender-main/`
- **Source:** https://github.com/EdoardoBotta/RQ-VAE-Recommender
- **License:** MIT License (see `RQ-VAE-Recommender-main/LICENSE`)
- **Author:** Edoardo Botta (2026)
- **Archive:** https://doi.org/10.5281/zenodo.22076163
- **Trained model:** `edobotta/rqvae-amazon-beauty` (Hugging Face), checkpoint
  `trained_models/rqvae_amazon_beauty/checkpoint_399999.pt` used as-is.

Citation:

```bibtex
@software{botta2026rqvae,
  author  = {Botta, Edoardo},
  title   = {RQ-VAE Recommender},
  version = {1.0.1},
  year    = {2026},
  doi     = {10.5281/zenodo.22076163},
  url     = {https://doi.org/10.5281/zenodo.22076163}
}
```

**Modifications:** none. The upstream source tree is used unmodified. Our
pipeline (`fyp/`) imports it as a library; the runtime workarounds for
upstream/environment incompatibilities (sentence-transformers 5.x input
validation, fp16 encoder loading, user-id sanitization of the P5 dataset zip)
live entirely in `fyp/common/prepare_data.py` as runtime monkeypatches and
workspace-level data fixes, documented in `fyp/README.md`.

## Underlying research

- Rajput et al., *Recommender Systems with Generative Retrieval* (TIGER),
  NeurIPS 2023 — the generative-retrieval pipeline this project audits.
- Lee et al., *Autoregressive Image Generation using Residual Quantization*
  (RQ-VAE), 2022 — residual quantization scheme.
- Amazon Product Data (McAuley Lab), classic Beauty 5-core — evaluation data.

## Model and library dependencies

- `sentence-transformers/sentence-t5-xxl` (Apache 2.0) — item text encoder.
- PyTorch (BSD-3), transformers (Apache 2.0), polars (MIT), matplotlib (PSF) —
  see `fyp/README.md` for the environment setup.
