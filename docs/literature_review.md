# Literature Review & Annotated Bibliography

**Status:** Phase 0 deliverable; feeds interim report §2 (literature review).
**Provenance:** references identified during the project and the pre-interim
audit; every entry verified against the source (arXiv listing /
ACM DL / project page). The positioning claim in §6 reflects
this non-exhaustive search and must be re-checked before the final report
(plan §1).

## 1. Generative retrieval and semantic IDs (foundations)

- **Rajput, S., Mehta, N., Singh, A., et al. (2023).** *Recommender Systems
  with Generative Retrieval* (TIGER). NeurIPS 2023. arXiv:2305.05065.
  Introduces generative retrieval for recommendation: a Transformer
  sequence-to-sequence model autoregressively decodes the *semantic ID* of the
  next item, where semantic IDs are tuples of RQ-VAE codewords computed from
  content embeddings. Outperforms then-SOTA on Amazon benchmarks and
  generalises better to cold-start items.
  *Relevance:* the architecture we audit. Our Beauty baseline is a local-scale
  reproduction of this pipeline on the same dataset slice; the paper's
  item-level Beauty numbers (Recall@5 0.0454, Recall@10 0.0648, NDCG@10
  0.0384) are the comparison target for our item-level re-evaluation.

- **Singh, A., Vu, T., Mehta, N., et al. (2023).** *Better Generalization with
  Semantic IDs: A Case Study in Ranking for Recommendations.* arXiv:2306.08121.
  Replaces randomly-hashed item IDs with content-derived semantic IDs inside
  an industry-scale ranking model (YouTube), showing improved generalisation
  on new and long-tail items without sacrificing overall quality; studies
  SentencePiece-style sub-pieces for adapting compact SIDs in ranking models.
  *Relevance:* semantic IDs are shared infrastructure beyond retrieval, so a
  tokenizer-level vulnerability propagates to ranking deployments, not only
  generative ones.

- **Lee, D., Kim, C., Kim, S., Cho, M., & Han, W.-S. (2022).** *Autoregressive
  Image Generation with Residual Quantization* (RQ-VAE). CVPR 2022.
  arXiv:2203.01941.
  Origin of residual quantization: a feature map is approximated by a stack of
  discrete codes, each level quantising the residual of the previous one.
  *Relevance:* the quantizer class whose Voronoi-cell geometry we attack.
  Boundary margins, codebook utilisation and training tricks (k-means init,
  commitment loss) determine how easily an item can be pushed across a cell
  boundary.

- **Wang, W., Bao, H., Lin, X., et al. (2024).** *Learnable Item Tokenization
  for Generative Recommendation* (LETTER). CIKM 2024, pp. 2400–2409.
  arXiv:2405.07314.
  Makes item tokenization learnable by integrating hierarchical semantics,
  collaborative signals, and code-assignment diversity into RQ-VAE training;
  adds a ranking-guided generation loss.
  *Relevance:* the main tokenizer-improvement line over TIGER's frozen RQ-VAE;
  motivates our checkpoint-robustness replication (a second, undocumented
  high-entropy checkpoint) and the margin-regularised-defence idea in Phase 5.

- **Botta, E. (2026).** *RQ-VAE-Recommender* (software, MIT).
  github.com/EdoardoBotta/RQ-VAE-Recommender.
  Open-source reproduction of TIGER-style RQ-VAE tokenization + T5 generative
  retrieval on Amazon data; ships trained RQ-VAE checkpoints.
  *Relevance:* vendored unmodified as our upstream (see
  THIRD_PARTY_NOTICES.md); all our audits run on its published checkpoint and
  its decoder/evaluation code paths.

## 2. Evaluating semantic-ID tokenizers

- **Zhang, Q., et al. (2026).** *Faithful Evaluation of Semantic-ID Tokenizers
  for Generative Recommendation.* arXiv:2605.25330 (v2, Aug 2026; v1 title:
  *How Reliable Are Semantic-ID Tokenizer Comparisons in Generative
  Recommendation?*).
  Shows SID-level evaluation is faithful only when each ID uniquely identifies
  one item: measured collision rates reach 30.52% across five tokenizers and
  four datasets, and SID-level Hit@10 is inflated by up to 103.36% relative to
  item-level Hit@10, enough to flip pairwise tokenizer comparisons. Proposes
  Collision-Corrected Evaluation (CCE: collision-aware ItemHit@K / ItemNDCG@K
  computed from generated SID sequences) and Zero-Collision Reassignment
  (ZCR: minimum-cost reassignment producing zero-collision IDs).
  *Relevance:* the methodological basis for our item-level retriever
  re-evaluation. Our catalogue has 27.1% of items in collision groups, squarely
  in the regime they study; our `item_strict` / `item_fractional` metrics are
  the same style of correction (unique-ID credit and 1/group-size credit)
  applied to our trained decoder, whose upstream metric is SID-level only.

## 3. Attacks on recommender systems

- **Wang, Z., et al. (2024).** *Poisoning Attacks and Defenses in Recommender
  Systems: A Survey.* arXiv:2406.01022.
  Surveys poisoning attacks from the attacker's perspective with a four-stage
  pipeline (goals → capabilities → victim architecture → strategy) and
  classifies defences into poisoning-data filtering and robust training.
  *Relevance:* threat-model scaffolding. In its taxonomy our attack is
  provider-side item promotion with listing-edit capability only — no fake
  users, no interaction injection, no ranker access.

- **Oh, S., Verma, G., & Kumar, S. (2024).** *Adversarial Text Rewriting for
  Text-aware Recommender Systems* (ATR). CIKM 2024. arXiv:2408.00312.
  Sellers adversarially rewrite their item descriptions with LLMs (two-phase
  fine-tuning or in-context learning) to unfairly uprank products in
  text-aware recommenders; rewrites are judged realistic by human evaluators.
  Demonstrated on 3 datasets and 4 recommender approaches.
  *Relevance:* closest prior work — same seller threat model. Differences: ATR
  attacks continuous text representations end to end, while we target the
  *discrete quantization layer* of a generative retriever (codebook
  boundaries, ID collisions); our current edits are greedy keyword stuffing,
  with LLM-faithful rewrites planned as attack v2.

- **Wang, Z., et al. (2025).** *ID-Free Not Risk-Free: LLM-Powered Agents
  Unveil Risks in ID-Free Recommender Systems.* arXiv:2409.11690.
  Deploys LLM agents that synthesise deceptive item descriptions imitating
  popular items (popularity extraction + multi-agent refinement) to promote
  low-quality items in ID-free recommenders under black-box settings, and
  proposes a detector for the generated text.
  *Relevance:* ID-free/generative recommenders share the text→embedding attack
  surface; their black-box setting matches our seller model, and their
  detection method is a defence-side baseline for Phase 5.

- **Deldjoo, Y., et al. (2025).** *Poison-RAG: Adversarial Data Poisoning
  Attacks on Retrieval-Augmented Recommender Systems.* arXiv:2501.11759.
  Manipulates item metadata (tags, descriptions) with LLM-generated content to
  promote long-tail items or demote popular ones in RAG-based recommenders;
  local (per-item BERT-embedding-guided) modifications outperform global ones
  by up to 50% on MovieLens; popular items prove more susceptible.
  *Relevance:* metadata poisoning of retrieval-based recommenders. Our attack
  needs no interaction data and targets a different pipeline stage (the
  tokenizer), but their popularity-susceptibility finding contrasts with our
  margin-vs-popularity null result and is worth a discussion paragraph.

- **Deldjoo, Y., et al. (2025).** *Provider-Side Data Poisoning in
  Retrieval-Augmented Recommender Systems.* arXiv:2505.05196.
  Formalises provider-side attacks under token-edit and semantic-similarity
  constraints: adding emotional keywords or borrowing phrases from related
  items shifts rankings and exposures on MovieLens while evading naive
  detection.
  *Relevance:* their constraint formalisation parallels our cosine budget; our
  finding that cos ≥ 0.95 is a weak semantic certificate (48.5% of items have
  a different product within 0.95) quantifies why such constraints do not
  neutralise this attack class.

## 4. Adversarial text methods (attack-v2 heritage)

- **Ebrahimi, J., Rao, A., Lowd, D., & Dou, D. (2018).** *HotFlip: White-Box
  Adversarial Examples for Text Classification.* ACL 2018. arXiv:1712.06751.
  Gradient-guided character/word substitutions that flip classifier decisions
  with few edits.
  *Relevance:* the template for our planned HotFlip-style token substitution
  against the Sentence-T5 encoder (plan §3.6), the text-realisable counterpart
  of our embedding-space steering.

- **Morris, J., Lifland, E., Yoo, J. Y., Grigsby, J., Jin, D., & Qi, Y. (2020).**
  *TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and
  Adversarial Training in NLP.* EMNLP 2020. arXiv:2005.05909.
  Modular framework decomposing attacks into goal function, constraints,
  transformation, and search.
  *Relevance:* our edit operators and the plan's attack baselines (random
  paraphrase, untargeted substitution) follow this decomposition; our
  random-keyword control arm is the untargeted-substitution baseline the plan
  §6 calls for.

## 5. Quantization and robustness (defence side)

- **Dong, Z., & Mao, K. (2023).** *Adversarial Defenses via Vector
  Quantization.* arXiv:2305.13651.
  Uses a vector quantizer as a preprocessing defence for image classifiers
  (patched / sliding-window RandDisc), with certifiable robust accuracy from
  rate–distortion arguments; remains effective under STE and EOT adaptive
  attacks.
  *Relevance:* quantization can *defend* as well as be attacked. Informs the
  Phase-5 defence candidates (stochastic quantization, margin regularisation)
  and, methodologically, the need for adaptive re-attack evaluation of any
  defence we propose.

## 6. Positioning — the gap this project occupies

Attacks on recommenders have targeted the ranker or the training data
(profile injection, poisoning; §3 survey). The closest text-rewriting works
(ATR; ID-Free Not Risk-Free; Poison-RAG; provider-side RAG poisoning)
manipulate *continuous* text representations of text-aware or RAG-based
recommenders. SID evaluation work (Zhang 2026) identifies the
collision-inflation problem but does not exploit it adversarially.
VQ-as-defence work is in computer vision. We found **no published work that
attacks or defends the RQ-VAE quantization layer of a generative recommender
itself**: its boundary geometry as an attack surface, full-ID collisions as an
exploit mechanism (a collision makes an item indistinguishable from a
bestseller at generation time), or tokenizer-level defences. That is the niche
of this project. This claim is based on a targeted, non-exhaustive search and
will be re-checked before the final report.

## References

1. Rajput et al. 2023 — arXiv:2305.05065 (NeurIPS 2023)
2. Singh et al. 2023 — arXiv:2306.08121
3. Lee et al. 2022 — arXiv:2203.01941 (CVPR 2022)
4. Wang et al. 2024 (LETTER) — arXiv:2405.07314 (CIKM 2024)
5. Botta 2026 — github.com/EdoardoBotta/RQ-VAE-Recommender (MIT)
6. Zhang et al. 2026 — arXiv:2605.25330
7. Wang et al. 2024 (survey) — arXiv:2406.01022
8. Oh, Verma & Kumar 2024 (ATR) — arXiv:2408.00312 (CIKM 2024)
9. Wang et al. 2025 (ID-Free Not Risk-Free) — arXiv:2409.11690
10. Deldjoo et al. 2025 (Poison-RAG) — arXiv:2501.11759
11. Deldjoo et al. 2025 (provider-side RAG poisoning) — arXiv:2505.05196
12. Ebrahimi et al. 2018 (HotFlip) — arXiv:1712.06751 (ACL 2018)
13. Morris et al. 2020 (TextAttack) — arXiv:2005.05909 (EMNLP 2020)
14. Dong & Mao 2023 — arXiv:2305.13651
