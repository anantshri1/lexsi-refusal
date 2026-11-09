# LITERATURE — Phase 3 step 1 (2026-11-09). Novelty positioning; read before writing up.

1. Wang, Wang, Liu, Schütze, Plank — "Refusal Direction is Universal Across Safety-Aligned Languages", NeurIPS 2025 (arXiv 2505.17306; PolyRefuse, 14 langs).
   English refusal direction ablates refusal in other languages near-perfectly; refusal vectors ~parallel across languages;
   BUT harmful vs harmless prompts are less separated in non-English (esp. low-resource / non-Latin) -> cause of cross-lingual jailbreaks.
   ~ Hypothesis A ("shared direction, weaker trigger").
2. Aziz, Hanif, Koto — "Low-Resource Safety Failures Are Action Failures, Not Representation Failures", arXiv 2606.01196 (May 2026).
   Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B, 23 langs: high-resource harmfulness direction separates low-resource harmful/harmless
   nearly as well as high-resource; failure is calibration of the decision. Fix: low-rank logistic readout, threshold reset with 1-4
   target-language examples. ~ Hypothesis C. No comprehension / translation-quality controls described.
   -> DIRECT TENSION with (1) on whether separation is weaker. Our comprehension control + paired twins can adjudicate.
3. Zhang — "Does Safety Live in a Shared Multilingual Subspace? Cross-Lingual (Non-)Transfer of the Refusal Direction", MRL 2026 (ACL Anthology 2026.mrl-1.19).
   Qwen2.5-7B (4-bit), Llama-3.2-3B; 10 MultiJail langs (incl. it). High cross-lingual cosine is GENERIC: a non-safety control
   contrast has about the same cosine (Llama 0.84 vs 0.86) -> cosine alone says nothing about a shared safety mechanism.
   Causal: English ablation transfers (Qwen), English addition induces refusal elsewhere (Llama) but the two halves dissociate.
   Limitations they list: translationese confound, LLM judge, no control ablation, degenerate layer selection. Aya-Expanse pre-registered, not finished.
   -> CLOSEST prior work (MultiJail + diff-of-means + English vs own direction). We must differ visibly.

Implications for our Phase 3/4:
- Cosine matrices MUST be reported against a non-safety control contrast (same estimator, same language) — else uninformative (3).
- Central adjudication (1) vs (2): per language & layer, how well does r_en separate harmful vs benign (AUROC), CONDITIONAL on
  comprehension (pair_keep)? Equal separation + behavioural gap -> calibration/readout (C, Aziz). Weaker separation -> A (Wang).
- Our distinct contributions: multilingual-first model (Tiny Aya); comprehension-conditioned analysis (pair_keep); human-audited,
  judge-error-corrected rates; MT control with human Italian; naive baselines incl. translate_en as the bar; script question
  (hi vs hi-Latn); ml as out-of-training-set language; paired-twin (same prompt in en) layer-wise divergence.
- Include a control direction in ablation/addition (Zhang's missing control).
