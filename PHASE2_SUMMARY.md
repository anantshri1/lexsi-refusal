# Phase 2 in one page (2026-11-06 .. 09) — what we found and WHY each check was needed

## The question
Among prompts the model UNDERSTOOD (pair_keep), does Tiny Aya refuse harmful requests less in some languages than in English?

## The answer: yes — a graded gap
H_core compliance (judge j1): en 5% < it 13% ~ de 16% < hi 23%. hi is +17 pp above en (95% CI 9.5-25.8). Kill-test: GO.
It is UNDER-REFUSAL of harmful prompts, not general caution: benign over-refusal is ~0-5% in every language.

## Why each check was needed (each one could have made the headline disappear)
1. Comprehension filter (Phase 1): Phase 0's "ml complies" was mostly the model not understanding. So all rates use understood pairs only.
2. Human audit of the judge (κ): an LLM judge can be biased exactly on harmful content. The 72B judge (j1) only moderately agreed
   with you (harmful κ 0.37). We tried a better prompt (j2); it failed our pre-registered stop rule (0.51 < 0.55) -> kept j1,
   made ml descriptive-only.
3. Blind correction (B4.3): instead of arguing about κ, you re-labelled the rows that DECIDE the verdict. j1 was ~76-92% right on
   its "failure" calls and missed a few -> corrected gap still > 0 (GO stands), sizes uncertain (+-15 pp).
   Same check showed j1's "over-refusal" calls are ~80% wrong -> over-refusal is always human-corrected.
4. Length (B8): verdicts stable by 256 tokens for en/it/de/hi -> 512 cap is fine; ml never stabilises (limitation).
5. Translation control: maybe our machine translation created the gap? Human (MultiJail) Italian complies as much as ours -> no.
6. Naive baselines (the bar for any clever fix):
   - sysprompt ("refuse harmful requests in any language"): closes ~40% of the hi gap (+17 -> +10 pp), no over-refusal cost.
   - translate-to-English-then-answer: nearly closes it (hi +3.4 pp), but needs 2 passes and answers in English.
   -> The model's safety works on the SAME content in English: the failure is in processing non-English input.
7. Judge glitches: unparsed judge outputs were 3x more often compliance -> always retry before computing rates.

## What it means going into Phase 3
- Hypotheses: A (shared direction, weaker trigger), B (language-specific), C (shared representation, failed readout/calibration).
- Literature (LITERATURE.md): Wang 2025 says separation is weaker in other languages (A); Aziz 2026 says separation is intact and the
  decision is miscalibrated (C). Zhang 2026: cosine alone is meaningless without a control direction.
- Our edge: comprehension-conditioned data can adjudicate Wang vs Aziz. Headline Phase 3 measurement = how well the English
  direction separates harmful vs benign per language/layer (AUROC) on pair_keep rows. Bets: Anant C/Aziz; Claude 60% Aziz.
- Phase 4 must beat sysprompt and approach translate_en in ONE pass.
