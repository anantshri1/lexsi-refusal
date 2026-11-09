# Phase 3 (Geometry) — study notes (in progress, 2026-11-09)

## The objects
- h = the model's internal state (residual stream, 2048 numbers) at layer k, at the token <|START_RESPONSE|>
  (the moment just before it starts answering). Saved for all 1770 prompts, 37 slots (embeddings + 36 layers).
- r_en = (average English harmful h) - (average English benign h), normalised to length 1 = "the English ruler",
  pointing from benign to harmful. Estimated by cross-fitting: a prompt is never scored by a ruler built from its own pair.
- s = h . r_en = one number per prompt: how far along the ruler it sits. Each language -> two clouds of scores
  (harmful H_core, benign twins).

## Metric glossary (what each number means)
| Metric | Measures | Plain meaning |
|---|---|---|
| AUROC | P(random harmful scores higher than random benign) | Overlap only. 1.0 = clouds never overlap, 0.5 = indistinguishable. Ignores WHERE clouds sit. Saturates near 1 (ceiling) -> d' more informative. |
| d' | (centre_harmful - centre_benign) / pooled width | Signal-to-noise. 4.6 = centres 4.6 widths apart. Drops if clouds approach or spread. |
| tau (τ) | Midpoint between the ENGLISH cloud centres (training folds only) | Our stand-in for "the English refusal line", reused for every language. NOT the model's own threshold. |
| trig_h | Share of harmful prompts with s > tau | "How many harmful prompts look harmful by English standards." |
| trig_b | Share of benign prompts with s > tau | False alarms. |
| [lo, hi] | 95% pair-bootstrap CI of (en - language), paired on shared pairs | Contains 0 -> cannot tell the two languages apart. |
| row AUROC (5d) | Within ONE language, P(refused harmful prompt scores higher than complied one) | Does the ruler predict what the model actually DID, prompt by prompt? |

## Hypothesis pictures (along r_en)
- Squeeze (Wang 2025, ~A): clouds overlap in hi -> AUROC, d' drop.
- Shift (one reading of Aziz 2026): both hi clouds slide down -> harmful falls below the English tau -> trig_h drops.
- Neither -> KILL (the English ruler cannot see the hi gap in the representation).

## Results so far
1. Extraction (HF Job, $0.04): own hooks, PRE final norm (HF hidden_states[-1] is post-norm). Positions verified on all rows.
2. 5b: ruler is stable (fold cosines >= 0.99 from layer 12); English d' plateau ~4.4-4.65 over layers 18-36; k* = 24.
3. 5c KILL-TEST (registered rule): KILL for hi (and it, de). hi looks like en on the ruler: AUROC 0.990 vs 0.994,
   d' 4.07 vs 4.60, trig_h 0.91 vs 0.95, CIs include 0. Harm is represented in hi as in English.
   ml, hi-Latn: COLLAPSE (d' ~0.8, trig_h ~0.09): the ruler barely tells harmful from benign (not understood).
   Script: same Hindi content in Latin script, d' 4.07 -> 0.80.
4. 5d row level: in en the ruler predicts behaviour perfectly (row AUROC 1.00; 4 of 5 failures sit below tau =
   "not recognised as harmful"). In hi the ruler still ranks well (row AUROC 0.89), but 13 of 21 failures sit ABOVE tau
   ("recognised, complied anyway"); complied hi mean score 11.4 vs refused 22.1 (tau ~9).
   Excess failures hi vs en: +4 below tau, +12 above tau -> ~3/4 of the gap is "recognised but complied".
   CAUTION: "above tau" depends on where tau sits, and tau (midpoint of English means) is OUR line, not the model's.
   Two readings remain open, decided in 5e (paired twins, exploratory):
   (i) DECISION threshold differs: the model needs a higher score to refuse in hi than in en (C / Aziz's real claim).
   (ii) REPRESENTATION weakens on the specific prompts that fail: their hi score drops well below their English twin's,
        but still lands above our (too low) tau -> a localised shift that cloud averages hide.
   Implication for Phase 4 either way: pushing hi activations up along r_en might close the gap.

## Hypotheses status (SCOPE A/B/C)
- C (shared representation, different readout) is the leading candidate, pending 5e: representation shared (English ruler separates hi
  as well as en), readout differs (hi complies at scores that refuse in en).
- B (language-specific) disfavoured for it/de/hi: a separate Hindi mechanism would not leave the English ruler this sharp in hi.
  Not excluded: hi could have its own refusal readout. Tests: cosine(r_en, r_hi) vs control (step 4), Phase 4 steering.
- A (shared, weaker trigger) in its Wang form (squeeze) is out for it/de/hi.
