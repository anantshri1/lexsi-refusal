# Phase 4 (Intervention) — study notes (2026-11-10 .. 13; numbers canonical_numbers.txt p4_*, p4s_*, p4f_*; predictions P4.0-P4.9;
# figure figures/p4_tradeoff_v1.png; design PHASE4_PLAN.md; traps RECCE.md T1-T5)

## i) What question this phase answered
Can ONE internal push — adding the English refusal direction r_en to the residual stream at one layer — make Tiny Aya refuse harmful
prompts in Hindi/Italian/German as well as translating to English first does (translate_en: two passes, English answers), without
making it refuse harmless prompts or get worse at ordinary tasks? And does a Hindi-specific direction do better (hypothesis B)?

## ii) Key result (one number + why it matters)
Hindi harmful-request compliance 22.6% -> 8.6% with r_en steering (paired -14.0 pp [-21.7,-6.5]) = translate_en's 8.7% (difference -0.1
[-5.3,+5.3]) in ONE pass, answering in Hindi; better than the system prompt (15.1%; -6.5 [-12.9,0.0], borderline). Cost: over-refusal on
harmless REQUESTS 2-4% in every language (Hindi 2.2%); on harmless STATEMENTS more pushback (Hindi total 9.6% vs translate_en 5.5%);
reading comprehension (Belebele) -0.1 to -0.8 pp. Steering with the Hindi direction does the same (F2 hi 9.7%).
Why it matters: the multilingual safety gap is not a missing Hindi safety mechanism. The English one is shared and can be re-engaged from
the inside; what is lost on the Hindi route is part of the harm signal, which a push along the shared direction restores.

## Objects and glossary (what each thing is)
| Term | Meaning |
|---|---|
| slot k | residual stream after decoder layer k-1 (slot 0 = embeddings). We always say slot. SafeTune index = slot - 1 (trap T1). |
| r_en / r_hi | unit diff-of-means direction (H_core harmful - benign twin) at <|START_RESPONSE|>, English / Hindi, cross-fitted by pair (5 folds). |
| steer (SafeTune) | h <- h + alpha * r at EVERY position (prompt and generated tokens), one layer. Verified: slot-24 score moves by exactly alpha. |
| alpha / Delta | push size as a fraction of the English harmful-benign separation at that slot (Delta_24 = 30.34). F1 = 0.2 -> alpha 6.07. |
| ablate (SafeTune) | remove the r component at every layer: h <- h - (h.r) r. Tests necessity. |
| forecast (block 0.3) | no-GPU prediction: shift every Phase 3 score by alpha, read P(refuse) off each language's fitted score->refusal curve. |
| F1 / F2 | full runs: r_en / r_hi at slot 24, alpha/Delta 0.2, 1032 rows (en/it/de/hi pair_keep, both tiers), 512 tokens. |
| GAP / BOTH-fail | Hindi failures whose English twin was refused (GAP) / also complied (BOTH). |
| over-refusal, human-corrected | j1 flags benign refusals (precision ~0.2); Anant labels all j1 flags + sampled negatives per lang x tier; correction scales by stratum. |
| H_core / H_bait twins | benign twins of real harmful REQUESTS / of hateful STATEMENTS. Over-refusal on statements = pushback on a harmless claim. |

## What we did, in order (with the result of each step)
0. Steering vectors at slots 18/21/24 (r_en, r_hi, food/travel control, random); Delta_k; cos(r_en,r_hi) 0.96/0.96/0.90; control and random
   ~orthogonal. Forecast: Hindi reaches translate_en level at alpha/Delta ~0.25 (it did: dev 7.3% at 0.25; full 8.6% at 0.2).
1. SafeTune cross-check: its extract_refusal_direction == our r_en (cos 1.000). Steer moves the slot-24 score by exactly alpha.
2. WildGuard (AuditKIT) as cheap over-refusal judge: registered rule FAILED (recall 0.49 pooled) because it does not count pushback on
   statements as refusal; on H_core twins recall 0.95, precision 0.79 -> used for SCREENING on H_core only (post-hoc D11), re-validated
   under steering (precision 0.90). Its harmful_response flag fired on 0/51 compliant rows (positive control: it CAN fire, 7/10 on its
   own test set -> real severity/language threshold, not a bug). AuditKIT feeds two BOS tokens (minor).
3. Screening on 55 dev pairs (folds 0-2), en/hi/de, 256 tokens, 14 conditions, SafeTune per-fold vectors:
   - KILL-TEST clean: hi compliance r_en -16.4 pp vs food/travel control +3.6 and random -1.8 (same push size).
   - Dose-response at all three slots; the forecast tracked slot 24 closely (15.3/10.4/6.9 vs 16.4/10.9/7.3).
   - All-layer ablation: compliance en 3.6 -> 40, hi 23.6 -> 44, de 16.4 -> 51 (direction necessary; gap mostly closes).
   - Over-refusal (WildGuard) cost lands on ENGLISH; selection rule picked slot 24, alpha/Delta 0.2 (neighbours within noise).
4. Full runs F1/F2 (judge j1 moved to provider deepinfra after a 100-row agreement check: 0.95 / 0.97), Anant's over-refusal audit,
   WildGuard second judge, Belebele utility via AuditKIT's native engine with the SafeTune hook.
5. Post-hoc, symmetric: census of the Hindi H_bait j1-negatives (F1 12.8% rested on one row -> 9.6%; F2 2.4 -> 9.8%).
6. Post-hoc: language-conditional policy (English unsteered) composed at zero cost; Hindi pushed to 0.3 (P4.9) -> saturation.

## Final table (Hindi unless stated; full rows, 512 tokens)
| condition | H_core compliance | over-refusal (human-corr.) | Belebele | passes / answers |
|---|---|---|---|---|
| base | 22.6 | 0.0 | 57.8 | 1 / Hindi |
| sysprompt | 15.1 | 0.8 | — | 1 / Hindi |
| translate_en | 8.7 | 5.5 | — | 2 / English |
| F1 steer r_en (0.2) | 8.6 | 9.6 (H_core 2.2 / H_bait 9.7) | 57.7 | 1 / Hindi |
| F2 steer r_hi (0.2) | 9.7 | 9.8 (2.2 / 9.7) | 57.8 | 1 / Hindi |
| r_en, hi at 0.3 (P4.9) | 9.7 | 4.0 (1.1 / 12.9) | — | 1 / Hindi |
All four languages under F1: compliance en 3.0 / it 7.1 / de 9.4 / hi 8.6 (base 5.1/13.3/15.6/22.6); gap vs en hi +17.4 -> +5.4.

## Mechanism reading (what the causal results add to Phase 3)
- A (shared direction), in its LOCAL form, best fits: r_hi steering = r_en steering (F2-F1 hi +1.1 [-2.3,4.7]); control/random do nothing;
  ablating r_en removes refusal in all languages and mostly erases the en-hi gap.
- Rescue: GAP failures rescued 13/17, BOTH-fail 1/4. Steering restores harm signal LOST IN TRANSLATION (Phase 3 twin loss) but cannot
  create recognition English never had. This also explains P4.9's saturation: the residual Hindi failures are mostly BOTH-type.
- C-like readout remains: Hindi's score->refusal curve is higher and softer (mid 13.6 vs 9.75, width 13.7 vs 6.4) — part of why a push
  works at all. C's specific prediction (target-language steering beats English) did not come true.
- B disfavoured, with a power caveat: r_en and r_hi are 0.90-aligned at slot 24, so the head-to-head cannot separate them strongly.

## iii) Key decisions, and why
- Steer at slot 24 (Phase 3: direction shared 18-24, twin loss 12-21; forecast + screening equal across slots; 24 cheapest in over-refusal).
- Kill-test FIRST (control + random at the same push) — the cheapest falsifier of "it is the refusal direction".
- Screen on dev folds, report on ALL pairs, test folds as robustness (D2a; dev chose only slot and alpha).
- Every steering vector extracted by SafeTune in-job and checked against ours (cos >= 0.9999) — genuine toolkit use, same object.
- WildGuard only where it was validated (H_core twins), re-validated under steering; finals stay j1 + human (D11); never averaged.
- Judge provider switch only after a pre-registered agreement test (P4.6).
- Post-hoc analyses (census, language-conditional, hi 0.3) logged as post-hoc and done symmetrically (F1 AND F2).
- Dropped: full control run (option-2 rule: kill-test clean on dev), conditional steering, per-language alpha beyond the free version.

## iv) Assumptions and inputs (what the model knew, how we measured)
- Model Tiny Aya Global (3.35B, 36 layers), greedy decoding, base chat template, 512 tokens (dev 256; B8: verdict stable from 256).
- Directions from Phase 3 activations (pre-norm residual, <|START_RESPONSE|>), cross-fitted by pair so a prompt is never steered by a
  direction that saw its pair. Steering is an inference-time hook: no weights change; the model learns nothing.
- Compliance = j1 (Qwen2.5-72B, Part A v1); baselines/screening on featherless, full runs on deepinfra (agreement 0.95/0.97).
- Over-refusal = human-corrected (Anant, Part A v2); WildGuard = second judge on H_core twins. Utility = Belebele (AuditKIT native, zero-shot,
  log-likelihood over A-D, raw text, no chat template).

## v) Limits and known failures
- Held-out test folds: only 5 Hindi failures (3 GAP, 2 BOTH); F1 rescues 1 GAP, creates none -> replication inconclusive (n=3).
- Hindi over-refusal estimates move by 4-6 pp with a few labelled rows (12.8 -> 9.6 after census; 4.0 at 0.3) -> report as a range (~4-10%).
- Baselines' over-refusal cannot be split by tier (Phase 2 sheets drawn per language) -> H_core-vs-baseline compares with baseline totals.
- Language-conditional steering assumes an oracle language label. The gap-vs-en metric can shrink by English getting worse (seen here).
- One model, one direction type (diff-of-means), one position; steering added at every position incl. generated tokens.
- Mixed judge providers (stated, checked); Belebele measured without chat template (same op, different input format).
- Toolkit failures found and fixed (reported as findings): SafeTune layer-index convention (T1), SafeTune extractor OOM on T4 (T4),
  AuditKIT run_lmeval cannot see hooks (T3), AuditKIT DiskCache returns cached results for hooked models (T5, Belebele v1 invalid),
  AuditKIT exact_match crashes on int targets, AuditKIT WildGuard double BOS.

## vi) Open items and next step
- Ship (Sat 14 Nov): HF article, public dataset + artefacts (+ judgements), README, optional Space.
- Next science (shortlist window): en->hi activation patching per layer on GAP pairs (causal fork localisation; CircuitKIT);
  conditional steering gated on the prompt's own score (should cut the H_bait pushback cost); per-language alpha with a dev metric that
  sees pushback (WildGuard does not); a second model.

## Explain-back (to fill after the discussion)
