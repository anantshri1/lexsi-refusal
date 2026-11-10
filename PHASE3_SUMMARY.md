# Phase 3 (Geometry) — study notes (2026-11-09/10; figure figures/p3_geometry_v1.png; numbers canonical_numbers.txt)

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
| transfer (step 4) | cross-language cosine of directions from DISJOINT prompt halves / sqrt(within-language split-half reliabilities) | 1.0 = switching language costs nothing beyond re-sampling prompts. Removes shared-content and noise effects. |
| random-sign null | cosine of a direction built from randomly signed food/travel differences | How much cosine ANY grouping of the same translated content gets for free (Zhang). |
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

5. 5e paired twins (exploratory): for pairs where en refused but hi complied, the hi prompt scores 8.2 below its own
   English twin (median), vs 3.4 for pairs both refused -> the prompts that fail in Hindi LOSE EXTRA HARM SIGNAL (reading ii).
   Same in it (-6.8 vs -2.1), de (-5.2 vs -2.9). Plus: outside English, complied and refused scores overlap (softer decision).
   Why 5c said KILL anyway: the shift hits ~17 prompts (local, averaged away in clouds) and our tau (~9) is below the score
   where English actually starts refusing (11-18), so shifted prompts still counted as "above tau".
   Caveat: selecting pairs BY hi compliance favours low hi scores (selection bias); size makes pure artefact unlikely.
   Current story: shared ruler; a subset of prompts loses harm signal in translation and drops below the refusal zone,
   and the score->refusal mapping is noisier outside English. Closer to A ("shared direction, weaker trigger" on a
   subset) + a bit of C, than to either pure form.

6. 5f (post-hoc): trigger gap at fixed thresholds 9/12/15/18. At 9 (≈ registered tau) nothing; at 15, hi +0.12 [0.05,0.20],
   de +0.06, it +0.02 (ordered like behaviour); at 18 all three significant. So the registered KILL came from tau sitting
   below the zone where English actually refuses (11-18). Verdict reported as registered + this sensitivity, flagged post-hoc.
   Lesson: thresholds must be calibrated to BEHAVIOUR, not to class-mean midpoints.

7. Step 4 cosine vs a neutral control (food vs travel, 64 pairs, built 9 Nov; Zhang 2026). Raw cosines ~0.95 in the middle
   layers, but a RANDOM grouping of the same content already gets 0.56-0.77 -> raw cosine means little. Two defects found and
   fixed on the way: (a) unequal resampling biased the CIs; (b) the control direction is noisier (split-half 0.87 vs 0.96),
   which lowers its cosine for free. Fix = disattenuated transfer on disjoint halves.
   Result: refusal transfers NO better than the topic control in the shared middle (slot 18 diff ~0; slot 24 it/de -0.02/-0.03,
   hi +0.01), slightly better at slot 12 (+0.03..+0.05), worse in the last layer (hi -0.13). Zhang holds for Tiny Aya:
   the refusal direction is shared, but not SPECIALLY shared. ml/hi-Latn: not estimable (reliability of 32-pair halves ~0).
   Depth profile (panel B): everything is shared in layers ~15-24 and becomes language-specific again at 30-36 (the model
   prepares to answer in the input language).
8. Step 5 twin depth (exploratory, panel C): the extra harm signal lost by failing prompts appears in the MIDDLE layers
   (from ~12-15, peak -0.15..-0.20 of the English separation at 18-21), not at input reading and not only at output.
   Slots < 9 unreadable (division by ~0 separation). Late fade = the English ruler stops transferring, not recovery.

## Phase 3 answer in one paragraph
Tiny Aya represents harm along a direction that is shared across it/de/hi exactly as much as any other concept in its middle
layers (step 4). Along that English direction, harmful and benign stay just as separable in it/de/hi as in English (5c), so
the gap is not a squeeze. Instead, a subset of prompts loses ~15-20% of their harm signal in the middle layers (step 5, 5e)
and drops out of the zone where the model refuses (5f, post-hoc); outside English the score->refusal mapping is also noisier
(5d/5e). In ml and hi-Latn (not understood) the signal largely collapses (d' ~0.8). The registered kill-test said KILL only
because our threshold (class-mean midpoint) sat below where English actually refuses -> lesson: calibrate to behaviour.

## Hypotheses status (SCOPE A/B/C)
- A (shared, weaker trigger): best fit, in a LOCAL form (a subset of prompts loses signal along the shared direction), not
  Wang's global squeeze (out for it/de/hi).
- C (shared representation, different readout): partly (noisier score->decision outside English).
- B (language-specific): disfavoured for it/de/hi (English ruler separates them as well; transfer ~0.95). Phase 4 checks
  causally: if steering with r_hi beats r_en, B returns.

## Phase 4 consequences (decided 2026-11-10)
- Intervene at layers 18, 21, 24 (where the loss sits AND the direction is shared). English direction (+ target direction,
  + food/travel control direction as the non-safety control). Prediction to register: adding r_en rescues exactly the
  prompts that lost signal.
- Metric thresholds calibrated to English behaviour; SafeTune for the hooks; AuditKIT for capability (D9).

## Mistakes / lessons this phase (for the write-up and for me)
- Class-mean midpoint threshold != the model's decision threshold (5c KILL vs 5f).
- Cosine comparisons need matched n, equal resampling, and reliability correction (step 4 v1 had two defects).
- Ratios with noisy denominators explode at small n (ml/hi-Latn transfer CIs).
- Nearly dropped the neutral control for time; building it properly took ~1.5 h and changed the conclusion's strength.

## Explain-back (2026-11-10 14:29) — Anant
Solid: r_en + why cross-fit (1), kill-test vs post-hoc (2), 5d readings (3), 5e + caveat (4), script confound (7), hypothesis (8).
Gaps filled by Claude:
- Q2 nuance: the verdict was NOT changed; 5f is a labelled post-hoc GRID of thresholds (9/12/15/18), not one tau picked after looking.
- Q5 (missed): raw cosine is meaningless alone because any contrast on the same translated content aligns (random-sign null 0.56-0.77,
  topic control ~0.95). Three controls: matched n + equal resampling; reliability (noisier direction -> lower cosine) -> divide by
  split-half reliability; shared content -> disjoint prompt halves. Then refusal ~ topic.
- Q6 (partly): panel B early layers are not noise (cosine ~0.9 at 12); shape = rise to ~0.96 at 18, flat to 24, drop at 30-36
  (language-specific output). The "noise" was panel C slots < 9. Steer at 18-24 because: direction shared there (B), twin loss sits
  there (C), separation on its plateau (A).
