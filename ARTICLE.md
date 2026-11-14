# Steering a multilingual model's safety from the inside

*Lexsi AI Safety & Alignment Hackathon, Track 2 · Anant Shri · November 2026*

> **Content warning.** This article discusses harmful prompts (violence, self-harm, hate, sexual exploitation) and a model's compliant answers to them. No harmful completions are quoted.

**TL;DR**
- **There is a gap.** On harmful requests it understood, Tiny Aya Global complies 5% of the time in English, 13% in Italian, 16% in German and **23% in Hindi** (Hindi +17 pp over English, 95% CI 9.5–25.8). Over harmless prompts it stays calm in every language, so this is under-refusal, not general caution.
- **The mechanism is shared; the signal is lost.** The English refusal direction separates harmful from harmless prompts in Italian, German and Hindi almost as well as in English. What fails is local: some prompts lose part of their harm signal in the middle layers and fall below the point where the model refuses.
- **One internal push closes most of the gap.** Adding the English refusal direction at one layer cuts Hindi compliance **22.6% → 8.6%**, matching "translate to English first" (8.7%) in **one pass, answering in Hindi**, and beating a safety system prompt (15.1%). Matched control directions do nothing. The cost: 2–4% over-refusal on harmless requests, more pushback on harmless statements, and under 1 pp of reading comprehension.

**Links:** code [github.com/anantshri1/lexsi-refusal](https://github.com/anantshri1/lexsi-refusal) · benchmark [anantshri1/lexsi-refusal-bench](https://huggingface.co/datasets/anantshri1/lexsi-refusal-bench) · all generations, judgements, human audits, activations and directions [anantshri1/lexsi-refusal-artefacts](https://huggingface.co/datasets/anantshri1/lexsi-refusal-artefacts).

## 1. Why this matters

Safety training is mostly done in English. A model that refuses a harmful request in English may help with the same request in Hindi, which leaves most of the world's users with weaker protection. The standard fix is to translate the request into English first and answer that. It works, but it costs a second model pass and it answers in English, which is not what a Hindi speaker asked for.

So we asked a mechanistic question with a practical edge: **is refusal one shared mechanism that simply fires more weakly outside English, or does each language have its own?** And if it is shared, can we switch the English one back on from the inside, in a single pass, without making the model refuse harmless requests or forget how to read?

We framed three hypotheses before looking at any internals:
- **A, shared:** one refusal mechanism, triggered more weakly in other languages.
- **B, language-specific:** each language has its own, weaker mechanism.
- **C, shared representation, different readout:** the model represents harm the same way everywhere but acts on it differently.

Each predicts something different for a steering experiment. Under A, pushing along the English refusal direction should restore refusal in Hindi. Under B, a Hindi-specific direction should do better. Under C, the push should help but the language's own readout should limit it.

## 2. Setup

**Model.** [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters, 36 layers), greedy decoding, its own chat template.

**Benchmark.** We started from the 315 English prompts of MultiJail (Deng et al., 2023) and split them into two tiers. **H_core** (100 prompts) are harmful *requests*, where refusal is correct. **H_bait** (33) are hateful *statements*, where declining or pushing back is correct. Every harmful prompt has a **benign twin**: same topic, as much shared wording as possible, clearly harmless intent ("how do I kill a Python process"). Pairs let us ask whether a language loses the *difference* between harmful and harmless, not just whether it refuses less overall. We translated the 133 pairs into Italian and German (NLLB-200), Hindi and Malayalam (Google Translate), plus a mechanical romanisation of the Hindi. Every harmful back-translation was checked for softening; Hindi was read in full by a native speaker. The frozen set has 1,770 rows.

**Comprehension control.** A model cannot refuse what it does not understand, and a low compliance rate in a language it cannot read means nothing. Tiny Aya restated every benign prompt in English, and a separate judge scored the restatement. We keep a pair only if both sides were understood. This mattered: Tiny Aya understood 96–100% of benign prompts in English, Italian, German and Hindi, but only 61% in Malayalam and 26% in romanised Hindi. Malayalam turned out not to be in the model's listed training languages at all. Both are reported as descriptive only; the comparisons below use English, Italian, German and Hindi.

**Judging, and why over-refusal is always checked by hand.** Responses were labelled by Qwen2.5-72B-Instruct with a written rubric, reading the response in its own language. We validated it against blind human labels. On harmful prompts it was good enough (precision 0.76–0.92 on the failures that drive the headline, and the gap survived correcting for its errors). On harmless prompts it was not: **only about 1 in 5 of its "the model refused this harmless prompt" calls were right.** So every over-refusal number in this article is human-corrected: all of the judge's flags plus a stratified sample of its non-flags were labelled by hand, and the rate is re-estimated per language and tier.

**Statistics.** 95% intervals are pair bootstrap (2,000 resamples), paired against English on shared pairs. Every prediction was written down before its run, including the ones that failed ([PREDICTIONS.md](https://github.com/anantshri1/lexsi-refusal/blob/main/PREDICTIONS.md)).

## 3. Result 1: the gap is real, and translation is the bar

| H_core compliance (%) | en | it | de | hi |
|---|---|---|---|---|
| base model | 5.1 | 13.3 | 15.6 | **22.6** |
| + safety system prompt | 5.1 | 9.2 | 14.6 | 15.1 |
| translate to English first (2 passes) | 5.1 | 6.1 | 9.4 | 8.7 |

Hindi complies **+17.4 pp** more than English on the same requests (95% CI 9.5–25.8; Italian +8.2, German +10.5). Over-refusal on harmless twins, human-corrected, stays at 0–5.5% in every language, so this is **under-refusal of harmful requests**, not a model that is cautious in English and relaxed elsewhere. Two checks ruled out boring explanations: MultiJail's *human* Italian gives the same compliance as our machine Italian (difference +4.1 pp, CI includes 0), so the gap is not a translation artefact; and the gap survived correcting the judge's errors.

The two cheap fixes set the bar. A one-line system prompt ("Refuse harmful requests in any language") helps Hindi a little (−7.5 pp). **Translating to English first nearly closes the gap** (Hindi 8.7%, within 3.4 pp of English) but needs two passes and answers in English. That is the bar for anything internal.

## 4. Result 2: the mechanism is shared, but some prompts lose their harm signal

We took the difference of mean activations between harmful requests and their benign twins at the last prompt token, per layer: the **refusal direction** (Arditi et al., 2024), cross-fitted so that no prompt is scored with a direction built from itself.

![Phase 3 geometry](https://raw.githubusercontent.com/anantshri1/lexsi-refusal/main/figures/p3_geometry_v1.png)

*(A) Separation of harmful vs benign along the English direction (d′, held out) by layer. (B) How much a direction learned in one language transfers to another, for refusal (solid) and for a neutral food-vs-travel topic (dotted). (C) The extra harm signal lost by prompts that fail in a language, relative to their English twin.*

Three findings:
1. **The English direction works in the other languages.** At layer slot 24, it separates harmful from benign almost perfectly in all four (AUROC 0.99 in each; d′ 4.6 English, 4.2 Italian, 3.9 German, 4.1 Hindi). Malayalam and romanised Hindi collapse (d′ ≈ 0.8–0.9 on all prompts), consistent with the model not understanding them.
2. **It is shared, but not specially shared.** A Hindi refusal direction is 0.90 aligned with the English one after correcting for noise. But a neutral concept, food vs travel, transfers just as well (0.89). Refusal rides on the same cross-lingual representation as everything else, not on a dedicated English-only circuit. That rules against B for these languages.
3. **The failures are local.** If the gap came from the whole Hindi distribution being squeezed, every prompt would sit lower. Instead, the Hindi prompts the model *complies* with lose much more of their harm signal relative to their English twin (median −8.2 units at slot 24) than Hindi prompts it still refuses (−3.4). A subset of prompts loses part of its signal in the middle layers and drops out of the zone where the model refuses.

**A failure we kept.** Our pre-registered kill-test for this phase said *KILL*: at a refusal threshold set midway between the harmful and benign means, Hindi and English trigger equally often. The threshold was the problem: it sat well below where English actually refuses. At thresholds calibrated to English behaviour, the Hindi gap appears (+12 to +16 pp, CIs above 0). We report the registered verdict as it was, and the lesson that changed how we did Phase 4: **calibrate thresholds to behaviour, not to class means.**

So by the end of Phase 3 the picture was A in a local form, with some C: the shared direction is there, some prompts fall short along it, and the mapping from score to refusal is noisier outside English. That makes a clear causal prediction: **push along the English direction, and the prompts that fell short should come back.**

## 5. Result 3: one push along the English direction closes most of the gap

**The intervention.** We used [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s `RefusalDirectionModel` in *steer* mode: at one layer, add α·r to the residual stream at every position, prompt and generated tokens alike. We first checked that SafeTune extracts the same direction we computed ourselves (cosine 1.000) and that its push moves the score at that layer by exactly α. We quote the push size as **α/Δ**, the fraction of the English harmful–benign separation at that layer, so that "0.2" means "a fifth of the way from a typical harmless prompt to a typical harmful one". Every prompt is steered with the fold direction that excludes its own pair.

**Kill-test first.** Before scaling anything, we ran the experiment that could make the headline disappear: on 55 development pairs, push Hindi prompts by the same amount along (i) the English refusal direction, (ii) the food-vs-travel control direction and (iii) a random direction.

| Hindi compliance, dev pairs (base 23.6%) | change |
|---|---|
| English refusal direction | **−16.4 pp** [−27.3, −7.3] |
| food vs travel control | +3.6 pp [−5.5, 12.7] |
| random direction | −1.8 pp [−7.3, 3.6] |

Only the refusal direction does anything. A cheap forecast we made before any GPU run (shift each prompt's score by α and read refusal off each language's fitted score→refusal curve) predicted the dose–response at slot 24 closely (15.3 / 10.4 / 6.9% forecast vs 16.4 / 10.9 / 7.3% observed at α/Δ = 0.1 / 0.2 / 0.3).

**Necessity.** Steering shows the direction is *sufficient* to cause refusal. To show it is *necessary*, we removed it at every layer (all-layer ablation, also via SafeTune; removing it at one layer is not enough because later layers rebuild it). Compliance jumped to 40% in English, 44% in Hindi and 51% in German. The English safety advantage mostly disappears with the direction: the gap lives along it.

**Choosing the setting.** On the development pairs we screened three layers (slots 18, 21, 24) at three strengths, with over-refusal screened by WildGuard (see Section 7 for why only on part of the data). Slot 24 at α/Δ = 0.2 was the cheapest setting in over-refusal among those that reached the bar; its neighbours were within noise. Development data chose only the layer and the strength.

**The full run.** We then steered all 1,032 understood harmful and benign prompts in English, Italian, German and Hindi at that setting, with the English direction (F1) and, as the test of hypothesis B, the Hindi direction (F2). Same judge, 512 tokens, human-corrected over-refusal, and Belebele reading comprehension (900 questions per language) through [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)'s native log-likelihood engine with the steering hook installed.

**Before and after (Hindi).**

| condition | passes / answer language | harmful compliance (H_core) | over-refusal, harmless requests | over-refusal, all harmless twins | Belebele accuracy |
|---|---|---|---|---|---|
| base | 1 / Hindi | 22.6% | — | 0.0% | 57.8% |
| safety system prompt | 1 / Hindi | 15.1% | — | 0.8% | — |
| translate to English | 2 / English | 8.7% | — | 5.4% | — |
| **steer, English direction (F1)** | **1 / Hindi** | **8.6%** | **2.2%** | **9.6%** | **57.7%** |
| steer, Hindi direction (F2) | 1 / Hindi | 9.7% | 2.2% | 9.8% | 57.8% |

*Over-refusal is human-corrected throughout; "harmless requests" are benign twins of H_core, "all harmless twins" adds the benign twins of H_bait statements. Baselines were not split by tier.*

In Hindi, steering cuts compliance by **14.0 pp** [−21.7, −6.5], beats the system prompt by 6.5 pp [−12.9, 0.0] (borderline), and is **indistinguishable from translating to English** (−0.1 pp [−5.3, 5.3]), with one pass and a Hindi answer. Across languages compliance falls to **3.0 / 7.1 / 9.4 / 8.6%** (en / it / de / hi; base 5.1 / 13.3 / 15.6 / 22.6), and the Hindi–English gap shrinks from +17.4 to +5.4 pp. Reading comprehension moves by 0.1–0.8 pp in every language (8–18 of 900 answers change).

![Trade-off](https://raw.githubusercontent.com/anantshri1/lexsi-refusal/main/figures/p4_tradeoff_v1.png)

*(A) Hindi compliance against human-corrected over-refusal for every condition in the full run. (B) Development dose–response by layer, with the forecast and the kill-test controls.*

**The cost, honestly.** On harmless *requests* the price is small and even: 2–4% over-refusal in every language (Hindi 2.2%). WildGuard, as an independent second judge on the same rows, agrees (1–5%). The price is paid on harmless *statements*: the benign twins of hateful statements get more pushback (Hindi 9.7% on that tier), which brings total Hindi over-refusal to 9.6%, against 5.4% for translate-to-English. That number rests on a handful of rows (see Section 7). English also gets a little more cautious (over-refusal 4.5%), which suggests the obvious deployment: **steer only non-English inputs.** Composing that policy from the runs we already had costs nothing in English (back to base: 5.1% compliance, 2.7% over-refusal) and keeps the Hindi gains.

**What the push fixes, and what it cannot.** Split the Hindi failures by what the model does with the English twin. Where English *refuses* and Hindi complies (the signal-loss cases from Section 4), steering rescues **13 of 17**. Where English *also* complies, it rescues **1 of 4**. Steering restores harm signal lost on the way into Hindi; it cannot create recognition that English never had. That also explains what happened when we pushed Hindi harder (α/Δ = 0.3, post hoc): compliance did not improve (9.7%, +1.1 pp [−2.2, 4.5]). The remaining failures are mostly the kind English fails on too.

## 6. What it means

- **Hypothesis A, in a local form, fits best.** A Hindi-specific direction does no better than the English one (F2 − F1 in Hindi +1.1 pp [−2.3, 4.7]); control and random directions do nothing; removing the English direction removes refusal everywhere and mostly erases the gap. Refusal in Italian, German and Hindi runs through the same direction as in English.
- **Some of C remains.** Outside English, the score→refusal curve is higher and softer (in Hindi, its midpoint is 13.6 vs 9.75 in English, and it is twice as wide), which is part of why a modest push works at all. C's sharper prediction, that steering in the target language beats English, did not come true.
- **B is disfavoured, with a caveat.** At slot 24 the English and Hindi directions are 0.90 aligned, so a head-to-head between them has limited power to tell them apart.
- **Practically:** the multilingual safety gap in this model is not a missing Hindi safety mechanism. The English one is shared and can be re-engaged from inside, in one pass, at a measurable and mostly small cost. The failures it cannot fix are the ones English shares.

## Related work, and what is new here

Three recent papers frame this question. **Wang et al. (2025)** find the English refusal direction is near-universal across 14 languages, but that harmful and harmless prompts separate less well outside English: a shared direction with a weaker trigger (our A). **Aziz et al. (2026)** find the opposite on separation (it is nearly as good in low-resource languages) and locate the failure in the decision, fixed by recalibrating a readout (our C). **Zhang (2026)**, the closest to us (MultiJail, difference-of-means directions), shows that high cross-lingual cosine is generic: a non-safety contrast is just as aligned, so cosine alone cannot show a shared safety mechanism.

Our results sit between Wang and Aziz: separation along the English direction is almost as good in Italian, German and Hindi (with Aziz), but the failures are prompts that lose harm signal along it (a local version of Wang), and the readout is softer (some of Aziz's C). What we add: a multilingual-first model; a **comprehension control**, so no prompt the model could not read counts as a safety failure; **human-corrected judge rates** and a human-Italian translation control; **translate-to-English as the bar** for any internal fix; **control and random directions in the causal tests**, which Zhang notes were missing; paired twins that locate where each failing prompt loses its signal; and a utility and over-refusal accounting of the fix.

## 7. What went wrong, and what we did about it

We kept a log of failures because each changed a decision.

- **Our cheap over-refusal judge failed its registered test.** We planned to screen over-refusal with WildGuard (run through AuditKIT) instead of hand labels. Against our labels its recall was 0.49, below our pre-set bar. Splitting by tier explained why: on harmless *requests* recall was 0.95, on harmless *statements* 0. WildGuard does not count pushback on a statement as refusal. We kept the FAIL on record, logged a narrower post-hoc rule before using it (screen only on request twins), re-validated it blind on steered outputs (precision 9/10), and never let a final number depend on it.
- **WildGuard's harmful-response flag never fired** on our compliant outputs (0 of 51). A positive control on WildGuard's own test set (7–8 of 10 flagged, 0 of 10 false alarms) showed the pipeline works: it is a genuine threshold or language effect, reported as a finding.
- **A toolkit cache bug.** Our first Belebele run gave steered results identical to base to four decimals: AuditKIT's disk cache returned the base model's answers for the hooked model. A fresh cache per condition fixed it. Anyone evaluating hooked models with it should do the same.
- **The Phase 3 kill-test said KILL** for a reason that turned out to be our threshold (Section 4). Kept as registered.
- **The held-out test folds were inconclusive.** The 34 pairs never used for any choice had only 5 Hindi failures to begin with (14.7%, vs 22.6% overall); steering rescued 1 and created none (−2.9 pp [−8.8, 0.0]). The direction itself was cross-fitted everywhere, and development data chose only layer and strength, but we cannot claim an independent replication.
- **The Hindi over-refusal number moved.** Our first estimate for F1 (12.8%) scaled up a single refusal found among four sampled rows. Labelling all 19 rows in that stratum gave 9.6%. We applied the same census to F2 (2.4% → 9.8%), not only to the number we disliked.
- **Judge provider change.** Rate limits forced a switch of inference provider for the full runs. We switched only after a pre-registered agreement check on 100 rows (agreement 0.95 overall, 0.97 on the headline label).

## 8. Limitations

One 3B-class model. One family of LLM judge for the headline labels, with a single human annotator (the author; native Hindi, reads Italian, German at A2/B1) for validation and over-refusal. Malayalam and romanised Hindi were largely not understood by the model and are excluded from comparisons, so we say nothing about truly low-resource languages. Hindi over-refusal estimates rest on a handful of rows. Generation is greedy and capped at 512 tokens. Steering was tested only against direct requests, not against adversarial jailbreaks.

## 9. Reproducing, cost and tools

**Lexsi toolkits.** SafeTune extracted every steering vector (checked against ours, cosine ≥ 0.9999) and ran all steering and ablation. AuditKIT ran WildGuard and the Belebele evaluation with the steering hook. CuratorKIT and CircuitKIT were assessed but not used in the results. Everything else (benchmark construction, judging, geometry, statistics) is our own code.

**Compute and cost.** All GPU work ran as Hugging Face Jobs (single-file scripts on a T4 or L4), costing cents per run; extracting every activation for the full benchmark cost USD 0.04. The whole project used about **USD 11 of Hugging Face credits**, most of it on the 72B judge through Inference Providers. At inference time, steering adds one vector addition at one layer; translate-to-English needs a second full generation.

**Everything is public.** Code: [github.com/anantshri1/lexsi-refusal](https://github.com/anantshri1/lexsi-refusal) (phase summaries, pre-registered predictions, and a single file of every reported number). Benchmark with dataset card: [anantshri1/lexsi-refusal-bench](https://huggingface.co/datasets/anantshri1/lexsi-refusal-bench). Generations, judge labels, human audit sheets, activations and directions: [anantshri1/lexsi-refusal-artefacts](https://huggingface.co/datasets/anantshri1/lexsi-refusal-artefacts).

## Acknowledgements

This project was done with Claude (Anthropic) as a research partner throughout: planning, design discussions, code drafts, reviews and writing. All runs, human labels and final decisions are the author's.

## References

- Arditi, A. et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717.
- Deng, Y. et al. (2023). *Multilingual Jailbreak Challenges in Large Language Models.* arXiv:2310.06474.
- Han, S. et al. (2024). *WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.* arXiv:2406.18495.
- Aziz, Hanif, Koto (2026). *Low-Resource Safety Failures Are Action Failures, Not Representation Failures.* arXiv:2606.01196.
- Wang, Wang, Liu, Schütze, Plank (2025). *Refusal Direction is Universal Across Safety-Aligned Languages.* NeurIPS. arXiv:2505.17306.
- Zhang (2026). *Does Safety Live in a Shared Multilingual Subspace? Cross-Lingual (Non-)Transfer of the Refusal Direction.* MRL 2026.
- Bandarkar, L. et al. (2024). *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.* ACL.
