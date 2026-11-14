---
license: cc-by-nc-4.0
language: [en, it, de, hi, ml]
tags: [safety, refusal, multilingual, red-teaming, interpretability, steering]
pretty_name: "Lexsi Multilingual Refusal: generations, judgements and steering artefacts"
---

# Lexsi Multilingual Refusal: artefacts

> **Content warning.** Contains completions by a small open model (Tiny Aya Global) to harmful prompts, including compliant ones. Safety research only.

Everything behind the results of *"Is refusal shared across languages?"* (Lexsi AI Safety Hackathon, Track 2):
generations, LLM-judge labels, human audit sheets, activations, refusal and steering directions, and the steering-run outputs.
Prompts live in the benchmark repo [anantshri1/lexsi-refusal-bench](https://huggingface.co/datasets/anantshri1/lexsi-refusal-bench);
code in [github.com/anantshri1/lexsi-refusal](https://github.com/anantshri1/lexsi-refusal). The repo's `PATHS.md` is the full file
contract and `canonical_numbers.txt` holds every reported number.

**Join key everywhere: `uid`** = `{pair_id}-{h|b}-{lang}-{orig|para}` (benchmark row). Model `CohereLabs/tiny-aya-global`
(sha af89d219, transformers 5.22.0), greedy decoding, base chat template unless the condition says otherwise.

## Folder map
| folder | what | made by |
|---|---|---|
| `generations/` | `gen_tinyaya_{cond}_v1.jsonl`; cond = `base` (all 1770 rows), `sysprompt` (safety system prompt), `translate_en` (translate to English, answer), `itHT` (MultiJail human Italian). `base_v0` = throwaway smoke test | `src/generate.py` |
| `judgements/` | `judge_qwen72b-j{n}_tinyaya_{cond}_g{gen}_t{cut}.jsonl`: Qwen2.5-72B-Instruct labels (REFUSAL / COMPLIANCE / NON_ANSWER …) on the first `cut` new tokens. **j1 = headline judge**; j2 = dev only. Read with `judge.load_judgements()` (files are append-only with retries) | `src/judge.py` |
| `judgements/audit_*` | human audits by the author: `_sheet` (blind, what was labelled), `_key` (strata, unblinding), `_anant` (labels). `audit_v1` / `audit_j1check` = judge validation; `audit_overref_*` / `audit_supp_*` = over-refusal correction; `audit_reval_K1` = WildGuard re-validation under steering | `src/audit_*.py` |
| `comprehension/` | does the model understand the prompt? restatements by Tiny Aya, 1–5 scores by Qwen2.5-7B, second-pass labels, human validation sheets | Phase 1 |
| `curation/` | benchmark construction: MultiJail labelling sheets (canonical: `multijail_labelling_sheet_reviewed.csv`, `multijail_selection_log.csv`, `multijail_selected_v1.csv`), blind κ sets (**valid: `multijail_blind30_v3_anant.csv`**; v2 accidentally duplicated the seed-0 set and is invalid), benign twins (`benign_twins_reviewed.csv` is canonical), raw MT outputs incl. rejected engines (NLLB hi/ml), Sheets exports, control-set drafts | Phase 1 |
| `activations/` | `act_tinyaya_base_v1_{lang}_{orig\|para}.pt` (+ `_ctrl_` shards): dict `uids`, `acts` fp16 `[n, 2, 37, 2048]`, `seq_len`, `meta`. Axis 1: 0 = last token (`<\|START_RESPONSE\|>`), 1 = last user token. Axis 2: **slot k = output of decoder layer k−1, pre final norm** (slot 0 = embeddings). `smoke` shards are never analysed | `src/extract_activations.py` |
| `directions/` | `dir_tinyaya_refusal_last_en_v1.pt` (Phase 3 difference-of-means refusal directions); `dir_tinyaya_steer_v1.pt` (Phase 4 unit steering vectors at slots 18/21/24: cross-fitted per-fold r_en / r_hi, food/travel control, random; `delta_en` = harmful–benign gap per slot) | `src/steer_prep.py` |
| `control/` | food vs travel control set inputs and NLLB translations | `src/build_control.py` |
| `phase4/` | steering runs. `step1/` SafeTune cross-check; `wildguard/` WildGuard validation and positive control (ids and labels only, no WildGuardMix text); `screen/` dev screening (14 conditions + ablation, split, logs); `full/` full runs F1 (r_en) / F2 (r_hi) at slot 24, α/Δ = 0.2, and Hindi at 0.3; `belebele/` utility | `src/screen_job.py` etc. |

**Steering conventions.** Conditions are named `steer-{ren|rhi|ctrl|rand}-s{slot}-a{100·α/Δ}` or `ablate-ren-s24-all`. Steering used
[SafeTune](https://github.com/Lexsi-Labs/SafeTune) (commit 26f59e6) `RefusalDirectionModel`, whose `layers=[i]` indexes decoder blocks: **SafeTune index = slot − 1**.
Strength is reported as α/Δ_k, the fraction of the English harmful–benign gap at that slot. Each prompt is steered by the fold
direction that excludes its own pair (5-fold cross-fitting by `pair_id`).

## How the numbers were made
- Refusal = j1 label on the first 512 new tokens (screening: 256). j1 was validated against the author's labels (see the article).
- **Over-refusal is always human-corrected**: j1's over-refusal precision was low (~0.2), so j1-positive benign rows were labelled by
  the author and j1-negatives sampled per language × tier (`src/audit_overref.py`). WildGuard (via [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT), commit 71316da) is reported as a
  second-judge column on H_core benign twins only, never averaged with j1.
- Utility: Belebele (900 questions/language) via AuditKIT's native log-likelihood engine with the steering hook installed.

## Licence and provenance
CC BY-NC 4.0. Prompts derive from MultiJail (MIT) with NLLB-200 (CC BY-NC 4.0) and Google Translate translations. Completions
were generated by Tiny Aya Global (CC BY-NC 4.0, Cohere Labs acceptable-use policy applies); labels by Qwen2.5-72B-Instruct and
WildGuard. Human labels are by the author.

## Known limitations
- One model (3B-class); one LLM judge family for headline labels; the author is the only human annotator.
- Over-refusal rates for Hindi rest on a handful of rows (≈4–10% depending on which are counted); see the article for intervals.
- Generations are greedy and length-capped; behaviour under sampling is untested.
