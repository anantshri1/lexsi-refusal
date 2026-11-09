# PATHS, RECCE — read this first in every new chat

Code defines paths ONLY in `src/paths.py`. Notebooks/scripts: `from paths import *`.
Never hard-code a path string anywhere else.

## Environments
- Local (Mac M2, 8 GB): code only. Repo: anantshri1/lexsi-refusal
- Colab: all model runs. Clone repo; artefacts -> /content/drive/MyDrive/lexsi-refusal (Drive mount path verified 2026-11-02)
- HF Hub (both private until submission):
  - anantshri1/lexsi-refusal-bench (dataset): frozen benchmark ONLY
  - anantshri1/lexsi-refusal-artefacts (dataset): generations, judgements, directions, logs
- Model: CohereLabs/tiny-aya-global (fallback CohereLabs/aya-expanse-8b)

### Sync rule (Mac <-> Drive) — added 2026-11-03 after a FileNotFoundError
- Drive (/content/drive/MyDrive/lexsi-refusal) is the SOURCE OF TRUTH for DATA_RAW / DATA_INTERIM / outputs.
- Mac data/interim is a MIRROR for Claude to read/write. Anything Claude writes there
  must be uploaded to Drive before Colab uses it; anything Colab writes must be
  downloaded to Mac before Claude can check it.
- REVERSED for judgements (Phase 2): the API judge runs on the Mac, so JUDGEMENTS is written on the Mac
  first -> upload outputs/judgements/ to Drive after each run.
- Files made on Mac (Claude or Sheets) so far: checker_flags_v1.csv, gsheets_input.csv,
  gsheets_fix_input.csv, mt_gsheets.csv, mt_gsheets_fix.csv, benign_twins_claude.csv,
  multijail_labelling_sheet_claude(_v2).csv

## Roots
| Name | Local | Colab | In git? |
|---|---|---|---|
| REPO_ROOT | repo | /content/lexsi-refusal (clone) | — |
| ARTEFACT_ROOT | REPO_ROOT | /content/drive/MyDrive/lexsi-refusal | — |

## Paths
| Constant | Location | In git? | Contents |
|---|---|---|---|
| DATA_RAW | ARTEFACT_ROOT/data/raw | no | untouched downloads (MultiJail, XSTest) |
| DATA_INTERIM | ARTEFACT_ROOT/data/interim | no | translations awaiting checks |
| DATA_FROZEN | REPO_ROOT/data/frozen | YES | benchmark_v1.jsonl — read-only once frozen |
| GENERATIONS | ARTEFACT_ROOT/outputs/generations | no | model outputs, one file per condition |
| JUDGEMENTS | ARTEFACT_ROOT/outputs/judgements | no | judge labels |
| ACTIVATIONS | ARTEFACT_ROOT/outputs/activations | no | cached activations (.pt) — expensive |
| DIRECTIONS | ARTEFACT_ROOT/outputs/directions | no | refusal directions r_{l,k} |
| FIGURES | REPO_ROOT/figures | YES | plots |
| CANONICAL | REPO_ROOT/canonical_numbers.txt | YES | ground-truth numbers |
| COMPREHENSION | ARTEFACT_ROOT/outputs/comprehension | no | restatements + judge scores, keyed by uid |

## File naming (to fill as we go)
- Generations: GENERATIONS/gen_{model}_{condition}_v{n}.jsonl, e.g. gen_tinyaya_base_v1.jsonl
  model in {tinyaya, expanse}; condition in {base, sysprompt, translate_en}
  fields: uid, model, condition, max_new_tokens, n_new_tokens, hit_eos, response, ids (generated token ids, EOS excluded; for exact B8 truncation)
- Judgements: JUDGEMENTS/judge_{judge}-j{n}_{model}_{condition}_g{n}_t{cut}.jsonl
  e.g. judge_qwen72b-j1_tinyaya_base_g1_t512.jsonl. j = judge version (bump when src/judge.py SYSTEM or
  JUDGE_RUBRIC.md Part A changes); g = generation version; t = response cut in new tokens (B8: 512/256/128 all rows, 64 audit only).
  fields: uid, judge, cut, truncated, raw, reason, label, response_lang, has_disclaimer, offers_alternative, restates_prompt, raw_clean
  label None = unparseable judge output (kept, never mapped). Built by src/judge.py run_judge().
- Phase 2 generation files (GENERATIONS): gen_tinyaya_base_v1 (all 1770), gen_tinyaya_sysprompt_v1 (en/it/de/hi pair_keep, 1032),
  gen_tinyaya_itHT_v1 (100 H_core, MultiJail human Italian, uid {pair_id}-h-itHT-orig), gen_tinyaya_to_en_v1 (step-1 translations,
  it/de/hi 768), gen_tinyaya_translate_en_v1 (answers to cleaned translations, same uids). gen_tinyaya_base_v0 = throwaway test.
- Phase 2 judgement files (JUDGEMENTS): judge_qwen72b-j1_* for base (t512/256/128 all; t64 audit), sysprompt, itHT, translate_en (t512);
  judge_qwen72b-j2_* = DEV ONLY (audit rows). Always read via judge.load_judgements(); retry unparsed rows (missingness biased).
- Human audits (JUDGEMENTS; Anant): audit_v1_anant (72, j1 dev, adjudication in src/audit_kappa.py ADJ), audit_j1check_anant (112, B4.3),
  audit_overref_{sysprompt,translate_en}_anant. Each has _sheet (blind) and _key (strata). Never re-draw.
- Activations (Phase 3; built by src/extract_activations.py as an HF Job, NOT Colab):
  HF_ARTEFACT_REPO/activations/act_tinyaya_base_{tag}_{lang}_{var}.pt, tag v1 (full) | smoke (--limit, never analysed);
  lang in {en,it,de,hi,ml,hi-Latn}, var in {orig,para} (para: en/hi/ml only) -> 9 v1 shards (6x266 + 3x58 = 1770 rows).
  Each .pt = dict(uids [n], acts fp16 [n, 2, 37, 2048], seq_len [n], meta).
  acts axis 1: 0 = last token (<|START_RESPONSE|>, pos -1), 1 = last user-content token (pos -5; gap 4 verified on all rows).
  acts axis 2: slot 0 = embeddings, slot k = output of decoder layer k-1, ALL PRE final norm (own hooks; HF hidden_states[-1]
  is POST-norm -> never use output_hidden_states for directions; smoke 2026-11-09: slots 0-35 identical, slot 36 differs).
  Base template = generate.build_messages(text, "base"); model sha af89d219; transformers 5.22.0.
  SOURCE OF TRUTH = HF artefact repo (no Drive copy). Mac mirror = ACTIVATIONS (download with hf_hub_download).
- Control set (Phase 3 step 4; neutral TOPIC contrast food vs travel, 64 drafted pairs; uid {ctrlNNN}-{f|t}-{lang}-orig):
  DATA_INTERIM/control_neutral_en_claude.csv (draft, wide) -> control_neutral_en_reviewed.csv (Anant; keep blank=keep, n=drop;
  edit_food/edit_travel override) -> [build_control.py prep] control_neutral_en_v1.csv (long) + gsheets_ctrl_input.csv
  -> HF ART_REPO control/control_neutral_en_v1.csv -> [HF Job translate_control_job.py] control/mt_ctrl_itde_nllb_v1.csv
  (download to DATA_INTERIM) + Sheets export DATA_INTERIM/mt_gsheets_ctrl.csv (pair_id, side, mt_input, hi, ml)
  -> [build_control.py build] DATA_FROZEN/control_neutral_v1.jsonl (in git; also upload to ART_REPO control/)
  -> [extract_activations.py --control] activations/act_tinyaya_base_v1_ctrl_{lang}_orig.pt (6 shards).
- Directions: DIRECTIONS/dir_tinyaya_{contrast}_{pos}_{lang}_v1.pt, [37, 2048], contrast in {refusal, control}, pos in {last, user}
  (computed on the Mac from activations; TBD details in Phase 3 block 5)

## Rules
1. A frozen file is never edited; changes = new version (v2), old kept.
2. Every compute-heavy result is saved before it's used.
3. Any number quoted in the write-up must appear in canonical_numbers.txt.

## Model facts (Tiny Aya Global, measured 2026-11-02 on Colab T4)
- model_type: cohere2 | layers: 36 | hidden_size: 2048 | fp16 on GPU: 6.7 GB
- Direction tensor per language: [36, 2048] (one r_{l,k} per layer)
- transformers 5.x: apply_chat_template needs return_dict=True; pass **enc to generate

## Benchmark schema (v1, frozen 2026-11-03)
File: DATA_FROZEN/benchmark_v1.jsonl — long format, one row = one prompt × lang × variant. Never edited; changes -> v2.
Join key for EVERY downstream output (generations, judgements, activations, comprehension): `uid`.

| field | type | values / example |
|---|---|---|
| uid | str | `{pair_id}-{h|b}-{lang}-{orig|para}` e.g. `mj009-h-hi-orig` |
| pair_id | str | `mj` + MultiJail id zero-padded to 3, e.g. `mj009` |
| mj_id | int/null | MultiJail id; null for benign |
| polarity | str | harmful / benign |
| tier | str | H_core / H_bait (benign inherits its pair's tier) |
| primary_tag | str | first MultiJail tag — coverage only, never per-category reporting |
| lang | str | en / it / de / hi / ml / hi-Latn |
| variant | str | original / paraphrase |
| text | str | prompt |
| origin | str | multijail_en / mt:<engine> / translit / authored / paraphrase:<who> |
| validation | str | source / reviewed:anant / bt+llm_check / derived (hi-Latn) |
| bt_text | str/null | English back-translation (null for en) |
| notes | str | |

Counts (verified 2026-11-06): 133 pairs (100 H_core + 33 H_bait) × 2 polarities × 6 langs = 1596
        + paraphrase: 29 pairs × 2 × {en, hi, ml} = 174
        = 1770 rows
Hinglish dropped (Phase 1); hi-Latn = mechanical transliteration of hi. Paraphrase subset: 30 pairs drawn (seed 0), #169 dropped -> 29.

Interim files (DATA_INTERIM, not frozen):
- multijail_labelling_sheet_reviewed.csv — curation labels (H_core/H_bait/B/A/X)
- multijail_selection_log.csv — all 315 rows + selected flag (drop log)
- multijail_selected_v1.csv — 133 selected + it_multijail (MT reference, NOT a benchmark row)
NOT in the frozen file: comprehension results (outputs/, keyed by uid).