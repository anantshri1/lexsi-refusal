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
- Generations: TBD in Phase 2
- Directions: TBD in Phase 3

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
| lang | str | en / it / de / hi / ml / hinglish |
| variant | str | original / paraphrase |
| text | str | prompt |
| origin | str | multijail_en / mt:<engine> / authored / paraphrase:<who> |
| validation | str | source / native / back_translation / spot_check |
| bt_text | str/null | English back-translation (null for en) |
| notes | str | |

Counts: 133 pairs (100 H_core + 33 H_bait) × 2 polarities × 5 langs = 1330
        + hinglish: 30 pairs × 2 = 60 (variant=original)
        + paraphrase: same 30 pairs × 2 × {en, hi, ml} = 180
        = 1570 rows
Variant subset: 30 pairs, seed 0, stratified ~23 H_core + 7 H_bait.

Interim files (DATA_INTERIM, not frozen):
- multijail_labelling_sheet_reviewed.csv — curation labels (H_core/H_bait/B/A/X)
- multijail_selection_log.csv — all 315 rows + selected flag (drop log)
- multijail_selected_v1.csv — 133 selected + it_multijail (MT reference, NOT a benchmark row)
NOT in the frozen file: comprehension results (outputs/, keyed by uid).