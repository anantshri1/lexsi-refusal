# PATHS — read this first in every new chat

Code defines paths ONLY in `src/paths.py`. Notebooks/scripts: `from paths import *`.
Never hard-code a path string anywhere else.

## Environments
- Local (Mac M2, 8 GB): code only. Repo: anantshri1/lexsi-refusal
- Colab: all model runs. Clone repo; artefacts -> /content/drive/MyDrive/lexsi-refusal (Drive mount path verified 2026-11-02)
- HF Hub (both private until submission):
  - anantshri1/lexsi-refusal-bench (dataset): frozen benchmark ONLY
  - anantshri1/lexsi-refusal-artefacts (dataset): generations, judgements, directions, logs
- Model: CohereLabs/tiny-aya-global (fallback CohereLabs/aya-expanse-8b)

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