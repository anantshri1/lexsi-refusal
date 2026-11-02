# RECCE — toolkit findings (Phase 0, 2026-11-02)

Pinned env (Colab): Python 3.13.16 | torch 2.11.0+cu130 | transformers 5.22.0 | huggingface_hub 1.33.0 | accelerate 1.15.0
All toolkits installed from source clones; record commit per toolkit.

## SafeTune @ 26f59e6 — USABLE (core: directions + interventions)
Setup: `pip uninstall -y torchao` BEFORE import; `sys.path.insert(0, "/content/SafeTune/src")` (editable install not seen without restart).
- Explicit cohere2 (Tiny Aya) support; tests use tiny random models only → real-model behaviour still to verify.
- Tiny Aya chat template adds ~366-token system preamble (affects compute; activations read at last token after it).
- `safetune.interventions.steer.refusal_direction` implements Arditi et al. 2024 (arXiv:2406.11717):
  - `extract_refusal_direction(model, tok, harmful, harmless, config) -> (direction, layer, {layer: tensor})`
  - `RefusalDirectionModel(model, direction, mode="ablate", strength=1.0, layers=None)`; `.install()` / `.remove()` — inference-time hooks
  - `orthogonalize_weights(model, direction, ...) -> snapshots`; `restore_weights(model, snapshots)` — abliteration (weight-space)
  - `RefusalDirectionConfig`: pool_method="last_token" (ours ✓), normalize=True (we need False), pick_layer to force layer
- DO NOT use their automatic layer sweep for multilingual: `_REFUSAL_SUBSTRINGS` is English-only (33 strings), max_new_tokens=24,
  n_val=16. Their notes: unguarded sweep picked layer 1/36 on Tiny Aya; they suggest layer 24. Always set pick_layer; use our judge.
- Generic `SteeringVectorExtractor` (vector_extraction.py) defaults pool_method="mean" — would be dominated by the 366-token preamble. Avoid, or force last_token.
- To verify on real run: dict values normalised?; dict filled when pick_layer set?; exact string for additive mode.
- Feedback for Lexsi: English-only substring judge mis-scores non-English refusals as bypass; "i'm just" false positives.

## AuditKIT @ 71316da — USABLE (evaluation + judges)
- src/ layout -> sys.path.insert(0, "/content/AuditKIT/src"); install changed no pins.
- Cohere/Aya tested (tests/test_cohere_aya.py); unknown archs fail loudly.
- Custom judges: LLMJudge(choices={...}), JudgeMetric base class. Data via HF load_dataset layout.
- Guards (metrics/guard.py): llama_guard (default hf:meta-llama/Llama-Guard-3-8B), wildguard (hf:allenai/wildguard), ShieldGemma, Granite Guardian.
  Llama Guard = safe/unsafe (compliance); WildGuard also gives response_refusal (needed for over-refusal).
- Capability: ak.run_lmeval(task, model=...) via lm-eval-harness (Belebele — to verify).
- AVOID KeywordDetector for multilingual (substring).
- Compute: 8B guard ~16 GB fp16 -> 4-bit and/or separate judging session.
- To verify: Llama Guard 3 language list (expect en/it/de/hi, not ml); WildGuard multilingual reliability -> human audit.


## CuratorKIT @ dd9e835 — USABLE, narrow role (benchmark QC + provenance)
- Flat layout: sys.path.insert(0, "/content/CuratorKIT"); install changed no pins.
- NO translation/paraphrase task; 8 tasks generate training data. English-centric (pii_language="en").
- LLM backends: LiteLLM (API, incl. Cohere) or local Ollama/vLLM.
- Use for: JSONL/HF ingest, exact + embedding dedup, diversity gate, provenance (manifest.json, rejected.jsonl, dataset_card.md, checksums.txt).
- TRAPS: (1) disable toxicity filter — harmful prompts ARE the data; verify rejected.jsonl. (2) Dedup WITHIN language only — cross-lingual pairs are near-duplicates by design.
- Translations/paraphrases: ours (MT + human checks), stated as such.


## CircuitKIT — TODO