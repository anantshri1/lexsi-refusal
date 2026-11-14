---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
language: [en, it, de, hi]
tags: [safety, refusal, activation-steering, multilingual, safetune]
---

# Tiny Aya Global: refusal steering vectors (no weights)

**Not a fine-tuned model.** This repo holds the activation-steering vectors that close most of Tiny Aya Global's multilingual
refusal gap at inference time. The base model is unchanged; you add one vector at one layer with
[SafeTune](https://github.com/Lexsi-Labs/SafeTune) (commit 26f59e6).

**Result (Hindi, harmful requests the model understood):** compliance 22.6% → 8.6% with one pass, answering in Hindi, equal to
translating to English first (8.7%). Over-refusal on harmless requests 2.2%; Belebele reading comprehension −0.1 pp.
Full write-up and all numbers: [github.com/anantshri1/lexsi-refusal](https://github.com/anantshri1/lexsi-refusal).

## What is in `steer_tinyaya_s24_v1.pt`
| key | meaning |
|---|---|
| `r_en`, `r_hi` | unit refusal directions (difference of means, H_core harmful − benign twin, at `<\|START_RESPONSE\|>`), English / Hindi, all pairs, residual stream at **slot 24** |
| `r_en_folds`, `r_hi_folds`, `fold_map` | the 5 cross-fitted versions used in the evaluation (each prompt steered by the fold that excludes its pair); min cosine with `r_en` = {FOLD_COS_MIN} |
| `slot`, `safetune_layer` | 24 and 23: slot k = output of decoder layer k−1, and SafeTune's `layers=[i]` indexes decoder blocks |
| `delta_en`, `alpha_over_delta`, `alpha` | English harmful−benign separation at slot 24, the chosen fraction 0.2, and the strength α = 0.2·Δ |

## Use
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetune.interventions.steer.refusal_direction import RefusalDirectionModel

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", torch_dtype=torch.float16, device_map="cuda")
v = torch.load(hf_hub_download("anantshri1/tiny-aya-global-refusal-steer", "steer_tinyaya_s24_v1.pt"))
hook = RefusalDirectionModel(model, v["r_en"].to(model.device, model.dtype), mode="steer",
                             strength=v["alpha"], layers=[v["safetune_layer"]])
hook.install()          # every forward pass now adds alpha * r_en to the residual stream after decoder layer 23
# ... generate as usual (greedy, the model's chat template) ...
hook.remove()
```
Recommended deployment: steer **non-English inputs only** (English gets slightly more cautious when steered and gains nothing).

## Limits
Evaluated on one model, direct (non-adversarial) MultiJail requests in en/it/de/hi, greedy decoding, 512 tokens. Not tested
against jailbreaks. More pushback on harmless *statements* (Hindi over-refusal 9.6% across all benign twins). Licence follows the
base model and the benchmark (CC BY-NC 4.0).
