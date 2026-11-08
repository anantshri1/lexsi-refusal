"""Phase 2 — naive baselines vs base (judge j1 @ t512, pair_keep, original, en/it/de/hi).
For each language: H_core compliance base vs baseline (paired on pair_id), and the baseline's own gap vs en.
Over-refusal is NOT reported from j1 (precision ~0.2, B4.3) — human check separately."""
import sys
import pandas as pd
from rates import load, metrics, paired_diff
from paths import GENERATIONS

LANGS = ["en", "it", "de", "hi"]
cond = sys.argv[1] if len(sys.argv) > 1 else "sysprompt"
base = load()
base = base[base.lang.isin(LANGS)]
bl = load(gen=GENERATIONS / f"gen_tinyaya_{cond}_v1.jsonl")
bl = bl[bl.lang.isin(LANGS)]

print(f"\nH_core compliance (conditional), base vs {cond}:")
print(pd.DataFrame({"base": {l: metrics(base[base.lang == l])["core_comp_cond"] for l in LANGS},
                    cond:   {l: metrics(bl[bl.lang == l])["core_comp_cond"] for l in LANGS},
                    "core_na_" + cond: {l: metrics(bl[bl.lang == l])["core_na"] for l in LANGS}}).round(3).to_string())

# effect of the baseline within each language (same pairs, base vs baseline)
dd = pd.concat([base, bl.assign(lang=bl.lang + "+" + cond)], ignore_index=True)
eff = [paired_diff(dd, f"{l}+{cond}", ref=l) for l in LANGS]
print(f"\n{cond} - base, per language (negative = fewer compliances):")
print(pd.DataFrame(eff).round(3).to_string(index=False))

gap = [paired_diff(bl, l, ref="en") for l in LANGS[1:]]
print(f"\nGap vs en UNDER {cond} (compare with base: it +0.082, de +0.105, hi +0.174):")
print(pd.DataFrame(gap).round(3).to_string(index=False))
