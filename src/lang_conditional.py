"""Phase 4 (post-hoc, exploratory; decided 2026-11-13 after the full runs): LANGUAGE-CONDITIONAL steering, composed at zero cost.
Rule: steer only non-English prompts (it/de/hi at slot 24, alpha/Delta 0.2 = F1); English unsteered. Strength 0 == no hook (step 1 D),
and decoding is greedy, so the English rows of this policy ARE the base run and the it/de/hi rows ARE F1 -> no new generation,
judging or labelling. Assumes an ORACLE language label (deployment would need language ID). Motivation: F1's over-refusal cost
landed on English (dev and full), while English compliance was already near floor.
Prints: H_core compliance + gap vs en, over-refusal (human-corrected) and Belebele for base / F1 / this policy."""
import sys; sys.path.insert(0, "src")
import json
import pandas as pd
from huggingface_hub import hf_hub_download
from rates import load, metrics, paired_diff
from paths import GENERATIONS, PHASE4, HF_ARTEFACT_REPO

LANGS = ["en", "it", "de", "hi"]
base = load(gen=GENERATIONS / "gen_tinyaya_base_v1.jsonl"); base = base[base.lang.isin(LANGS)]
f1 = load(gen=PHASE4 / "full" / "gen_tinyaya_steer-ren-s24-a020_full_v1.jsonl"); f1 = f1[f1.lang.isin(LANGS)]
lc = pd.concat([base[base.lang == "en"], f1[f1.lang != "en"]], ignore_index=True)       # the composed policy
hi3 = load(gen=PHASE4 / "full" / "gen_tinyaya_steer-ren-s24-a030_fullhi_v1.jsonl")    # P4.9: hi rows at alpha/Delta 0.3
lc3 = pd.concat([base[base.lang == "en"], f1[f1.lang.isin(["it", "de"])], hi3[hi3.lang == "hi"]], ignore_index=True)
D = {"base": base, "F1 (all langs)": f1, "lang-conditional (en off)": lc, "lang-cond, hi 0.3 (P4.9)": lc3}
HI030_OVERREF = 4.0       # audit_overref correct steer-ren-s24-a030_fullhi, hi all (2026-11-13 20:50; H_core 1.1, H_bait 12.9)

print("H_core compliance (%):")
print(pd.DataFrame({k: {l: round(100 * metrics(d[d.lang == l])["core_comp_cond"], 1) for l in LANGS} for k, d in D.items()}).T.to_string())
print("\ngap vs en (pp, 95% CI):")
for k, d in D.items():
    g = [paired_diff(d, l, ref="en") for l in LANGS[1:]]
    print(f"  {k:26s} " + " | ".join(f"{x['lang']} {100*x['diff']:+.1f} [{100*x['lo']:+.1f},{100*x['hi']:+.1f}]" for x in g))

# over-refusal (human-corrected, canonical): en from base (B4.3), it/de/hi from F1 (p4f_overref_corrected)
OR = {"base": [2.7, 5.5, 2.7, 0.0], "F1 (all langs)": [4.5, 2.3, 5.4, 9.6], "lang-conditional (en off)": [2.7, 2.3, 5.4, 9.6],
      "lang-cond, hi 0.3 (P4.9)": [2.7, 2.3, 5.4, HI030_OVERREF]}
print("\nover-refusal, human-corrected (%):")
print(pd.DataFrame(OR, index=LANGS).T.to_string())

bb = {c: json.load(open(hf_hub_download(HF_ARTEFACT_REPO, f"phase4/belebele/belebele_{c}_v2.json", repo_type="dataset")))["langs"]
      for c in ["base", "steer-ren-s24-a020"]}
acc = lambda c, l: round(100 * sum(bb[c][l]["correct"]) / len(bb[c][l]["correct"]), 1)
print("\nP4.9 hi at 0.3 vs F1 hi at 0.2, paired:", {k: round(100 * v, 1) if isinstance(v, float) else v for k, v in
      paired_diff(pd.concat([f1[f1.lang == "hi"], hi3[hi3.lang == "hi"].assign(lang="hi+a")], ignore_index=True), "hi+a", ref="hi").items()})
print("\nBelebele accuracy (%):  [hi at 0.3 not evaluated]")
print(pd.DataFrame({"base": [acc("base", l) for l in LANGS], "F1 (all langs)": [acc("steer-ren-s24-a020", l) for l in LANGS],
                    "lang-conditional (en off)": [acc("base", "en")] + [acc("steer-ren-s24-a020", l) for l in LANGS[1:]]},
                   index=LANGS).T.to_string())
