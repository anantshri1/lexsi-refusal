"""Phase 4 post-hoc (2026-11-13): census of hi H_bait benign rows j1 called NOT refusal, under F1 and F2 (symmetric),
because F1 hi over-refusal (12.8%) rested on ONE labelled row. Excludes rows already in the audit sheets.
  python src/audit_supp.py  -> JUDGEMENTS/audit_supp_hi_sheet.csv (blind, shuffled). Label it, then APPEND the labelled rows
  to the matching audit_overref_<cond>_anant.csv (key file says which cond) and rerun audit_overref.py correct / final_analysis.py."""
import sys; sys.path.insert(0, "src")
import pandas as pd
from rates import load
from paths import JUDGEMENTS, PHASE4

rows = []
for cond in ["steer-ren-s24-a020_full", "steer-rhi-s24-a020_full"]:
    gen = PHASE4 / "full" / f"gen_tinyaya_{cond}_v1.jsonl"
    d = load(gen=gen)
    d = d[(d.lang == "hi") & (d.polarity == "benign") & (d.tier == "H_bait") & ~d.label.isin(["REFUSAL", "PUSHBACK"])]
    done = set(pd.read_csv(JUDGEMENTS / f"audit_overref_{cond}_key.csv").uid)
    d = d[~d.uid.isin(done)].merge(pd.read_json(gen, lines=True)[["uid", "response", "hit_eos"]], on="uid")
    rows.append(d.assign(cond=cond))
a = pd.concat(rows).sample(frac=1, random_state=1).reset_index(drop=True)
a.insert(0, "audit_id", range(1, len(a) + 1))
a["truncated"] = ~a.hit_eos
for c in ["response_en", "label", "note"]:
    a[c] = ""
a[["audit_id", "uid", "lang", "text", "response", "truncated", "response_en", "label", "note"]] \
    .rename(columns={"text": "prompt"}).to_csv(JUDGEMENTS / "audit_supp_hi_sheet.csv", index=False)
a[["audit_id", "uid", "cond"]].to_csv(JUDGEMENTS / "audit_supp_hi_key.csv", index=False)
print(len(a), "rows:", a.cond.value_counts().to_dict())