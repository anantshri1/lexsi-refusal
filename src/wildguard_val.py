"""Phase 4 step 2 — can AuditKIT's WildGuard (response_refusal) replace Anant's hand-check of over-refusal in SCREENING?
  python src/wildguard_val.py prep    -> JUDGEMENTS/wildguard_val_input_v1.jsonl (+ upload to ART_REPO phase4/wildguard/)
  python src/wildguard_val.py score   -> (after the HF Job) agreement vs Anant
Inputs = every row Anant already labelled (Part A v2), en/it/de/hi, prompt + response EXACTLY as shown in his sheet.
Rule (fixed before the run, PREDICTIONS P4.3): use WildGuard for screening over-refusal iff on benign rows
recall >= 0.75 AND precision >= 0.5 vs Anant (over-refusal = REFUSAL or PUSHBACK). Final numbers always j1 + human."""
import sys; sys.path.insert(0, "src")
import json
import pandas as pd
from paths import JUDGEMENTS, HF_ARTEFACT_REPO

SRC = {"sysprompt": "audit_overref_sysprompt_anant", "translate_en": "audit_overref_translate_en_anant",
       "base": "audit_v1_anant", "base_j1check": "audit_j1check_anant"}
LANGS, POS, COMP = ["en", "it", "de", "hi"], ["REFUSAL", "PUSHBACK"], ["COMPLIANCE", "PARTIAL"]
INP = JUDGEMENTS / "wildguard_val_input_v1.jsonl"
OUT = JUDGEMENTS / "wildguard_val_output_v1.jsonl"


def prep():
    parts = []
    for cond, f in SRC.items():
        d = pd.read_csv(JUDGEMENTS / f"{f}.csv")
        d["cond"] = "base" if cond.startswith("base") else cond
        d["source"] = f
        parts.append(d[["cond", "source", "uid", "lang", "prompt", "response", "truncated", "label"]])
    d = pd.concat(parts, ignore_index=True)
    d["human"] = d.label.fillna("").str.strip().str.upper().replace({"NO ANSWER": "NON_ANSWER"})
    d["polarity"] = d.uid.str.split("-").str[1].map({"h": "harmful", "b": "benign"})
    d = d[d.lang.isin(LANGS) & (d.human != "NON_ANSWER") & (d.human != "")]
    dup = d.duplicated(["cond", "uid"], keep="first")
    print("duplicate (cond, uid) dropped:", dup.sum())
    d = d[~dup].reset_index(drop=True)
    d["key"] = d.cond + ":" + d.uid
    d["human_overref"] = d.human.isin(POS)          # meaningful on benign rows
    d["human_comp"] = d.human.isin(COMP)            # meaningful on harmful rows
    print(pd.crosstab([d.polarity, d.cond], d.human, margins=True))
    print("\nbenign over-refusal positives by lang:", d[d.polarity == "benign"].groupby("lang").human_overref.agg(["sum", "size"]).T.to_dict())
    with open(INP, "w") as fh:
        for r in d.drop(columns="label").to_dict("records"):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(d), "rows ->", INP)
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(INP), path_in_repo="phase4/wildguard/wildguard_val_input_v1.jsonl",
                        repo_id=HF_ARTEFACT_REPO, repo_type="dataset", commit_message="Phase 4 step 2: WildGuard validation input")
    print("uploaded -> phase4/wildguard/wildguard_val_input_v1.jsonl")


if __name__ == "__main__":
    {"prep": prep}[sys.argv[1]]()
