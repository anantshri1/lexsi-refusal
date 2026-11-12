"""Phase 4 block 4.1a — rows of the FULL runs (D2a): en/it/de/hi, original variant, pair_keep, ALL tiers (H_core + H_bait),
harmful + benign = the same row set as the Phase 2 sysprompt run (1032), so steered and baseline numbers are comparable.
Writes PHASE4/full/full_uids_v1.json (+ upload to ART_REPO phase4/full/)."""
import sys; sys.path.insert(0, "src")
import json
import pandas as pd
from generate import load_bench
from paths import COMPREHENSION, GENERATIONS, PHASE4, HF_ARTEFACT_REPO

b = load_bench().merge(pd.read_csv(COMPREHENSION / "comprehension_v1.csv")[["uid", "pair_keep"]], on="uid")
u = b[(b.variant == "original") & b.lang.isin(["en", "it", "de", "hi"]) & b.pair_keep].uid.tolist()
sp = set(pd.read_json(GENERATIONS / "gen_tinyaya_sysprompt_v1.jsonl", lines=True).uid)
print(f"full rows: {len(u)} | == sysprompt row set: {set(u) == sp} | by lang:",
      b[b.uid.isin(u)].groupby("lang").size().to_dict(), "| by tier:", b[b.uid.isin(u)].groupby("tier").size().to_dict())
out = PHASE4 / "full" / "full_uids_v1.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(uids=u, note="en/it/de/hi original pair_keep, all tiers; = Phase 2 sysprompt rows"), open(out, "w"), indent=1)
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=str(out), path_in_repo="phase4/full/full_uids_v1.json", repo_id=HF_ARTEFACT_REPO,
                    repo_type="dataset", commit_message="Phase 4 block 4.1a: full-run rows")
print("saved + uploaded:", out)
