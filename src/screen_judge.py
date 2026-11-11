"""Phase 4 block 3.3 — j1 (Qwen2.5-72B, Part A v1, same judge as every baseline) on the screening generations, HARMFUL dev rows
only (compliance; benign over-refusal is WildGuard's job in screening, D11). cut t256 (= what was generated).
  python src/screen_judge.py            -> all 14 conditions in CONDS order (kill-test first)
  python src/screen_judge.py K1 K2 K3   -> a subset by position-free id prefix, e.g. steer-ren-s24-a025
Downloads phase4/screen/gen_*_dev_v1.jsonl to PHASE4/screen (Mac mirror). Unparsed rows are retried (up to 2 extra passes,
MNAR rule); whatever stays unparsed is reported. Output: JUDGEMENTS/judge_qwen72b-j1_tinyaya_{cond}_dev_g1_t256.jsonl."""
import sys; sys.path.insert(0, "src")
import json
import pandas as pd
from huggingface_hub import hf_hub_download
import judge as J
from generate import load_bench
from paths import PHASE4, HF_ARTEFACT_REPO
from screen_job import CONDS, MAX_NEW

CUT, WORKERS = 256, 3
split = json.load(open(PHASE4 / "screen" / "split_v1.json"))
harm = [u for u in split["dev_uids"] if "-h-" in u]
want = sys.argv[1:] or [c[0] for c in CONDS]
bench = load_bench()

for cid in [c[0] for c in CONDS if c[0] in want]:
    name = f"gen_tinyaya_{cid}_dev_v1.jsonl"
    p = PHASE4 / "screen" / name
    if not p.exists():
        src = hf_hub_download(HF_ARTEFACT_REPO, f"phase4/screen/{name}", repo_type="dataset")   # HF cache path
        p.write_bytes(open(src, "rb").read())                                                     # Mac mirror (PATHS)
    g = pd.read_json(p, lines=True)
    assert len(g) == 330 and g.ids.map(len).max() <= MAX_NEW == CUT, "generation length > cut: truncation would need tok"
    for attempt in range(3):                                   # first pass + 2 retries of API errors / unparsed labels
        out = J.run_judge(p, bench, cut=CUT, tok=None, workers=WORKERS, uids=harm)
        j = J.load_judgements(out)
        miss = set(harm) - set(j[j.label.notna()].index)
        if not miss:
            break
        print(f"  {cid}: {len(miss)} harmful rows unparsed after pass {attempt + 1}")
    print(f"== {cid}: judged {len(set(harm) - miss)}/{len(harm)} harmful rows; unparsed left: {sorted(miss)}")
