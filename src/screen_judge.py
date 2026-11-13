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

FULL = "--full" in sys.argv                       # block 4.4: full runs F1/F2 (ALL 1032 rows, harmful AND benign, t512)
FULLHI = "--fullhi" in sys.argv                   # P4.9: hi rows only at alpha/Delta 0.3
args = [a for a in sys.argv[1:] if a not in ("--full", "--fullhi")]
if FULLHI:
    from screen_job import HI_CONDS, FULL_NEW
    CUT, WORKERS, SUB, TAG, NROWS = FULL_NEW, 3, "full", "fullhi", 248
    CONDS = HI_CONDS
    harm = [u for u in json.load(open(PHASE4 / "full" / "full_uids_v1.json"))["uids"] if "-hi-" in u]
    assert J.PROVIDER == "deepinfra", "full runs are judged on deepinfra (P4.6): export LEXSI_JUDGE_PROVIDER=deepinfra"
elif FULL:
    from screen_job import FULL_CONDS, FULL_NEW
    CUT, WORKERS, SUB, TAG, NROWS = FULL_NEW, 3, "full", "full", 1032
    CONDS = FULL_CONDS
    harm = json.load(open(PHASE4 / "full" / "full_uids_v1.json"))["uids"]      # j1 on benign too: over-refusal flags (D11)
    assert J.PROVIDER == "deepinfra", "full runs are judged on deepinfra (P4.6): export LEXSI_JUDGE_PROVIDER=deepinfra"
else:
    CUT, WORKERS, SUB, TAG, NROWS = 256, 3, "screen", "dev", 330
    split = json.load(open(PHASE4 / "screen" / "split_v1.json"))
    harm = [u for u in split["dev_uids"] if "-h-" in u]
want = args or [c[0] for c in CONDS]
bench = load_bench()

for cid in [c[0] for c in CONDS if c[0] in want]:
    name = f"gen_tinyaya_{cid}_{TAG}_v1.jsonl"
    p = PHASE4 / SUB / name
    if not p.exists():
        src = hf_hub_download(HF_ARTEFACT_REPO, f"phase4/{SUB}/{name}", repo_type="dataset")    # HF cache path
        p.write_bytes(open(src, "rb").read())                                                     # Mac mirror (PATHS)
    g = pd.read_json(p, lines=True)
    assert len(g) == NROWS and g.ids.map(len).max() <= CUT, "generation length > cut: truncation would need tok"
    for attempt in range(3):                                   # first pass + 2 retries of API errors / unparsed labels
        out = J.run_judge(p, bench, cut=CUT, tok=None, workers=WORKERS, uids=harm)
        j = J.load_judgements(out)
        miss = set(harm) - set(j[j.label.notna()].index)
        if not miss:
            break
        print(f"  {cid}: {len(miss)} harmful rows unparsed after pass {attempt + 1}")
    print(f"== {cid}: judged {len(set(harm) - miss)}/{len(harm)} harmful rows; unparsed left: {sorted(miss)}")
