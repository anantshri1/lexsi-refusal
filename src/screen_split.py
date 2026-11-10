"""Phase 4 block 3.1 — which rows screening uses (PHASE4_PLAN.md, decisions A2/B). Writes PHASE4/screen/split_v1.json and
uploads it to ART_REPO phase4/screen/split_v1.json for the screening job (which has no repo and no comprehension file).
 - fold_map: LOADED from the Phase 3 direction file (never recomputed) -> same folds as r_en / r_hi / dir_tinyaya_steer_v1.pt
 - train_pairs[L]: H_core pairs with pair_keep in L = the rows SafeTune averages for L's fold directions (L in en, hi)
 - dev_pairs: H_core, fold in {0,1,2}, pair_keep in en AND hi AND de;  test_pairs: same with fold in {3,4} (robustness only)
 - dev_uids: dev pairs x {harmful, benign} x {en, hi, de}, original variant = exactly the rows every screening condition generates."""
import sys; sys.path.insert(0, "src")
import json
import pandas as pd, torch
from paths import COMPREHENSION, DIRECTIONS, PHASE4, HF_ARTEFACT_REPO
from generate import load_bench
from rates import load

DEV_FOLDS, TEST_FOLDS, DEV_LANGS, TRAIN_LANGS = [0, 1, 2], [3, 4], ["en", "hi", "de"], ["en", "hi"]
OUT = PHASE4 / "screen" / "split_v1.json"

fmap = torch.load(DIRECTIONS / "dir_tinyaya_refusal_last_en_v1.pt", weights_only=False)["fold_map"]
fmap = {str(k): int(v) for k, v in fmap.items()}
b = load_bench()
b = b[b.variant == "original"].merge(pd.read_csv(COMPREHENSION / "comprehension_v1.csv")[["uid", "pair_keep"]], on="uid")
core = b[b.tier == "H_core"]
keep = {L: set(core[(core.lang == L) & core.pair_keep].pair_id) for L in ["en", "it", "de", "hi"]}
assert set(core.pair_id) <= set(fmap), "a H_core pair is missing from the fold map"

train = {L: sorted(keep[L]) for L in TRAIN_LANGS}
pick = lambda folds: sorted(p for p in set(core.pair_id) if fmap[p] in folds and all(p in keep[L] for L in DEV_LANGS))
dev, test = pick(DEV_FOLDS), pick(TEST_FOLDS)
uids = [f"{p}-{s}-{L}-orig" for p in dev for L in DEV_LANGS for s in ("h", "b")]
assert set(uids) <= set(b.uid), "constructed uid not in benchmark"
assert not set(dev) & set(test)

# base dev compliance from Phase 2 j1 labels at t256 (greedy: = the first 256 tokens of the t512 run; B8)
lab = load(cut=256).set_index("uid").label
h = pd.DataFrame({"uid": [u for u in uids if "-h-" in u]})
h["lang"], h["label"] = h.uid.str.split("-").str[2], lab.reindex(h.uid).values
h = h[h.label.notna() & (h.label != "NON_ANSWER")]
base = h.assign(c=h.label.isin(["COMPLIANCE", "PARTIAL"])).groupby("lang").c.agg(["sum", "size", "mean"]).round(3)

print("train pairs (SafeTune fold extraction):", {L: len(v) for L, v in train.items()})
print("train pairs per fold, en:", pd.Series([fmap[p] for p in train["en"]]).value_counts().sort_index().to_dict())
print(f"dev pairs: {len(dev)} (folds {DEV_FOLDS}) | test pairs: {len(test)} (folds {TEST_FOLDS}) | dev rows per condition: {len(uids)}")
print("dev pairs per fold:", pd.Series([fmap[p] for p in dev]).value_counts().sort_index().to_dict())
print("base dev H_core compliance, j1 @ t256 (sum / n / rate):\n", base.to_string())

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(fold_map=fmap, train_pairs=train, dev_pairs=dev, test_pairs=test, dev_uids=uids, dev_langs=DEV_LANGS,
               dev_folds=DEV_FOLDS, test_folds=TEST_FOLDS, note="H_core only; pair_keep per comprehension_v1; PHASE4_PLAN A2/B"),
          open(OUT, "w"), indent=1)
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=str(OUT), path_in_repo="phase4/screen/split_v1.json", repo_id=HF_ARTEFACT_REPO,
                    repo_type="dataset", commit_message="Phase 4 block 3.1: screening split")
print("saved + uploaded:", OUT)
