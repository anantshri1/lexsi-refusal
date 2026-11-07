"""Phase 2 B2.3 — stratified, ENRICHED audit sample for the refusal judge. Drawn ONCE (seeded); never re-drawn."""
import pandas as pd
from paths import GENERATIONS, COMPREHENSION, JUDGEMENTS
from generate import load_bench

SEED, N_H_LONG, N_H_RAND, N_B = 0, 4, 4, 4           # per language
SHEET = JUDGEMENTS / "audit_v1_sheet.csv"            # what Anant labels (no stratum, tier, polarity, judge label)
KEY   = JUDGEMENTS / "audit_v1_key.csv"              # uid -> stratum, tier, polarity (for analysis only)
assert not SHEET.exists(), "audit already drawn — never re-draw (it would leak)"

bench = load_bench()
gen  = pd.read_json(GENERATIONS / "gen_tinyaya_base_v1.jsonl", lines=True)
v0   = set(pd.read_json(GENERATIONS / "gen_tinyaya_base_v0.jsonl", lines=True).uid)   # judge labels seen in tests
keep = pd.read_csv(COMPREHENSION / "comprehension_v1.csv")[["uid", "pair_keep"]]

d = gen[["uid", "response", "n_new_tokens", "hit_eos"]].merge(bench, on="uid").merge(keep, on="uid")
d = d[d.pair_keep & ~d.uid.isin(v0)]
en = bench[bench.lang == "en"].set_index(["pair_id", "polarity", "variant"]).text
d["prompt_en"] = [en[k] for k in zip(d.pair_id, d.polarity, d.variant)]   # original English, for understanding

parts = []
for lang, g in d.groupby("lang"):
    h, b = g[g.polarity == "harmful"], g[g.polarity == "benign"]
    long_ = h[h.n_new_tokens >= h.n_new_tokens.median()]          # enrichment: long harmful replies (likely hard cases)
    s_long = long_.sample(min(N_H_LONG, len(long_)), random_state=SEED)
    rest = h.drop(s_long.index)
    s_rand = rest.sample(min(N_H_RAND, len(rest)), random_state=SEED)
    s_b = b.sample(min(N_B, len(b)), random_state=SEED)
    parts += [s_long.assign(stratum="harmful_long"), s_rand.assign(stratum="harmful_rand"), s_b.assign(stratum="benign")]

a = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle: no language blocks
a.insert(0, "audit_id", range(1, len(a) + 1))
a["truncated"] = ~a.hit_eos
for c in ["response_en", "read_via", "label", "note"]:
    a[c] = ""
a[["audit_id", "uid", "lang", "prompt_en", "text", "response", "truncated", "response_en", "read_via", "label", "note"]] \
    .rename(columns={"text": "prompt"}).to_csv(SHEET, index=False)
a[["audit_id", "uid", "lang", "stratum", "tier", "polarity", "n_new_tokens"]].to_csv(KEY, index=False)
print(len(a), "rows ->", SHEET.name)
print(a.groupby(["lang", "stratum"]).size().unstack(fill_value=0))
