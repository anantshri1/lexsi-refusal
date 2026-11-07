"""Phase 2 B4.3 — blind human check of judge j1 on the rows that decide the kill-test (en/it/de/hi, pair_keep, original).
Strata by j1 label (hidden from the labeller): every H_core row j1 called failure; a sample j1 called safe;
benign rows j1 called over-refusal; benign rows j1 called compliance. Drawn ONCE (seeded); never re-drawn.
Label under JUDGE_RUBRIC.md Part A v2 (the clarified definitions)."""
import pandas as pd
import judge as J
from generate import load_bench
from paths import GENERATIONS, JUDGEMENTS, COMPREHENSION

SEED, LANGS = 0, ["en", "it", "de", "hi"]
N_CORE_SAFE, N_BEN_OR, N_BEN_OK = 8, 5, 3                       # per language; ALL core failures are taken
GEN = GENERATIONS / "gen_tinyaya_base_v1.jsonl"
SHEET = JUDGEMENTS / "audit_j1check_sheet.csv"
KEY = JUDGEMENTS / "audit_j1check_key.csv"
assert not SHEET.exists(), "already drawn — never re-draw (it would leak)"

b = load_bench()
keep = pd.read_csv(COMPREHENSION / "comprehension_v1.csv")[["uid", "pair_keep"]]
j1 = J.load_judgements(J.judge_path(GEN, 512, "qwen72b-j1"))[["label"]].rename(columns={"label": "j1"})
gen = pd.read_json(GEN, lines=True)[["uid", "response", "hit_eos"]]
seen = set(pd.read_csv(JUDGEMENTS / "audit_v1_key.csv").uid) | set(pd.read_json(GENERATIONS / "gen_tinyaya_base_v0.jsonl", lines=True).uid)

d = b.merge(keep, on="uid").merge(j1, left_on="uid", right_index=True).merge(gen, on="uid")
d = d[d.pair_keep & (d.variant == "original") & d.lang.isin(LANGS) & d.j1.notna() & ~d.uid.isin(seen)]
en = b[b.lang == "en"].set_index(["pair_id", "polarity", "variant"]).text
d["prompt_en"] = [en[k] for k in zip(d.pair_id, d.polarity, d.variant)]

fail = d.j1.isin(["COMPLIANCE", "PARTIAL"])
core, ben = (d.polarity == "harmful") & (d.tier == "H_core"), d.polarity == "benign"
ovr = d.j1.isin(["REFUSAL", "PUSHBACK"])
parts = []
for lang, g in d.groupby("lang"):
    m = g.index
    take = lambda mask, n: g[mask[m]].sample(min(n, mask[m].sum()), random_state=SEED) if n else g[mask[m]]
    parts += [take(core & fail, 0).assign(stratum="core_j1fail"),          # n=0 -> take all
              take(core & ~fail, N_CORE_SAFE).assign(stratum="core_j1safe"),
              take(ben & ovr, N_BEN_OR).assign(stratum="ben_j1overref"),
              take(ben & d.j1.eq("COMPLIANCE"), N_BEN_OK).assign(stratum="ben_j1ok")]
a = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
a.insert(0, "audit_id", range(1, len(a) + 1))
a["truncated"] = ~a.hit_eos
for c in ["response_en", "label", "note"]:
    a[c] = ""
a[["audit_id", "uid", "lang", "prompt_en", "text", "response", "truncated", "response_en", "label", "note"]] \
    .rename(columns={"text": "prompt"}).to_csv(SHEET, index=False)
a[["audit_id", "uid", "lang", "stratum", "tier", "polarity", "j1"]].to_csv(KEY, index=False)
print(len(a), "rows ->", SHEET.name)
print(a.groupby(["lang", "stratum"]).size().unstack(fill_value=0))
