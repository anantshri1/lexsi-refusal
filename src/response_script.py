"""Judge-free check: which script is each response written in? Share of letters in Devanagari / Malayalam / Latin."""
import pandas as pd
from paths import GENERATIONS
from generate import load_bench

g = pd.read_json(GENERATIONS / "gen_tinyaya_base_v1.jsonl", lines=True).merge(
        load_bench()[["uid", "lang", "polarity"]], on="uid")

def share(t, lo, hi):
    letters = [c for c in t if c.isalpha()]
    return sum(lo <= ord(c) <= hi for c in letters) / max(1, len(letters))

for name, lo, hi in [("deva", 0x0900, 0x097F), ("mlym", 0x0D00, 0x0D7F), ("latn", 0x0041, 0x024F)]:
    g[name] = g.response.map(lambda t: share(t, lo, hi))
s = g[["deva", "mlym", "latn"]]
g["script"] = s.idxmax(axis=1).where(s.max(axis=1) >= 0.5, "mixed")      # majority script, else mixed
print(pd.crosstab([g.lang, g.polarity], g.script, normalize="index").round(2))
