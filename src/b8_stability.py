"""Phase 2 B8 — does the j1 verdict depend on response length? Labels at t128 / t256 vs t512 (pair_keep, original)."""
import pandas as pd
from rates import load, metrics, LANGS

fail = lambda s: s.isin(["COMPLIANCE", "PARTIAL"])
ref = load(cut=512).set_index("uid")
rows = []
for cut in [128, 256]:
    x = load(cut=cut).set_index("uid")
    for l in LANGS:
        u = ref.index[(ref.lang == l) & ref.index.isin(x.index)]
        a, b = ref.loc[u], x.loc[u]
        h = a.polarity == "harmful"
        rows.append(dict(cut=cut, lang=l, n=len(u),
                         label_same=(a.label == b.label).mean(),               # 5-way label unchanged vs t512
                         harm_failflag_same=(fail(a.label[h]) == fail(b.label[h])).mean(),   # metric-level
                         core_comp_cut=metrics(b)["core_comp_cond"], core_comp_512=metrics(a)["core_comp_cond"]))
print(pd.DataFrame(rows).round(3).to_string(index=False))
