"""Phase 3 5e (EXPLORATORY, post-hoc) — paired twins: for pairs where en REFUSED but L COMPLIED, does the L prompt
score far below its own English twin (representation, ii) or about the same (decision threshold, i)?"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from geometry import crossfit
from killtest import K
from rates import load

meta, S, NRM, TAU, *_ = crossfit()
lab = load().set_index("uid").label
d = meta.assign(s=S[:, K], label=lab.reindex(meta.uid).values)
d = d[(d.tier == "H_core") & (d.polarity == "harmful") & d.label.notna() & (d.label != "NON_ANSWER")]
d["fail"] = d.label.isin(["COMPLIANCE", "PARTIAL"])

for L in ["en", "it", "de", "hi"]:            # where does each language put its failures vs its refusals?
    x = d[d.lang == L]
    print(f"{L}: complied scores {np.sort(x.s[x.fail].values).round(1)} | refused 5th/10th pct {np.percentile(x.s[~x.fail], [5, 10]).round(1)}")

w = d.pivot(index="pair_id", columns="lang", values=["s", "fail"])
for L in ["it", "de", "hi"]:
    x = w[[("s", "en"), ("s", L), ("fail", "en"), ("fail", L)]].dropna()
    x.columns = ["s_en", "s_L", "f_en", "f_L"]
    x = x.astype({"s_en": float, "s_L": float, "f_en": bool, "f_L": bool})
    for name, m in [("GAP: en refused, L complied", ~x.f_en & x.f_L), ("both refused", ~x.f_en & ~x.f_L)]:
        y, dl = x[m], x[m].s_L - x[m].s_en
        print(f"{L} | {name:28s} n={len(y):3d} | mean s_en {y.s_en.mean():5.1f} s_{L} {y.s_L.mean():5.1f} "
              f"| delta median {dl.median():+5.1f} [IQR {dl.quantile(.25):+.1f}, {dl.quantile(.75):+.1f}]")