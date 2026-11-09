"""Phase 3 step 5 (EXPLORATORY) — paired-twin divergence by depth: at which layer do GAP prompts (en refused, L complied)
lose harm signal vs their English twin, beyond pairs both refused? Units: fraction of the English harmful-benign separation."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from geometry import crossfit
from rates import load
from paths import FIGURES

meta, S, *_ = crossfit()
en = (meta.lang == "en") & (meta.tier == "H_core") & meta.pair_keep
sep = S[(en & (meta.polarity == "harmful")).values].mean(0) - S[(en & (meta.polarity == "benign")).values].mean(0)   # [37]

lab = load().set_index("uid").label
m = meta.assign(label=lab.reindex(meta.uid).values)
keep = ((m.tier == "H_core") & (m.polarity == "harmful") & m.label.notna() & (m.label != "NON_ANSWER")).values
m, S = m[keep].reset_index(drop=True), S[keep]
fail = m.label.isin(["COMPLIANCE", "PARTIAL"]).values
pos = {(p, l): i for i, (p, l) in enumerate(zip(m.pair_id, m.lang))}

rows = []
for L in ["it", "de", "hi"]:
    pr = [p for p in m.pair_id[m.lang == L] if (p, "en") in pos]
    ie, il = np.array([pos[(p, "en")] for p in pr]), np.array([pos[(p, L)] for p in pr])
    d = (S[il] - S[ie]) / np.where(np.abs(sep) > 1e-6, sep, np.nan)        # [pairs, 37] in units of English separation
    gap, both = ~fail[ie] & fail[il], ~fail[ie] & ~fail[il]
    for k in range(1, 37):
        rows.append(dict(lang=L, slot=k, gap=np.median(d[gap, k]), both=np.median(d[both, k]), n_gap=gap.sum(), n_both=both.sum()))
R = pd.DataFrame(rows); R["extra"] = R.gap - R.both
print("n GAP / both-refused:", R.groupby("lang")[["n_gap", "n_both"]].first().to_dict("index"))
print("EXTRA loss of GAP pairs beyond both-refused (fraction of English separation; negative = lost):")
print(R[R.slot.isin([3, 6, 9, 12, 15, 18, 21, 24, 30, 36])].pivot(index="slot", columns="lang", values="extra").round(2))
R.to_csv(FIGURES / "p3_twin_depth_v1.csv", index=False)