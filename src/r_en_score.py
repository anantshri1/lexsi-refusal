"""Phase 3 5d — row level: does the r_en score at k* predict what the model DID (j1 @ t512) within each language?
Pre-registered P3.2 bullet 3 (Claude: complied hi rows project lower, row AUROC ~0.7, 55%)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from geometry import crossfit
from killtest import K
from rates import load

meta, S, NRM, TAU, *_ = crossfit()
lab = load().set_index("uid").label                     # pair_keep, original, j1 @ t512, unparsed dropped
d = meta.assign(s=S[:, K], tau=TAU[:, K], label=lab.reindex(meta.uid).values)
d = d[(d.tier == "H_core") & (d.polarity == "harmful") & d.label.notna() & (d.label != "NON_ANSWER")]
d["fail"], d["above"] = d.label.isin(["COMPLIANCE", "PARTIAL"]), d.s > d.tau
for lang in ["en", "it", "de", "hi"]:
    x = d[d.lang == lang]
    r, c = x.s[~x.fail].values, x.s[x.fail].values
    auc = (r[:, None] > c[None]).mean() + 0.5 * (r[:, None] == c[None]).mean() if len(c) else np.nan
    print(f"{lang}: n={len(x)} fail={x.fail.sum()} | complied ABOVE tau {(x.fail & x.above).sum()} | complied below "
          f"{(x.fail & ~x.above).sum()} | refused below {(~x.fail & ~x.above).sum()} | row AUROC(refused>complied) {auc:.2f} "
          f"| mean s refused {r.mean():.1f} vs complied {c.mean():.1f} (tau ~{x.tau.mean():.1f})")