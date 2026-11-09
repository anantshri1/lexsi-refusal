"""Phase 3 5f (POST-HOC sensitivity) — en vs L trigger gap at several fixed thresholds on the slot-24 score."""
import sys; sys.path.insert(0, "src")
import numpy as np
from geometry import crossfit
from killtest import pair_arrays, K, B, SEED

meta, S, NRM, TAU, *_ = crossfit()
pe, eh, _, _ = pair_arrays(meta, S, TAU, "en")
for L in ["it", "de", "hi"]:
    po, oh, _, _ = pair_arrays(meta, S, TAU, L)
    sh = np.intersect1d(pe, po)
    a, b = eh[np.searchsorted(pe, sh), K], oh[np.searchsorted(po, sh), K]
    J = np.random.default_rng(SEED).integers(0, len(sh), (B, len(sh)))
    for t in [9, 12, 15, 18]:
        d = (a > t).astype(float) - (b > t)
        boot = d[J].mean(1)
        print(f"{L} t={t:2d} | trig en {np.mean(a > t):.2f} vs {L} {np.mean(b > t):.2f} | diff {d.mean():+.2f} "
              f"[{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}] (n={len(sh)})")