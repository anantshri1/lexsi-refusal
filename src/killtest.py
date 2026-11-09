"""Phase 3 5c — KILL-TEST (PREDICTIONS.md P3.2-rule): separation (AUROC, d') and calibration (trigger rate) of H_core vs
benign twins along the cross-fitted ENGLISH direction at k* (chosen on en, 5b) + all-layer curves. raw = headline, cos = sensitivity."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from geometry import crossfit, load_lang, LANGS, K_FOLD
from paths import FIGURES

B, SEED, K = 2000, 0, 24          # K = k* from 5b (argmax en held-out d' = 4.65)


def pair_arrays(meta, S, TAU, lang, keep=True):
    """H_core only, aligned by pair: sorted pair ids, harmful scores, benign scores, tau (rows [n_pairs, 37])."""
    m = (meta.lang == lang) & (meta.tier == "H_core")
    if keep:
        m &= meta.pair_keep
    d = meta[m]
    ih = pd.Series(d.index[d.polarity == "harmful"], index=d.pair_id[d.polarity == "harmful"].values)
    ib = pd.Series(d.index[d.polarity == "benign"], index=d.pair_id[d.polarity == "benign"].values)
    p = np.array(sorted(set(ih.index) & set(ib.index)))
    return p, S[ih[p].values], S[ib[p].values], TAU[ih[p].values]


def metrics(sh, sb, th):
    """Works for [n] (one layer) or [n, L] (curves). AUROC = P(harmful > benign), ties 1/2."""
    gt = (sh[:, None] > sb[None, :]).mean((0, 1)); eq = (sh[:, None] == sb[None, :]).mean((0, 1))
    dp = (sh.mean(0) - sb.mean(0)) / np.sqrt((sh.var(0, ddof=1) + sb.var(0, ddof=1)) / 2)
    return dict(auroc=gt + 0.5 * eq, dprime=dp, trig_h=(sh > th).mean(0), trig_b=(sb > th).mean(0))


def paired_boot(e, o):
    """en vs other at slot K on SHARED pairs: point estimates + 95% CI of (en - other). Resampling unit = pair."""
    shared = np.intersect1d(e[0], o[0])
    ie, io = np.searchsorted(e[0], shared), np.searchsorted(o[0], shared)
    E, O = [a[ie, K] for a in e[1:]], [a[io, K] for a in o[1:]]
    rng, draws = np.random.default_rng(SEED), []
    for _ in range(B):
        j = rng.integers(0, len(shared), len(shared))
        me, mo = metrics(*[a[j] for a in E]), metrics(*[a[j] for a in O])
        draws.append({k: me[k] - mo[k] for k in me})
    pe, po = metrics(*E), metrics(*O)
    return len(shared), pe, po, {k: np.nanpercentile([d[k] for d in draws], [2.5, 97.5]) for k in pe}


def cos_tau(meta, D, fmap):
    """tau for COSINE scores, recomputed the registered way: per fold, midpoint of en TRAIN-row means."""
    em, ex = load_lang("en")
    tr, fo, nrm = ((em.tier == "H_core") & em.pair_keep).values, em.pair_id.map(fmap).values, ex.norm(dim=-1)
    T = []
    for f in range(K_FOLD):
        c, m = ((ex * D[f]).sum(-1) / nrm).numpy(), tr & (fo != f)
        T.append((c[m & (em.polarity == "harmful").values].mean(0) + c[m & (em.polarity == "benign").values].mean(0)) / 2)
    return np.stack(T)[meta.fold.values]


if __name__ == "__main__":
    meta, S, NRM, TAU, D, T, full, fmap = crossfit()
    runs = {"raw": (S, TAU), "cos": (S / NRM, cos_tau(meta, D, fmap))}
    curves, lines = [], []
    for name, (SS, TT) in runs.items():
        en = pair_arrays(meta, SS, TT, "en")
        for lang in LANGS:
            for keep in ([True, False] if lang in ("ml", "hi-Latn") else [True]):   # descriptive langs: also all pairs
                o = pair_arrays(meta, SS, TT, lang, keep)
                curves.append(pd.DataFrame(metrics(*o[1:])).assign(slot=range(37), lang=lang, keep=keep, score=name))
                if lang == "en":
                    continue
                n, pe, po, ci = paired_boot(en, o)
                lines.append(dict(score=name, lang=lang, keep=keep, n=n, **{f"{k}_en": pe[k] for k in pe},
                                  **{f"{k}_L": po[k] for k in po}, **{f"{k}_lo": ci[k][0] for k in ci},
                                  **{f"{k}_hi": ci[k][1] for k in ci}))
    R = pd.DataFrame(lines)
    for name in runs:
        print(f"\n=== {name} | slot {K} | en vs L on shared pairs | [95% CI of en - L] ===")
        for x in R[R.score == name].itertuples():
            print(f"{x.lang:8s} keep={x.keep!s:5s} n={x.n:3d} | AUROC {x.auroc_en:.3f} vs {x.auroc_L:.3f} "
                  f"[{x.auroc_lo:+.3f},{x.auroc_hi:+.3f}] | d' {x.dprime_en:.2f} vs {x.dprime_L:.2f} "
                  f"| trig_h {x.trig_h_en:.2f} vs {x.trig_h_L:.2f} [{x.trig_h_lo:+.2f},{x.trig_h_hi:+.2f}] "
                  f"| trig_b {x.trig_b_en:.2f} vs {x.trig_b_L:.2f}")
    h = R[(R.score == "raw") & (R.lang == "hi")].iloc[0]
    A, C = h.auroc_lo > 0, h.trig_h_lo > 0
    print(f"\nKILL-TEST (hi, raw, slot {K}):", "GO" if (A or C) else "KILL", "| Wang/A branch (AUROC):", A, "| Aziz/C branch (trigger):", C)
    pd.concat(curves).to_csv(FIGURES / "p3_killtest_curves_v1.csv", index=False)
    R.to_csv(FIGURES / "p3_killtest_v1.csv", index=False)