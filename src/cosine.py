"""Phase 3 step 4 — cross-language cosine of diff-of-means directions: REFUSAL (H_core - benign twin) vs neutral CONTROL
(food - travel) at MATCHED n (64 pairs) + random-sign null + English split-half reliability. Zhang 2026; PREDICTIONS P3.3."""
import sys; sys.path.insert(0, "src")
import hashlib
import numpy as np, pandas as pd, torch
from geometry import load_lang, unit
from paths import ACTIVATIONS, DATA_FROZEN, FIGURES

CTRL_SHA = "441cd2d412ac1d1a88977c398cf2f3a4b4c89e7a8f91451fbafe6f510b83ba15"
B, SEED, SLOTS = 1000, 0, [12, 18, 24, 30, 36]
DESCR = ("ml", "hi-Latn")                                            # no pair_keep requirement on these sides
COMPARE = [("en", L) for L in ["it", "de", "hi", "ml", "hi-Latn"]] + [("hi", "hi-Latn")]   # last = script question

p = DATA_FROZEN / "control_neutral_v1.jsonl"
assert hashlib.sha256(p.read_bytes()).hexdigest() == CTRL_SHA, "wrong control file"
CTRL = pd.read_json(p, lines=True).set_index("uid")


def pair_index(meta, col, a, b):
    """pair_id -> row position for the two sides of a contrast."""
    s = lambda v: pd.Series(meta.index[meta[col] == v], index=meta.pair_id[meta[col] == v].values)
    return s(a), s(b)


REF, CT, KEEP = {}, {}, {}                                          # store: lang -> (x all slots, x at SLOTS, side A idx, side B idx)
for L in ["en", "it", "de", "hi", "ml", "hi-Latn"]:
    m, x = load_lang(L)                                              # benchmark, pos <|START_RESPONSE|>
    core = m[m.tier == "H_core"]
    REF[L] = (x, x[:, SLOTS].contiguous(), *pair_index(core, "polarity", "harmful", "benign"))
    KEEP[L] = set(core[core.pair_keep].pair_id)
    d = torch.load(ACTIVATIONS / f"act_tinyaya_base_v1_ctrl_{L}_orig.pt", weights_only=False)
    cm = CTRL.loc[d["uids"], ["pair_id", "side"]].reset_index()
    xc = d["acts"][:, 0].float()
    CT[L] = (xc, xc[:, SLOTS].contiguous(), *pair_index(cm, "side", "food", "travel"))


def direction(store, L, pairs, signs=None, sl=False):
    """unit( mean over pairs of (side A - side B) ); signs = random +-1 per pair (null); sl = only the 5 SLOTS (fast)."""
    x, x5, ia, ib = store[L]
    x = x5 if sl else x
    d = x[ia[pairs].values] - x[ib[pairs].values]
    if signs is not None:
        d = d * torch.as_tensor(signs, dtype=d.dtype)[:, None, None]
    return unit(d.mean(0))


def cos(a, b):
    return (a * b).sum(-1).numpy()


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    PC = np.array(sorted(CT["en"][2].index)); n = len(PC)            # 64 control pairs
    signs = rng.choice([-1.0, 1.0], n)                               # random-sign null, SAME signs in every language
    rows = []
    for L1, L2 in COMPARE:
        need = [l for l in (L1, L2) if l not in DESCR]
        PR = np.array(sorted(set.intersection(*[KEEP[l] for l in need])))      # refusal pool (pair_keep), subsampled to n
        pt_ref = np.mean([cos(direction(REF, L1, s), direction(REF, L2, s))
                          for s in (rng.choice(PR, n, replace=False) for _ in range(50))], 0)
        pt_ct = cos(direction(CT, L1, PC), direction(CT, L2, PC))
        pt_rn = cos(direction(CT, L1, PC, signs), direction(CT, L2, PC, signs))
        diffs = []
        for _ in range(B):                                           # both sides resampled with replacement, same n
            sub = rng.choice(PR, n, replace=False)                        # 64 unique refusal pairs, like control...
            jr, jc = sub[rng.integers(0, n, n)], rng.integers(0, n, n)    # ...then the SAME resampling on both
            diffs.append(cos(direction(REF, L1, jr, sl=True), direction(REF, L2, jr, sl=True))
                         - cos(direction(CT, L1, PC[jc], sl=True), direction(CT, L2, PC[jc], sl=True)))
        lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
        ci = {s: (lo[i], hi[i]) for i, s in enumerate(SLOTS)}
        for k in range(37):
            rows.append(dict(L1=L1, L2=L2, slot=k, n_ref_pool=len(PR), n=n, refusal=pt_ref[k], control=pt_ct[k],
                             random=pt_rn[k], diff_lo=ci.get(k, (np.nan,))[0], diff_hi=ci.get(k, (np.nan, np.nan))[1]))
    R = pd.DataFrame(rows)

    rel = {}                                                         # English split-half reliability, 32 vs 32 pairs, both contrasts
    for name, store, P in [("refusal", REF, np.array(sorted(KEEP["en"]))), ("control", CT, PC)]:
        cs = []
        for _ in range(50):
            q = rng.permutation(P)[:n]
            cs.append(cos(direction(store, "en", q[: n // 2]), direction(store, "en", q[n // 2:])))
        rel[name] = np.mean(cs, 0)
    print("en split-half reliability (32 vs 32) at slots", SLOTS, "| refusal", rel["refusal"][SLOTS].round(3),
          "| control", rel["control"][SLOTS].round(3))
    for (L1, L2), g in R.groupby(["L1", "L2"], sort=False):
        print(f"\n{L1} vs {L2}  (refusal pool {g.n_ref_pool.iloc[0]} pairs -> matched n={n})")
        for r in g[g.slot.isin(SLOTS)].itertuples():
            print(f"  slot {r.slot:2d} | refusal {r.refusal:.3f} | control {r.control:.3f} | random {r.random:+.3f} "
                  f"| refusal-control {r.refusal - r.control:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]")
    R.to_csv(FIGURES / "p3_cosine_v1.csv", index=False)

    print("\nDisattenuated transfer: cross-language cosine on DISJOINT halves / sqrt(within-language reliabilities)")
    print("(1.0 = same direction up to sampling noise; different prompts on each side -> no shared-content cosine)")
    TR=[]
    for L1, L2 in COMPARE:
        need = [l for l in (L1, L2) if l not in DESCR]
        PR = np.array(sorted(set.intersection(*[KEEP[l] for l in need])))
        out = {}
        for name, store, P in [("refusal", REF, PR), ("control", CT, PC)]:
            w1, w2, cr = [], [], []
            for _ in range(100):
                q = rng.permutation(P)[:n]; A, Bh = q[: n // 2], q[n // 2:]
                a1, b1 = direction(store, L1, A, sl=True), direction(store, L1, Bh, sl=True)
                a2, b2 = direction(store, L2, A, sl=True), direction(store, L2, Bh, sl=True)
                w1.append(cos(a1, b1)); w2.append(cos(a2, b2)); cr.append((cos(a1, b2) + cos(a2, b1)) / 2)
            out[name] = np.mean(cr, 0) / np.sqrt(np.clip(np.mean(w1, 0) * np.mean(w2, 0), 1e-6, None))
        print(f"{L1} vs {L2} | slots {SLOTS} | refusal {out['refusal'].round(2)} | control {out['control'].round(2)}")
        TR += [dict(L1=L1, L2=L2, slot=s, refusal=out["refusal"][i], control=out["control"][i]) for i, s in enumerate(SLOTS)]
    pd.DataFrame(TR).to_csv(FIGURES / "p3_transfer_v1.csv", index=False)

    def transfer(store, L1, L2, P, rs):
        """one draw: disjoint halves A/B, resampled WITHIN each half; cross-language cosine / sqrt(reliabilities)."""
        q = rs.permutation(P)[:n]; A, Bh = q[: n // 2], q[n // 2:]
        A, Bh = A[rs.integers(0, len(A), len(A))], Bh[rs.integers(0, len(Bh), len(Bh))]
        a1, b1 = direction(store, L1, A, sl=True), direction(store, L1, Bh, sl=True)
        a2, b2 = direction(store, L2, A, sl=True), direction(store, L2, Bh, sl=True)
        return ((cos(a1, b2) + cos(a2, b1)) / 2) / np.sqrt(np.clip(cos(a1, b1) * cos(a2, b2), 1e-6, None))

    print("\nTransfer(refusal) - transfer(control): median [95% CI], B=500 (splits AND pairs resampled, halves disjoint)")
    T = []
    for L1, L2 in COMPARE:
        need = [l for l in (L1, L2) if l not in DESCR]
        PR = np.array(sorted(set.intersection(*[KEEP[l] for l in need])))
        dd = np.array([transfer(REF, L1, L2, PR, rng) - transfer(CT, L1, L2, PC, rng) for _ in range(500)])
        lo, md, hi = np.percentile(dd, [2.5, 50, 97.5], axis=0)
        print(f"{L1:2s} vs {L2:7s} | " + " | ".join(f"s{s} {m:+.2f} [{l:+.2f},{h:+.2f}]" for s, l, m, h in zip(SLOTS, lo, md, hi)))
        T += [dict(L1=L1, L2=L2, slot=s, diff_median=m, lo=l, hi=h) for s, l, m, h in zip(SLOTS, lo, md, hi)]
    pd.DataFrame(T).to_csv(FIGURES / "p3_transfer_diff_v1.csv", index=False)