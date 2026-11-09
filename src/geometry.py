"""Phase 3 — refusal-direction geometry on cached activations (Mac CPU). Rule: PREDICTIONS.md P3.2-rule."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from paths import ACTIVATIONS, COMPREHENSION, DIRECTIONS
from generate import load_bench

LANGS = ["en", "it", "de", "hi", "ml", "hi-Latn"]
K_FOLD, SEED = 5, 0
POS = {"last": 0, "user": 1}                        # axis 1 of the shards (PATHS.md)
BENCH = load_bench().set_index("uid")
KEEP = pd.read_csv(COMPREHENSION / "comprehension_v1.csv").set_index("uid").pair_keep


def load_lang(lang, var="orig", pos="last"):
    """meta (row order == tensor order) + float32 activations [n, 37, 2048]."""
    d = torch.load(ACTIVATIONS / f"act_tinyaya_base_v1_{lang}_{var}.pt", weights_only=False)
    meta = BENCH.loc[d["uids"], ["pair_id", "polarity", "tier", "lang"]].reset_index()
    meta["pair_keep"] = KEEP.reindex(meta.uid).values
    assert meta.pair_keep.notna().all(), "uid missing from comprehension_v1"
    return meta, d["acts"][:, POS[pos]].float()


def fold_map(pair_ids):
    """pair_id -> fold (seed 0). ONE map shared by all languages, so a pair is held out everywhere at once."""
    p = np.array(sorted(set(pair_ids)))
    perm = np.random.default_rng(SEED).permutation(len(p))
    return {p[i]: j % K_FOLD for j, i in enumerate(perm)}


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def crossfit(pos="last", tier="H_core"):
    """5 English directions (each excluding one fold of pairs) + tau per fold; every row scored by the direction
    whose training set excluded its pair. Returns long meta, raw scores S [N,37], norms NRM [N,37], TAU [N,37]."""
    em, ex = load_lang("en", pos=pos)
    fmap = fold_map(em.pair_id)
    train = (em.tier == tier) & em.pair_keep
    dirs, taus = [], []
    for f in range(K_FOLD):
        m = train & (em.pair_id.map(fmap) != f)
        xh, xb = ex[(m & (em.polarity == "harmful")).values], ex[(m & (em.polarity == "benign")).values]
        r = unit(xh.mean(0) - xb.mean(0))                                   # [37, 2048]
        dirs.append(r)
        taus.append(((xh * r).sum(-1).mean(0) + (xb * r).sum(-1).mean(0)) / 2)   # [37] midpoint, TRAIN rows
    D, T = torch.stack(dirs), torch.stack(taus)                             # [5,37,2048], [5,37]

    metas, S, NRM, TAU = [], [], [], []
    for lang in LANGS:
        meta, x = (em, ex) if lang == "en" else load_lang(lang, pos=pos)
        fo = meta.pair_id.map(fmap).values
        S.append((x * D[fo]).sum(-1).numpy()); NRM.append(x.norm(dim=-1).numpy()); TAU.append(T[fo].numpy())
        metas.append(meta.assign(fold=fo))
    full = unit(ex[(train & (em.polarity == "harmful")).values].mean(0) - ex[(train & (em.polarity == "benign")).values].mean(0))
    return pd.concat(metas, ignore_index=True), np.concatenate(S), np.concatenate(NRM), np.concatenate(TAU), D, T, full, fmap


def dprime(a, b):
    """(mu_a - mu_b) / pooled sd, per layer; a, b: [n, 37]."""
    return (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)


if __name__ == "__main__":
    meta, S, NRM, TAU, D, T, full, fmap = crossfit()
    DIRECTIONS.mkdir(parents=True, exist_ok=True)
    torch.save(dict(full=full, folds=D, tau_folds=T, fold_map=fmap, tier="H_core", pos="last", train="en pair_keep"),
               DIRECTIONS / "dir_tinyaya_refusal_last_en_v1.pt")

    # sanity 1: at the last position the INPUT token is always <|START_RESPONSE|> -> embeddings (slot 0) identical
    _, ex = load_lang("en")
    print("slot-0 spread across rows (want 0):", ex[:, 0].std(0).max().item())
    # sanity 2: held-out fold directions should agree with each other (estimator stability)
    c = torch.stack([(D[i] * D[j]).sum(-1) for i in range(5) for j in range(i + 1, 5)])   # [10, 37]
    print("fold-direction cosine, min over pairs, slots 6/12/18/24/30/36:", c.min(0).values[[6, 12, 18, 24, 30, 36]].numpy().round(3))
    # layer choice ON ENGLISH ONLY (P3.2-rule)
    en = (meta.lang == "en") & (meta.tier == "H_core") & meta.pair_keep
    dp = dprime(S[(en & (meta.polarity == "harmful")).values], S[(en & (meta.polarity == "benign")).values])
    k = int(np.nanargmax(dp[1:]) + 1)
    print("en held-out d' by slot:", np.round(dp, 2))
    print("k* =", k, "| en d'(k*) =", round(float(dp[k]), 2))