"""Phase 4 step 0 — steering vectors at SLOTS 18/21/24 (Mac CPU). PREDICTIONS P4.0.
Naming rule: SLOT k = output of decoder layer k-1, pre final norm (PATHS.md). SafeTune RefusalDirectionModel(layers=[i])
hooks the output of decoder block i -> use st_idx(k) = k - 1 EVERYWHERE we call SafeTune. Files/notes always say 'slot'."""
import sys; sys.path.insert(0, "src")
import hashlib
import pandas as pd, torch
from geometry import load_lang, unit, K_FOLD
from paths import ACTIVATIONS, DATA_FROZEN, DIRECTIONS

SLOTS = [18, 21, 24]
CTRL_SHA = "441cd2d412ac1d1a88977c398cf2f3a4b4c89e7a8f91451fbafe6f510b83ba15"


def st_idx(slot):
    return slot - 1


R = torch.load(DIRECTIONS / "dir_tinyaya_refusal_last_en_v1.pt", weights_only=False)
FMAP = R["fold_map"]                       # pair_id -> fold, the SAME map Phase 3 used


def refusal_dirs(lang):
    """Cross-fitted H_core - benign directions for `lang` (pair_keep): fold f excludes the pairs of fold f, plus 'full'
    (all pairs). Returns folds [5, 37, 2048], full [37, 2048], and the meta/acts for scoring."""
    m, x = load_lang(lang)
    train = (m.tier == "H_core") & m.pair_keep
    diff = lambda t: unit(x[(t & (m.polarity == "harmful")).values].mean(0) - x[(t & (m.polarity == "benign")).values].mean(0))
    folds = torch.stack([diff(train & (m.pair_id.map(FMAP) != f)) for f in range(K_FOLD)])
    return folds, diff(train), m, x


def control_dir(lang="en"):
    """food - travel over all 64 control pairs (no folds needed: control pairs never appear in the benchmark)."""
    p = DATA_FROZEN / "control_neutral_v1.jsonl"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == CTRL_SHA, "wrong control file"
    ctrl = pd.read_json(p, lines=True).set_index("uid")
    d = torch.load(ACTIVATIONS / f"act_tinyaya_base_v1_ctrl_{lang}_orig.pt", weights_only=False)
    cm = ctrl.loc[d["uids"], ["pair_id", "side"]].reset_index()
    x = d["acts"][:, 0].float()
    f = cm[cm.side == "food"].sort_values("pair_id").index.values
    t = cm[cm.side == "travel"].sort_values("pair_id").index.values
    assert (cm.pair_id.values[f] == cm.pair_id.values[t]).all()
    return unit((x[f] - x[t]).mean(0))      # [37, 2048]


if __name__ == "__main__":
    en_f, en_full, em, ex = refusal_dirs("en")
    hi_f, hi_full, _, _ = refusal_dirs("hi")
    ctrl = control_dir("en")
    rand = unit(torch.randn(37, 2048, generator=torch.Generator().manual_seed(0)))

    # sanity: our refit of the English fold directions must reproduce the saved Phase 3 file exactly
    print("refit vs saved r_en folds, min cosine (want 1.000):", (en_f * R["folds"]).sum(-1)[:, SLOTS].min().item())

    # Delta_k = English separation in RAW score units (cross-fitted: each row scored by the fold excluding its pair).
    # SafeTune 'steer' strength alpha is in exactly these units at the hooked slot -> alpha / Delta = fraction of separation.
    core = ((em.tier == "H_core") & em.pair_keep).values
    S = (ex * en_f[em.pair_id.map(FMAP).values]).sum(-1).numpy()             # [n, 37]
    h, b = core & (em.polarity == "harmful").values, core & (em.polarity == "benign").values
    tab = pd.DataFrame({"mu_harm": S[h].mean(0), "mu_ben": S[b].mean(0), "sd_ben": S[b].std(0),
                        "delta": S[h].mean(0) - S[b].mean(0)}).loc[SLOTS]
    cos = lambda a, c: (a * c).sum(-1)[SLOTS].numpy()
    tab["cos_en_hi"], tab["cos_en_ctrl"] = cos(en_full, hi_full), cos(en_full, ctrl)
    tab["cos_hi_ctrl"], tab["cos_en_rand"] = cos(hi_full, ctrl), cos(en_full, rand)
    print(tab.round(3).rename_axis("slot"))

    i = torch.tensor(SLOTS)
    torch.save(dict(slots=SLOTS, st_idx=[st_idx(k) for k in SLOTS], fold_map=FMAP,
                    en_folds=en_f[:, i], en_full=en_full[i], hi_folds=hi_f[:, i], hi_full=hi_full[i],
                    ctrl=ctrl[i], rand=rand[i], delta_en=torch.tensor(tab.delta.values),
                    note="unit vectors [*, 3 slots, 2048]; slot order = slots; SafeTune layers=[st_idx]"),
               DIRECTIONS / "dir_tinyaya_steer_v1.pt")
    print("saved", DIRECTIONS / "dir_tinyaya_steer_v1.pt")
