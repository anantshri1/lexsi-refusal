"""Phase 4 block 0.3 — FORECAST (no GPU). Assumptions (flagged, tested later by real steering):
 (1) SafeTune 'steer' at slot k only shifts each prompt's slot-k score by alpha (exact at that slot; downstream = assumed);
 (2) each language keeps its baseline score->refusal mapping, fitted as a logistic on base j1 behaviour (H_core harmful);
 (3) benign prompts follow the SAME mapping as harmful ones (strong; checked at alpha=0 vs human-corrected over-refusal ~0-5%).
Scores are cross-fitted (each row scored by the r_en fold that excluded its pair)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from geometry import load_lang
from rates import load
from steer_prep import SLOTS, FMAP
from paths import DIRECTIONS, FIGURES

V = torch.load(DIRECTIONS / "dir_tinyaya_steer_v1.pt", weights_only=False)
LANGS = ["en", "it", "de", "hi"]


def logistic(s, y, l2=1.0):
    """P(refuse | s) = sigmoid(w*s + c). Penalised MLE (0.5*l2*w^2 on the slope only, = sklearn C=1 default), so the
    fit stays finite when the classes separate perfectly (en: row AUROC 1.00). Returns (w, c)."""
    s, y = torch.tensor(s, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    p = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([p], max_iter=500, tolerance_grad=1e-10, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(p[0] * s + p[1], y, reduction="sum") + 0.5 * l2 * p[0] ** 2
        loss.backward(); return loss
    opt.step(closure)
    return p[0].item(), p[1].item()


sig = lambda z: 1 / (1 + np.exp(-z))
FRACS = np.round(np.arange(0, 1.51, 0.01), 2)                       # alpha / Delta_k
lab = load().set_index("uid").label                                 # base, j1 @ t512, pair_keep, original

rows, fits = [], []
for L in LANGS:
    m, x = load_lang(L)
    S = (x[:, SLOTS] * V["en_folds"][m.pair_id.map(FMAP).values]).sum(-1).numpy()      # [n, 3]
    m["label"] = lab.reindex(m.uid).values
    core = ((m.tier == "H_core") & m.pair_keep).values
    h = core & (m.polarity == "harmful").values & m.label.notna().values & (m.label != "NON_ANSWER").values
    b = core & (m.polarity == "benign").values
    refused = ~m.label.isin(["COMPLIANCE", "PARTIAL"]).values
    for j, k in enumerate(SLOTS):
        s = S[:, j]
        w, c = logistic(s[h], refused[h])
        fits.append(dict(lang=L, slot=k, n_harm=h.sum(), n_fail=(~refused[h]).sum(), mid=-c / w, width=4 / w))
        for f in FRACS:
            a = f * V["delta_en"][j].item()
            rows.append(dict(lang=L, slot=k, frac=f, alpha=a,
                             comp=1 - sig(w * (s[h] + a) + c).mean(),
                             overref=sig(w * (s[b] + a) + c).mean()))

F, FIT = pd.DataFrame(rows), pd.DataFrame(fits)
F.to_csv(FIGURES / "p4_forecast_v1.csv", index=False)
print("score->refusal logistic per language (mid = score where P(refuse)=0.5; width = score range from 12% to 88%):")
print(FIT.round(2).to_string(index=False))
show = F[F.frac.isin([0, .1, .2, .3, .5, .75, 1.0])]
for col in ["comp", "overref"]:
    print(f"\nforecast {col} (%) by slot x alpha/Delta:")
    print((100 * show.pivot_table(index=["slot", "frac"], columns="lang", values=col)).round(1)[LANGS].to_string())
print("\nsmallest alpha/Delta reaching the hi bars, and forecast over-refusal there:")
for bar, name in [(0.151, "sysprompt 15.1"), (0.087, "translate_en 8.7")]:
    for k in SLOTS:
        g = F[(F.slot == k) & (F.lang == "hi") & (F.comp <= bar)]
        if len(g) == 0:
            print(f"  {name:17s} slot {k}: not reached by alpha/Delta 1.5"); continue
        f0 = g.frac.min(); at = F[(F.slot == k) & (F.frac == f0)].set_index("lang")
        print(f"  {name:17s} slot {k}: alpha/Delta {f0:.2f} (alpha {at.alpha.iloc[0]:.1f}) | over-refusal "
              + " ".join(f"{L} {100 * at.overref[L]:.1f}" for L in LANGS))
