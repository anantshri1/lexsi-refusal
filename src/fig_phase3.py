"""Phase 3 figure: (A) separation d' along r_en by layer; (B) transfer refusal vs topic control; (C) twin extra loss by layer."""
import sys; sys.path.insert(0, "src")
import pandas as pd, matplotlib.pyplot as plt
from paths import FIGURES

COL = {"en": "#2a78d6", "it": "#eb6834", "de": "#1baf7a", "hi": "#eda100", "ml": "#e87ba4", "hi-Latn": "#008300"}
LS = {L: ("--" if L in ("ml", "hi-Latn") else "-") for L in COL}            # dashed = descriptive (not understood)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

c = pd.read_csv(FIGURES / "p3_killtest_curves_v1.csv")
c = c[(c.score == "raw") & (c.keep == ~c.lang.isin(["ml", "hi-Latn"])) & (c.slot > 0)]   # pair_keep; ml/hi-Latn all pairs
for L, g in c.groupby("lang", sort=False):
    ax[0].plot(g.slot, g.dprime, LS[L], color=COL[L], lw=2, label=L)
    ax[0].annotate(L, (g.slot.iloc[-1], g.dprime.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                   va="center", fontsize=9, color="#3d3d3a")
ax[0].axvline(24, color="#888780", lw=1, ls=":"); ax[0].text(24.5, 0.15, "k* = 24", fontsize=8, color="#5f5e5a")
ax[0].set(title="A  Harmful vs benign separation along the English direction", xlabel="layer (slot)", ylabel="d′ (held-out)")
ax[0].legend(frameon=False, fontsize=8, loc="upper left")

t = pd.read_csv(FIGURES / "p3_transfer_v1.csv")
t = t[(t.L1 == "en") & t.L2.isin(["it", "de", "hi"])]
for L, g in t.groupby("L2", sort=False):
    ax[1].plot(g.slot, g.refusal, "-o", color=COL[L], lw=2, ms=6, label=f"en–{L} refusal")
    ax[1].plot(g.slot, g.control, ":s", color=COL[L], lw=2, ms=6, mfc="white", label=f"en–{L} topic control")
ax[1].set(title="B  Cross-language transfer: refusal (solid) vs topic control (dotted)", xlabel="layer (slot)",
          ylabel="disattenuated transfer (1 = same direction)", ylim=(0.3, 1.02))
ax[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

w = pd.read_csv(FIGURES / "p3_twin_depth_v1.csv")
ax[2].set_ylim(-0.7, 0.6)
ax[2].axvspan(0.5, 8.5, color="#f1efe8", lw=0)
ax[2].text(1, 0.45, "unreliable:\nEnglish separation ≈ 0", fontsize=8, color="#5f5e5a")
ax[2].axhline(0, color="#888780", lw=1)
for L, g in w.groupby("lang", sort=False):
    ax[2].plot(g.slot, g.extra, "-", color=COL[L], lw=2, label=f"{L} (n GAP = {g.n_gap.iloc[0]})")
ax[2].set(title="C  Extra harm signal lost by prompts that fail in L", xlabel="layer (slot)",
          ylabel="fraction of English separation (median)")
ax[2].legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(FIGURES / "p3_geometry_v1.png", dpi=200)
print("saved", FIGURES / "p3_geometry_v1.png")