"""Phase 4 figure (figures/p4_tradeoff_v1.png). Static PNG for the write-up (no hover layer).
 A  FULL runs, Hindi: H_core compliance (j1, 95% pair-bootstrap CI) vs human-corrected over-refusal, one point per condition.
 B  DEV screening, Hindi: H_core compliance vs alpha/Delta per slot (18/21/24), forecast for slot 24 (dashed, block 0.3),
    food/travel control and random direction at slot 24, 0.25 (kill-test). Dev (t256, 55 pairs) and full (t512) are NOT mixed.
Colours: validated default categorical slots 1-3 (dataviz skill palette) for the three slots; baselines in neutral grey."""
import sys; sys.path.insert(0, "src")
import pandas as pd, matplotlib.pyplot as plt
from rates import load, metrics, ci
from paths import GENERATIONS, PHASE4, FIGURES

INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1-3 (blue, orange, aqua)
GEN = {"base": GENERATIONS / "gen_tinyaya_base_v1.jsonl", "sysprompt": GENERATIONS / "gen_tinyaya_sysprompt_v1.jsonl",
       "translate_en": GENERATIONS / "gen_tinyaya_translate_en_v1.jsonl",
       "steer r_en (F1)": PHASE4 / "full" / "gen_tinyaya_steer-ren-s24-a020_full_v1.jsonl",
       "steer r_hi (F2)": PHASE4 / "full" / "gen_tinyaya_steer-rhi-s24-a020_full_v1.jsonl",
       "steer r_en, hi 0.3 (P4.9)": PHASE4 / "full" / "gen_tinyaya_steer-ren-s24-a030_fullhi_v1.jsonl"}
OVERREF_HI = {"base": 0.000, "sysprompt": 0.008, "translate_en": 0.054, "steer r_en (F1)": 0.096, "steer r_hi (F2)": 0.098,
             "steer r_en, hi 0.3 (P4.9)": None}  # canonical; P4.9 value from audit_overref correct (fill, as a fraction)
COL = {"base": GREY, "sysprompt": GREY, "translate_en": GREY, "steer r_en (F1)": S1, "steer r_hi (F2)": S2, "steer r_en, hi 0.3 (P4.9)": S1}
OFF = {"base": (6, 4), "sysprompt": (6, 4), "translate_en": (6, -12), "steer r_en (F1)": (16, -14), "steer r_hi (F2)": (12, 4),
       "steer r_en, hi 0.3 (P4.9)": (10, -4)}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.28})

# ---- A: full-run trade-off ----
pts = {}
for name, g in GEN.items():
    if OVERREF_HI[name] is None:
        print("skip (over-refusal not filled):", name); continue
    d = load(gen=g)
    h = d[d.lang == "hi"]
    y = metrics(h)["core_comp_cond"]
    lo, hi = ci(d, "hi", "core_comp_cond")
    x = OVERREF_HI[name]
    a.errorbar(100 * x, 100 * y, yerr=[[100 * (y - lo)], [100 * (hi - y)]], fmt="o", ms=8, color=COL[name], ecolor=COL[name],
               elinewidth=2, capsize=0, mec="white", mew=2, zorder=3)
    pts[name] = (100 * x, 100 * y)
    a.annotate(name, (100 * x, 100 * y), xytext=OFF[name], textcoords="offset points", color=INK, fontsize=9,
               ha="right" if OFF[name][0] < 0 else "left")                    # F1/F2 sit 0.2 pp apart: both labels right of the orange bar, F2 above, F1 below
if "steer r_en, hi 0.3 (P4.9)" in pts:                                 # same direction, two strengths: a short curve
    (x0, y0), (x1, y1) = pts["steer r_en (F1)"], pts["steer r_en, hi 0.3 (P4.9)"]
    a.plot([x0, x1], [y0, y1], "-", color=S1, lw=1, alpha=0.6, zorder=2)
    a.set_xlim(-1, max(13, x1 + 4))
a.set_xlabel("Hindi over-refusal, human-corrected (%)")
a.set_ylabel("Hindi harmful-request compliance (%)")
a.set_title("A  Full runs (Hindi, 512 tokens): lower-left is better", loc="left", color=INK, fontsize=10)
a.set_ylim(0, 35)
if "steer r_en, hi 0.3 (P4.9)" not in pts:
    a.set_xlim(-1, 13)                                                  # (widened above when the P4.9 point is drawn)
a.grid(color=GRID, linewidth=0.8); a.set_axisbelow(True)

# ---- B: dev screening dose-response ----
s = pd.read_csv(FIGURES / "p4_screen_v1.csv")
s = s[s.lang == "hi"]
base = s[s.cond == "base"].comp.iloc[0]
for slot, c in [(18, S3), (21, S2), (24, S1)]:
    g = s[(s.direction == "ren") & (s["mode"] == "steer") & (s.slot == slot)].sort_values("frac")
    b.plot([0] + g.frac.tolist(), [100 * base] + (100 * g.comp).tolist(), "-o", color=c, lw=2, ms=6, mec="white", mew=1.5,
           label=f"r_en, slot {slot}", zorder=3)
f = pd.read_csv(FIGURES / "p4_forecast_v1.csv")
f = f[(f.lang == "hi") & (f.slot == 24) & (f.frac <= 0.3)]
b.plot(f.frac, 100 * f.comp, "--", color=S1, lw=1.5, alpha=0.8, label="forecast, slot 24 (block 0.3)")
for cid, lab, mk in [("steer-ctrl-s24-a025", "food/travel control, slot 24", "s"), ("steer-rand-s24-a025", "random direction, slot 24", "^")]:
    v = s[s.cond == cid].comp.iloc[0]
    b.plot([0.25], [100 * v], mk, color=GREY, ms=8, mec="white", mew=1.5, label=lab, zorder=3)
b.axhline(100 * base, color=GREY, lw=1, ls=":")
b.annotate("base", (0.29, 100 * base), xytext=(0, 3), textcoords="offset points", color=INK2, fontsize=8)
b.set_xlabel("push strength  alpha / Delta (fraction of English harmful-benign separation)")
b.set_ylabel("Hindi harmful-request compliance (%)")
b.set_title("B  Dev screening (55 pairs, 256 tokens)", loc="left", color=INK, fontsize=10)
b.set_ylim(0, 35); b.set_xlim(-0.01, 0.31)
b.grid(color=GRID, linewidth=0.8); b.set_axisbelow(True)
b.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)   # keep the kill-test markers (top right) visible

fig.text(0.01, -0.07, "Tiny Aya Global. Steering = SafeTune RefusalDirectionModel 'steer', English refusal direction (cross-fitted), "
         "slot 24, alpha/Delta 0.2.\nCompliance: j1 judge (Qwen2.5-72B). Over-refusal: j1 flags + human audit. "
         "Error bars: 95% pair bootstrap (compliance only; corrected over-refusal has no CI).",
         fontsize=7.5, color=INK2)                                      # two lines: one long line widened the tight bbox
out = FIGURES / "p4_tradeoff_v1.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
print("saved", out)
