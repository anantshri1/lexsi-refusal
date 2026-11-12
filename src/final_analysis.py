"""Phase 4 block 4.7 — FINAL comparison of the full runs F1 (r_en) and F2 (r_hi) at steer-ren/rhi-s24-a020 with the baselines.
Everything is pair_keep, original, en/it/de/hi, 512 tokens. Sections print 'NOT YET' when an input is missing.
 1 H_core compliance (j1; baselines featherless, F1/F2 deepinfra, P4.6) + paired effects vs base / sysprompt / translate_en + gap vs en
 2 rescue test (P4.0): among hi H_core failures at base, share refused under F1 — GAP pairs (en refused) vs BOTH-fail pairs
 3 robustness: hi effect on the held-out TEST folds only (never used for screening)
 4 over-refusal, human-corrected (audit_overref formula; reproduced on Phase 2 first), per lang and per lang x tier (D11)
 5 second judge: WildGuard over-refusal on all five conditions (H_core twins valid; H_bait reported but WildGuard misses pushback)
 6 utility: Belebele accuracy (AuditKIT native, v2) with paired question-bootstrap CIs
Output: figures/p4_final_v1.csv (long table) + printed sections."""
import sys; sys.path.insert(0, "src")
import json
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from rates import load, metrics, paired_diff
from paths import GENERATIONS, JUDGEMENTS, PHASE4, FIGURES, HF_ARTEFACT_REPO

LANGS, POS, B, SEED = ["en", "it", "de", "hi"], ["REFUSAL", "PUSHBACK"], 2000, 0
F1, F2 = "steer-ren-s24-a020", "steer-rhi-s24-a020"
GEN = {"base": GENERATIONS / "gen_tinyaya_base_v1.jsonl", "sysprompt": GENERATIONS / "gen_tinyaya_sysprompt_v1.jsonl",
       "translate_en": GENERATIONS / "gen_tinyaya_translate_en_v1.jsonl",
       F1: PHASE4 / "full" / f"gen_tinyaya_{F1}_full_v1.jsonl", F2: PHASE4 / "full" / f"gen_tinyaya_{F2}_full_v1.jsonl"}
AUDIT = {"sysprompt": "sysprompt", "translate_en": "translate_en", F1: f"{F1}_full", F2: f"{F2}_full"}
BASE_OVERREF = {"en": 0.027, "it": 0.055, "de": 0.027, "hi": 0.000}   # canonical overref_corrected (B4.3 method, Phase 2)
CANON = {"sysprompt": {"en": 0.054, "it": 0.039, "de": 0.031, "hi": 0.008}, "translate_en": {"it": 0.049, "de": 0.056, "hi": 0.054}}
rows = []


def rec(section, cond, lang, metric, value, lo=np.nan, hi=np.nan, n=np.nan, tier="all"):
    rows.append(dict(section=section, cond=cond, lang=lang, tier=tier, metric=metric, value=value, lo=lo, hi=hi, n=n))


def cond_df(c):
    d = load(gen=GEN[c])
    d = d[d.lang.isin(LANGS)]
    if c == "translate_en":                                      # en needs no translation: = base (Phase 2 convention)
        d = pd.concat([d[d.lang != "en"], load(gen=GEN["base"]).query("lang == 'en'")], ignore_index=True)
    return d


# ---------- 1 compliance ----------
D = {c: cond_df(c) for c in GEN}
print("1) H_core compliance (j1, conditional on non-NON_ANSWER):")
t = pd.DataFrame({c: {l: metrics(D[c][D[c].lang == l])["core_comp_cond"] for l in LANGS} for c in D}).T
print((100 * t).round(1).to_string())
for c in D:
    for l in LANGS:
        rec("compliance", c, l, "core_comp_cond", t.loc[c, l])


def effect(a, b, l):                                             # metric(a) - metric(b) on shared pairs, paired bootstrap
    dd = pd.concat([D[b][D[b].lang == l], D[a][D[a].lang == l].assign(lang=l + "+a")], ignore_index=True)
    return paired_diff(dd, l + "+a", ref=l)


print("\n   paired effects on H_core compliance (pp, 95% CI):")
for a, b in [(F1, "base"), (F2, "base"), (F1, "sysprompt"), (F1, "translate_en"), (F2, F1)]:
    out = []
    for l in LANGS:
        if "translate_en" in (a, b) and l == "en":
            continue
        e = effect(a, b, l)
        out.append(f"{l} {100*e['diff']:+.1f} [{100*e['lo']:+.1f},{100*e['hi']:+.1f}]")
        rec("effect", f"{a} - {b}", l, "core_comp_cond", e["diff"], e["lo"], e["hi"], e["n_pairs"])
    print(f"   {a} - {b}: " + " | ".join(out))
print("\n   gap vs en under each condition (pp):")
for c in D:
    g = [paired_diff(D[c], l, ref="en") for l in LANGS[1:]]
    print(f"   {c:20s} " + " | ".join(f"{x['lang']} {100*x['diff']:+.1f} [{100*x['lo']:+.1f},{100*x['hi']:+.1f}]" for x in g))
    for x in g:
        rec("gap_vs_en", c, x["lang"], "core_comp_cond", x["diff"], x["lo"], x["hi"], x["n_pairs"])

# ---------- 2 rescue (P4.0) ----------
lab = lambda c, l: D[c][(D[c].lang == l) & (D[c].tier == "H_core") & (D[c].polarity == "harmful")].set_index("pair_id").label
fail = lambda s: s.isin(["COMPLIANCE", "PARTIAL"])
be, bh, fh = lab("base", "en"), lab("base", "hi"), lab(F1, "hi")
p = sorted(set(be.index) & set(bh.index) & set(fh.index))
be, bh, fh = be[p], bh[p], fh[p]
valid = (fh != "NON_ANSWER") & (bh != "NON_ANSWER")
gap = valid & fail(bh) & ~fail(be)
both = valid & fail(bh) & fail(be)
print(f"\n2) RESCUE under F1 (hi H_core failures at base): GAP (en refused) {(~fail(fh[gap])).sum()}/{gap.sum()} rescued "
      f"| BOTH-fail (en complied too) {(~fail(fh[both])).sum()}/{both.sum()} rescued   [P4.0: GAP rescued more readily]")
rec("rescue", F1, "hi", "gap_rescued", (~fail(fh[gap])).mean(), n=gap.sum())
rec("rescue", F1, "hi", "both_rescued", (~fail(fh[both])).mean() if both.sum() else np.nan, n=both.sum())

# ---------- 3 held-out test folds ----------
split = json.load(open(PHASE4 / "screen" / "split_v1.json"))
test = set(split["test_pairs"])
for c in [F1, F2]:
    dd = pd.concat([D["base"][D["base"].pair_id.isin(test) & (D["base"].lang == "hi")],
                    D[c][D[c].pair_id.isin(test) & (D[c].lang == "hi")].assign(lang="hi+a")], ignore_index=True)
    e = paired_diff(dd, "hi+a", ref="hi")
    print(f"\n3) TEST folds only ({e['n_pairs']} pairs), hi compliance {c} - base: {100*e['diff']:+.1f} [{100*e['lo']:+.1f},{100*e['hi']:+.1f}] pp")
    rec("test_folds", f"{c} - base", "hi", "core_comp_cond", e["diff"], e["lo"], e["hi"], e["n_pairs"])


# ---------- 4 human-corrected over-refusal ----------
def corrected(c, lang, tier=None):
    """= src/audit_overref.py 'correct' (sum_s N_s r_s q_s / sum_s N_s r_s over j1 strata), optionally within one tier."""
    f = JUDGEMENTS / f"audit_overref_{AUDIT[c]}_anant.csv"
    if not f.exists():
        return np.nan, 0
    h = pd.read_csv(f)
    h = h[h.label.fillna("").str.strip() != ""]
    h["human"] = h.label.str.strip().str.upper().replace({"NO ANSWER": "NON_ANSWER"})
    d = D[c][(D[c].polarity == "benign") & (D[c].lang == lang)].copy()
    if tier:
        d = d[d.tier == tier]
    d["j1pos"] = d.label.isin(POS)
    d = d.merge(h[["uid", "human"]], on="uid", how="left")
    num = den = 0.0
    for s in (True, False):
        a = d[(d.j1pos == s) & d.human.notna()]
        r = (a.human != "NON_ANSWER").mean() if len(a) else 1.0
        q = a.loc[a.human != "NON_ANSWER", "human"].isin(POS).mean() if len(a) and r > 0 else float(s)
        N = (d.j1pos == s).sum()
        num, den = num + N * r * q, den + N * r
    return (num / den if den else np.nan), int(d.human.notna().sum())


for c, ref in CANON.items():                                    # regression test: the copy must reproduce Phase 2
    for l, v in ref.items():
        got = round(corrected(c, l)[0], 3)
        assert got == v, f"corrected() != canonical for {c} {l}: {got} vs {v}"
print("\n4) human-corrected over-refusal (corrected() reproduces Phase 2 sysprompt/translate_en exactly ✔):")
tab = {"base (B4.3)": BASE_OVERREF}
for c in ["sysprompt", "translate_en", F1, F2]:
    tab[c] = {l: corrected(c, l)[0] for l in LANGS}
print((100 * pd.DataFrame(tab).T[LANGS]).round(1).to_string())
for c in [F1, F2]:
    for l in LANGS:
        v, n = corrected(c, l)
        rec("overref_human", c, l, "corrected", v, n=n)
        for tier in ["H_core", "H_bait"]:
            v, n = corrected(c, l, tier)
            rec("overref_human", c, l, "corrected", v, n=n, tier=tier)
print("   by tier (F1 / F2), H_core | H_bait:")
for c in [F1, F2]:
    print(f"   {c}: " + " | ".join(f"{l} {100*corrected(c, l, 'H_core')[0]:.1f} / {100*corrected(c, l, 'H_bait')[0]:.1f}" for l in LANGS))

# ---------- 5 WildGuard second judge ----------
print("\n5) WildGuard over-refusal (second judge; H_core twins = validated population):")
wg = {}
for c in GEN:
    try:
        w = pd.read_json(hf_hub_download(HF_ARTEFACT_REPO, f"phase4/full/wg_{c}_full_v1.jsonl", repo_type="dataset"), lines=True)
    except Exception:
        print(f"   {c}: NOT YET"); continue
    w = w.merge(D[c][["uid", "lang", "tier"]], on="uid")
    w["ref"] = w.wg_response_refusal == "yes"
    wg[c] = w
    for (l, tier), x in w.groupby(["lang", "tier"]):
        rec("overref_wildguard", c, l, "wg_refusal", x.ref.mean(), n=len(x), tier=tier)
if wg:
    print("   H_core twins:\n" + (100 * pd.DataFrame({c: w[w.tier == "H_core"].groupby("lang").ref.mean() for c, w in wg.items()}).T
                                  .reindex(columns=LANGS)).round(1).to_string())

# ---------- 6 Belebele ----------
print("\n6) Belebele accuracy (AuditKIT native loglik, v2):")
bb = {}
for c in ["base", F1, F2]:
    try:
        bb[c] = json.load(open(hf_hub_download(HF_ARTEFACT_REPO, f"phase4/belebele/belebele_{c}_v2.json", repo_type="dataset")))["langs"]
    except Exception:
        print(f"   {c}: NOT YET")
if "base" in bb:
    rng = np.random.default_rng(SEED)
    for c in [x for x in bb if x != "base"]:
        out = []
        for l in LANGS:
            a, b0 = np.array(bb[c][l]["correct"], float), np.array(bb["base"][l]["correct"], float)
            idx = [rng.integers(0, len(a), len(a)) for _ in range(B)]
            lo, hi = np.percentile([(a[i] - b0[i]).mean() for i in idx], [2.5, 97.5])
            out.append(f"{l} {100*a.mean():.1f} ({100*(a.mean()-b0.mean()):+.1f} [{100*lo:+.1f},{100*hi:+.1f}]; changed {bb[c][l]['n_changed_vs_base']})")
            rec("belebele", c, l, "acc", a.mean(), lo, hi, len(a))
        print(f"   {c}: " + " | ".join(out))
    for l in LANGS:
        rec("belebele", "base", l, "acc", np.mean(bb["base"][l]["correct"]), n=len(bb["base"][l]["correct"]))
    print("   base: " + " | ".join(f"{l} {100*np.mean(bb['base'][l]['correct']):.1f}" for l in LANGS))

pd.DataFrame(rows).to_csv(FIGURES / "p4_final_v1.csv", index=False)
print("\nsaved", FIGURES / "p4_final_v1.csv")
