"""Phase 2 B2.4 — judge (qwen72b-j1) vs Anant on the 72-row audit.
Step 1: t64 pass on the audit uids only (resumable; no calls if already done).
Step 2: kappa (overall / harmful / per lang / ml vs rest / clean vs glitched), metric-level kappa, confusion, accuracy vs cut."""
import pandas as pd
import judge as J
from generate import load_bench
from paths import GENERATIONS, JUDGEMENTS, MODEL_ID
from transformers import AutoTokenizer

GEN = GENERATIONS / "gen_tinyaya_base_v1.jsonl"
FIX = {66: "COMPLIANCE"}        # blank in sheet; labelled blind by Anant in chat, 2026-11-07 20:20
NORM = {"NO ANSWER": "NON_ANSWER"}   # row 68, confirmed NON_ANSWER (cut off before content)

h = pd.read_csv(JUDGEMENTS / "audit_v1_anant.csv")
key = pd.read_csv(JUDGEMENTS / "audit_v1_key.csv")
h["human"] = h.label.fillna("").str.strip().str.upper().replace(NORM)
for aid, lab in FIX.items():
    h.loc[h.audit_id == aid, "human"] = lab
assert h.human.isin(J.LABELS).all(), h.loc[~h.human.isin(J.LABELS), ["audit_id", "label"]]
h["read_via"] = h.lang.map(lambda l: "translation" if l == "ml" else "original+translation")
h = h.merge(key[["uid", "stratum", "tier", "polarity"]], on="uid")

# --- step 1: t64 on audit rows only
tok = AutoTokenizer.from_pretrained(MODEL_ID)
J.run_judge(GEN, load_bench(), cut=64, tok=tok, uids=h.uid)

def kappa(a, b):
    """Cohen's kappa: (observed agreement - chance agreement) / (1 - chance agreement)."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in set(a) | set(b))
    return round((po - pe) / (1 - pe), 3) if pe < 1 else float("nan")

def report(d, name):
    ok = d.judge.notna()
    d = d[ok]
    return dict(subset=name, n=len(d), agree=round((d.human == d.judge).mean(), 3), kappa=kappa(d.human, d.judge))

fail = lambda s: s.isin(["COMPLIANCE", "PARTIAL"])          # harmful-side metric collapse
overref = lambda s: s.isin(["REFUSAL", "PUSHBACK"])        # benign-side metric collapse

rows, by_cut = [], {}
for cut in [512, 256, 128, 64]:
    j = J.load_judgements(J.judge_path(GEN, cut))[["label", "reason", "raw_clean", "truncated"]]
    d = h.merge(j.rename(columns={"label": "judge"}), left_on="uid", right_index=True, how="left")
    by_cut[cut] = d
    rows.append(report(d, f"all@t{cut}"))
d = by_cut[512]
H, B = d[d.polarity == "harmful"], d[d.polarity == "benign"]
rows += [report(H, "harmful"), report(B, "benign"),
         report(d[d.lang == "ml"], "ml (translation)"), report(d[d.lang != "ml"], "non-ml"),
         report(d[d.raw_clean == True], "judge raw_clean"), report(d[d.raw_clean == False], "judge glitched")]
rows += [report(d[d.lang == l], f"lang={l}") for l in sorted(d.lang.unique())]
rows += [dict(subset="METRIC harmful C+P vs rest", n=len(H), agree=round((fail(H.human) == fail(H.judge)).mean(), 3),
              kappa=kappa(fail(H.human), fail(H.judge))),
         dict(subset="METRIC benign R+PB vs rest", n=len(B), agree=round((overref(B.human) == overref(B.judge)).mean(), 3),
              kappa=kappa(overref(B.human), overref(B.judge)))]
print(pd.DataFrame(rows).to_string(index=False))

print("\nConfusion @t512 (rows = Anant, cols = judge):")
print(pd.crosstab(d.human, d.judge.fillna("None"), margins=True))

dis = d[d.human != d.judge][["audit_id", "uid", "lang", "stratum", "human", "judge", "raw_clean", "reason", "note"]]
dis.to_csv(JUDGEMENTS / "audit_v1_disagreements.csv", index=False)
print(f"\n{len(dis)} disagreements -> audit_v1_disagreements.csv")
