import pandas as pd, judge as J
from generate import load_bench
from paths import GENERATIONS, MODEL_ID
from transformers import AutoTokenizer

"""
bench = load_bench().set_index("uid")
g = pd.read_json(GENERATIONS / "gen_tinyaya_base_v0.jsonl", lines=True)
print(len(J.SYSTEM), "chars in system prompt")

test = g.merge(bench[["lang", "polarity"]], left_on="uid", right_index=True)
pick = pd.concat([test[(test.polarity == "harmful") & (test.lang == "en")].head(1),
                  test[test.polarity == "benign"].head(1),
                  test[test.lang == "ml"].head(1)])
for r in pick.itertuples():
    t = time.time()
    out = J.judge_one(bench.loc[r.uid, "text"], r.response, not r.hit_eos)
    print(f"\n{r.uid}  ({time.time()-t:.1f}s)  label={out['label']}")
    print("reason:", out.get("reason"))
    print("extra:", {k: out.get(k) for k in ["response_lang", "has_disclaimer", "offers_alternative", "restates_prompt"]})
print("\nTESTED UIDS (exclude from audit):", pick.uid.tolist())
"""

tok = AutoTokenizer.from_pretrained(MODEL_ID)
bench = load_bench()                                  # NOT set_index: run_judge does that itself
v0 = GENERATIONS / "gen_tinyaya_base_v0.jsonl"
print(J.judge_path(v0, 64).name)

for cut in [64, 512]:
    J.run_judge(v0, bench, cut=cut, tok=tok, workers=4)

a = pd.read_json(J.judge_path(v0, 64), lines=True).set_index("uid")
b = pd.read_json(J.judge_path(v0, 512), lines=True).set_index("uid")
print("parsed:", a.label.notna().mean(), b.label.notna().mean(), "| truncated@64:", a.truncated.mean().round(2))
print("same label 64 vs 512:", (a.label == b.label.reindex(a.index)).mean().round(2))