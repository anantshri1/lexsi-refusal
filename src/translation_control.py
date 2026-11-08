"""Phase 2 — translation control. Judge (j1 @ t512) Tiny Aya's answers to MultiJail HUMAN Italian (itHT), then compare
H_core compliance with our NLLB Italian (it) and with en on shared pair_ids (it pair_keep). Resumable judge step."""
import pandas as pd
import judge as J
from generate import load_itHT
from rates import load, paired_diff
from paths import GENERATIONS

GEN_HT, TAG = GENERATIONS / "gen_tinyaya_itHT_v1.jsonl", "qwen72b-j1"      # SAME judge as the headline
ht = load_itHT()
J.run_judge(GEN_HT, ht, cut=512, tag=TAG)

j = J.load_judgements(J.judge_path(GEN_HT, 512, TAG))[["label"]]
d_ht = ht.merge(j, left_on="uid", right_index=True).assign(lang="itHT", polarity="harmful", variant="original")
print("itHT unparsed:", d_ht.label.isna().sum()); d_ht = d_ht[d_ht.label.notna()]

d = load()                                                 # benchmark rows, j1 t512, pair_keep, original
it_pairs = set(d[d.lang == "it"].pair_id)
d_ht = d_ht[d_ht.pair_id.isin(it_pairs)]                   # human-it rows only where NLLB-it was understood
dd = pd.concat([d[d.lang.isin(["en", "it"])], d_ht], ignore_index=True)
res = [paired_diff(dd, "itHT", ref="it"), paired_diff(dd, "itHT", ref="en"), paired_diff(dd, "it", ref="en")]
print(pd.DataFrame(res).round(3).to_string(index=False))
print("\nlabel counts itHT:", d_ht.label.value_counts().to_dict())
