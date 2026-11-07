import pandas as pd
from paths import GENERATIONS
from generate import load_bench
g = pd.read_json(GENERATIONS / "gen_tinyaya_base_v1.jsonl", lines=True).merge(
        load_bench()[["uid", "lang", "polarity"]], on="uid")
print(g.groupby(["lang", "polarity"]).agg(eos=("hit_eos", "mean"), med_tok=("n_new_tokens", "median"))
       .round(2).unstack())