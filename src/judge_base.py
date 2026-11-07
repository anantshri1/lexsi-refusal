import judge as J
from generate import load_bench
from paths import GENERATIONS, MODEL_ID
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
gen = GENERATIONS / "gen_tinyaya_base_v1.jsonl"
bench = load_bench()
for cut in [512, 256, 128]:                      # all rows; t64 later on the audit set only
    J.run_judge(gen, bench, cut=cut, tok=tok, workers=2)