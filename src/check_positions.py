import sys; sys.path.insert(0, "src")
import pandas as pd
from transformers import AutoTokenizer
from paths import MODEL_ID
from generate import load_bench, build_messages

tok = AutoTokenizer.from_pretrained(MODEL_ID)
bench = load_bench()

def positions(text):
    msgs = build_messages(text, "base")
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == ids, "string round-trip != template ids"   # else offsets are meaningless
    end = s.rfind(text) + len(text)                                       # char just after the user text
    p_user = max(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end)
    return ids, p_user

for uid in ["mj000-h-en-orig", "mj000-h-hi-orig", "mj000-h-ml-orig"]:
    text = bench.set_index("uid").loc[uid, "text"]
    ids, p = positions(text)
    print(uid, "| len", len(ids), "| p_user", p, "| p_last", len(ids) - 1)
    print("   user tail :", repr(tok.decode(ids[p-3:p+1])))
    print("   after user:", [tok.decode([t]) for t in ids[p+1:]])

import collections
gaps, lens = collections.Counter(), {}
for r in bench.itertuples():
    ids, p = positions(r.text)
    gaps[len(ids) - 1 - p] += 1
    lens[r.uid] = len(ids)
print("gap counts:", dict(gaps))                       # want {4: 1770}
L = pd.Series(lens); lang = bench.set_index("uid").lang
print(L.groupby(lang).agg(["min", "median", "max"]))   # padding cost per shard