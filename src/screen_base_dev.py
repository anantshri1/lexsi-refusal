"""Phase 4 block 3.4a — the BASE condition on the 330 dev rows at 256 tokens, as a screening-format file, so WildGuard (3.4b) scores
base and steered outputs identically. Greedy decoding -> the first 256 generated tokens of the Phase 2 t512 run ARE the t256 run
(strength-0 equivalence verified in step 1 D). ids truncated to 256 and re-decoded with the Tiny Aya tokenizer.
Output: PHASE4/screen/gen_tinyaya_base_dev_v1.jsonl (+ upload to ART_REPO phase4/screen/)."""
import sys; sys.path.insert(0, "src")
import json
import pandas as pd
from transformers import AutoTokenizer
from paths import GENERATIONS, PHASE4, HF_ARTEFACT_REPO, MODEL_ID

CUT = 256
split = json.load(open(PHASE4 / "screen" / "split_v1.json"))
g = pd.read_json(GENERATIONS / "gen_tinyaya_base_v1.jsonl", lines=True).set_index("uid").loc[split["dev_uids"]].reset_index()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
rows = []
for r in g.itertuples():
    ids = list(r.ids[:CUT])
    rows.append(dict(uid=r.uid, model="tinyaya", condition="base", max_new_tokens=CUT, ids=ids, n_new_tokens=len(ids),
                     hit_eos=bool(r.hit_eos and len(r.ids) <= CUT),
                     response=r.response if len(r.ids) <= CUT else tok.decode(ids, skip_special_tokens=True).strip(),
                     direction=None, slot=None, st_idx=None, layers=None, alpha=None, frac=None, mode="none", fold=None,
                     vec_source="phase2 base t512, first 256 tokens"))
assert len(rows) == 330
out = PHASE4 / "screen" / "gen_tinyaya_base_dev_v1.jsonl"
out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
print(f"{len(rows)} rows | truncated to {CUT}: {sum(len(x) > CUT for x in g.ids)} | hit_eos: {sum(x['hit_eos'] for x in rows)} -> {out}")
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=str(out), path_in_repo="phase4/screen/gen_tinyaya_base_dev_v1.jsonl",
                    repo_id=HF_ARTEFACT_REPO, repo_type="dataset", commit_message="Phase 4 block 3.4a: base dev @256")
print("uploaded")
