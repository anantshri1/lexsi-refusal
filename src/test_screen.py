"""Mac check: constants duplicated in src/screen_job.py agree with paths.py / generate.py / steer_prep.py / split file."""
import sys; sys.path.insert(0, "src")
import json
import paths, generate, steer_prep as P, screen_job as J
assert J.MODEL_ID == paths.MODEL_ID and J.BENCH_REPO == paths.HF_BENCH_REPO and J.ART_REPO == paths.HF_ARTEFACT_REPO
assert J.BENCH_SHA == generate.BENCH_SHA and J.CTRL_SHA == P.CTRL_SHA
assert J.build_messages("t") == generate.build_messages("t", "base")
assert J.SLOTS == P.SLOTS and [J.st_idx(k) for k in J.SLOTS] == [17, 20, 23]
s = json.load(open(paths.PHASE4 / "screen" / "split_v1.json"))
assert J.SPLIT_FILE == "phase4/screen/split_v1.json" and len(s["dev_uids"]) == 6 * len(s["dev_pairs"])
assert {str(k): int(v) for k, v in P.FMAP.items()} == s["fold_map"], "split fold map != Phase 3 fold map"
print("screen_job constants / template / slots / split fold map == Mac ✔")
ids = [c[0] for c in J.CONDS]
assert len(ids) == len(set(ids)) == 14 and ids[:3] == ["steer-ren-s24-a025", "steer-ctrl-s24-a025", "steer-rand-s24-a025"]
assert all(c[2] in J.SLOTS for c in J.CONDS) and {c[1] for c in J.CONDS} == {"ren", "rhi", "ctrl", "rand"}
print("CONDS:", ids)
