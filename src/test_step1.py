"""Mac check: constants duplicated in src/steer_step1_job.py agree with paths.py / generate.py / the data."""
import sys; sys.path.insert(0, "src")
import pandas as pd, torch
import paths, generate, steer_step1_job as J, steer_prep as P
assert J.MODEL_ID == paths.MODEL_ID and J.BENCH_REPO == paths.HF_BENCH_REPO and J.ART_REPO == paths.HF_ARTEFACT_REPO
assert J.BENCH_SHA == generate.BENCH_SHA and J.build_messages("t") == generate.build_messages("t", "base")
assert J.SLOTS == P.SLOTS and [J.st_idx(k) for k in J.SLOTS] == [P.st_idx(k) for k in P.SLOTS] == [17, 20, 23]
c = pd.read_csv(paths.COMPREHENSION / "comprehension_v1.csv"); b = generate.load_bench().merge(c[["uid", "pair_keep"]], on="uid")
e = b[(b.lang == "en") & (b.variant == "original") & (b.tier == "H_core")]
assert set(e[~e.pair_keep].pair_id) == J.EN_DROP, set(e[~e.pair_keep].pair_id)
V = torch.load(paths.DIRECTIONS / "dir_tinyaya_steer_v1.pt", weights_only=False)
assert V["slots"] == J.SLOTS and V["st_idx"] == [17, 20, 23]
print("steer_step1_job constants / template / EN_DROP / steer file == Mac ✔")
