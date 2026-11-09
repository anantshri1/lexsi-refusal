import sys; sys.path.insert(0, "src")
import paths, generate, extract_activations as X
assert X.MODEL_ID == paths.MODEL_ID and X.BENCH_REPO == paths.HF_BENCH_REPO and X.ART_REPO == paths.HF_ARTEFACT_REPO
assert X.BENCH_SHA == generate.BENCH_SHA
assert X.build_messages("t") == generate.build_messages("t", "base")
print("extract_activations constants/template == paths.py/generate.py ✔")