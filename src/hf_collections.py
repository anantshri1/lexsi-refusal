"""Ship: submission collections (both public, everything un-gated: D-W2) + un-gate the artefact repo. Default dry run; --go."""
import sys
from huggingface_hub import HfApi
from paths import HF_ARTEFACT_REPO as ART, HF_BENCH_REPO as BENCH, HF_USER

GO = "--go" in sys.argv
STEER = f"{HF_USER}/tiny-aya-global-refusal-steer"
COLL = [("Lexsi multilingual refusal: data",
         "Benchmark (en/it/de/hi/ml/hi-Latn harmful+benign twins) and all artefacts: generations, judgements, human audits, activations, directions",
         [(BENCH, "dataset"), (ART, "dataset")]),
        ("Lexsi multilingual refusal: steering",
         "Refusal steering vectors for CohereLabs/tiny-aya-global (no weights; applied with SafeTune)",
         [(STEER, "model")])]
api = HfApi()
print(f"{'SET' if GO else 'dry'}  {ART}: gated=False")
if GO:
    api.update_repo_settings(repo_id=ART, repo_type="dataset", gated=False)
for title, desc, items in COLL:
    print(f"{'MAKE' if GO else 'dry'}  collection '{title}': {[i for i, _ in items]}")
    if GO:
        c = api.create_collection(title=title, namespace=HF_USER, description=desc, private=False, exists_ok=True)
        for item, kind in items:
            api.add_collection_item(c.slug, item_id=item, item_type=kind, exists_ok=True)
        print("   https://huggingface.co/collections/" + c.slug)
if GO:
    for repo, kind in ((BENCH, "dataset"), (ART, "dataset"), (STEER, "model")):
        i = api.repo_info(repo, repo_type=kind)
        print(f"NOW  {repo}: private={i.private} gated={i.gated}")
