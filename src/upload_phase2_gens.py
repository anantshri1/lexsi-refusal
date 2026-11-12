"""Phase 4 block 4.6a — Phase 2 generations to the HF artefact repo (needed by the WildGuard second-judge job, and for shipping:
all artefacts on HF). Skips v0 (throwaway). Idempotent: HF dedupes identical content."""
import sys; sys.path.insert(0, "src")
from huggingface_hub import HfApi
from paths import GENERATIONS, HF_ARTEFACT_REPO

api = HfApi()
for f in sorted(GENERATIONS.glob("gen_tinyaya_*_v1.jsonl")):
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f"generations/{f.name}", repo_id=HF_ARTEFACT_REPO,
                    repo_type="dataset", commit_message=f"Phase 2 generations: {f.name}")
    print("uploaded", f.name, f"{f.stat().st_size/1e6:.1f} MB")
