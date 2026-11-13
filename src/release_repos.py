"""Ship block S4: push the two dataset cards, then flip visibility (D-S1: bench public & open; artefacts public & gated, auto-approve).
Default is a dry run; pass --go. Gating is set BEFORE the artefact repo goes public, so it is never open even for a moment."""
import sys
from huggingface_hub import HfApi
from paths import HF_ARTEFACT_REPO as ART, HF_BENCH_REPO as BENCH, DATA_FROZEN, REPO_ROOT

GO = "--go" in sys.argv
CARDS = [(BENCH, DATA_FROZEN / "README.md"), (ART, REPO_ROOT / "cards" / "artefacts_README.md")]
for _, p in CARDS:
    assert "[CONFIRM" not in p.read_text(), f"unresolved [CONFIRM] placeholder in {p}"

api = HfApi()
for repo, p in CARDS:
    print(f"{'UPLOAD' if GO else 'dry'}  {p.relative_to(REPO_ROOT)} -> {repo}/README.md")
    if GO:
        api.upload_file(path_or_fileobj=str(p), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                        commit_message="Ship S4: dataset card")
STEPS = [(ART, dict(gated="auto")), (ART, dict(private=False)), (BENCH, dict(private=False))]
for repo, kw in STEPS:
    print(f"{'SET' if GO else 'dry'}  {repo}: {kw}")
    if GO:
        api.update_repo_settings(repo_id=repo, repo_type="dataset", **kw)
if GO:
    for repo in (BENCH, ART):
        info = api.dataset_info(repo)
        print(f"NOW  {repo}: private={info.private} gated={info.gated}")
else:
    print("dry run only: rerun with --go")
