"""Ship block S2: push the local-only artefacts found by inventory_hf.py (S1) to the HF repos.
Repos stay PRIVATE here; visibility + gating is S4. Default is a dry run; pass --go to upload.
  artefacts repo: judgements/ (all), comprehension/ (all), curation/ (= data/interim, D-S3),
                  generations/gen_tinyaya_base_v0.jsonl, directions/dir_tinyaya_refusal_last_en_v1.pt
  bench repo:     control_neutral_v1.jsonl (food/travel control set, frozen)"""
import hashlib, sys
from huggingface_hub import HfApi
from paths import (HF_ARTEFACT_REPO as ART, HF_BENCH_REPO as BENCH, JUDGEMENTS, COMPREHENSION,
                   DATA_INTERIM, GENERATIONS, DIRECTIONS, DATA_FROZEN)

GO = "--go" in sys.argv
CTRL_SHA = "441cd2d412ac1d1a88977c398cf2f3a4b4c89e7a8f91451fbafe6f510b83ba15"   # canonical_numbers.txt
IGNORE = [".DS_Store", ".gitkeep", "*/.DS_Store", "checker_flags_v1"]          # last = extension-less duplicate of the .csv

FOLDERS = [(JUDGEMENTS, "judgements"), (COMPREHENSION, "comprehension"), (DATA_INTERIM, "curation")]
FILES = [(ART, GENERATIONS / "gen_tinyaya_base_v0.jsonl", "generations/gen_tinyaya_base_v0.jsonl"),
         (ART, DIRECTIONS / "dir_tinyaya_refusal_last_en_v1.pt", "directions/dir_tinyaya_refusal_last_en_v1.pt"),
         (BENCH, DATA_FROZEN / "control_neutral_v1.jsonl", "control_neutral_v1.jsonl")]

assert hashlib.sha256((DATA_FROZEN / "control_neutral_v1.jsonl").read_bytes()).hexdigest() == CTRL_SHA, "control set changed!"
api = HfApi()
for local, remote in FOLDERS:
    n = sum(1 for p in local.iterdir() if p.is_file() and p.name not in IGNORE)
    print(f"{'UPLOAD' if GO else 'dry'}  {local.name:<14} -> {ART}/{remote}/   ({n} files)")
    if GO:   # one commit per folder; upload_folder skips files whose content is already identical on the Hub
        api.upload_folder(folder_path=str(local), path_in_repo=remote, repo_id=ART, repo_type="dataset",
                          ignore_patterns=IGNORE, commit_message=f"Ship S2: {remote}/")
for repo, p, remote in FILES:
    assert p.is_file(), p
    print(f"{'UPLOAD' if GO else 'dry'}  {p.name:<40} -> {repo}/{remote}")
    if GO:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=remote, repo_id=repo, repo_type="dataset",
                        commit_message=f"Ship S2: {remote}")
print("done" if GO else "dry run only: rerun with --go")
