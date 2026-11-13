"""Ship block S1 (read-only): what is on the HF artefact repo vs what we hold locally.
Prints (1) remote file counts per top folder, (2) local files NOT on the remote, (3) same-name files whose size differs."""
from collections import Counter
from huggingface_hub import HfApi
from paths import (HF_ARTEFACT_REPO, HF_BENCH_REPO, JUDGEMENTS, COMPREHENSION, GENERATIONS,
                   DIRECTIONS, PHASE4, DATA_FROZEN)

api = HfApi()
remote = {f.path: f.size for f in api.list_repo_tree(HF_ARTEFACT_REPO, repo_type="dataset", recursive=True)
          if getattr(f, "size", None) is not None}           # files only (folders have no size)
print("REMOTE", HF_ARTEFACT_REPO, len(remote), "files")
for top, n in sorted(Counter(p.split("/")[0] for p in remote).items()):
    print(f"  {top:<16}{n}")
print("BENCH ", HF_BENCH_REPO, sorted(f.path for f in api.list_repo_tree(HF_BENCH_REPO, repo_type="dataset")))

# local folder -> its folder on the remote (mirror convention from PATHS.md)
LOCAL = {JUDGEMENTS: "judgements/", COMPREHENSION: "comprehension/", GENERATIONS: "generations/",
         DIRECTIONS: "directions/", PHASE4: "phase4/"}
missing, differ = [], []
for root, prefix in LOCAL.items():
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        rp = prefix + p.relative_to(root).as_posix()
        if rp not in remote:
            missing.append((rp, p.stat().st_size))
        elif remote[rp] != p.stat().st_size:
            differ.append((rp, p.stat().st_size, remote[rp]))
print(f"\nLOCAL NOT ON REMOTE ({len(missing)}):")
for rp, s in missing:
    print(f"  {s/1e6:7.2f} MB  {rp}")
print(f"\nSIZE DIFFERS local vs remote ({len(differ)}):")
for rp, a, b in differ:
    print(f"  {rp}  local {a}  remote {b}")
print("\nfrozen dir:", sorted(p.name for p in DATA_FROZEN.iterdir()))
