"""Push frozen benchmark + card to HF (private). Verifies checksum before upload."""
import hashlib
from huggingface_hub import HfApi
from paths import DATA_FROZEN as F, HF_BENCH_REPO

SHA = "9d727751d3cbbad74ad8e179e4e2b505f83681623febe9e2d424959b21a3871d"
data = F / "benchmark_v1.jsonl"
assert hashlib.sha256(data.read_bytes()).hexdigest() == SHA, "frozen file changed!"

api = HfApi()
api.create_repo(HF_BENCH_REPO, repo_type="dataset", private=True, exist_ok=True)
for p in [data, F / "README.md"]:
    api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=HF_BENCH_REPO,
                    repo_type="dataset", commit_message=f"benchmark v1: {p.name}")
print("pushed to", HF_BENCH_REPO)
