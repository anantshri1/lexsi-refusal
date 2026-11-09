import sys; sys.path.insert(0, "src")
import torch
from huggingface_hub import snapshot_download
from paths import ACTIVATIONS, HF_ARTEFACT_REPO
from generate import load_bench

snapshot_download(HF_ARTEFACT_REPO, repo_type="dataset", local_dir=ACTIVATIONS.parent,
                  allow_patterns="activations/act_tinyaya_base_v1_*.pt")      # -> outputs/activations/, ~560 MB
bench = load_bench().set_index("uid")
for f in sorted(ACTIVATIONS.glob("act_tinyaya_base_v1_*.pt")):
    d = torch.load(f, weights_only=False)
    langs = set(bench.loc[d["uids"], "lang"])
    print(f.name, tuple(d["acts"].shape), langs, "finite" if torch.isfinite(d["acts"]).all() else "NON-FINITE",
          "| norm slot 18 (mean):", f"{d['acts'][:, 0, 18].float().norm(dim=-1).mean():.1f}")