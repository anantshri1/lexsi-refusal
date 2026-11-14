"""Ship (T2b): the steering 'adapter' as a model repo. No weights: Tiny Aya Global is unchanged; the repo holds the slot-24
refusal steering vectors + a SafeTune snippet. Default = build locally (steer_model/); --push uploads (public)."""
import sys, shutil
import torch
from huggingface_hub import HfApi
from paths import DIRECTIONS, REPO_ROOT, HF_USER

REPO = f"{HF_USER}/tiny-aya-global-refusal-steer"
OUT = REPO_ROOT / "steer_model"
src = torch.load(DIRECTIONS / "dir_tinyaya_steer_v1.pt", map_location="cpu", weights_only=False)   # our own file (fold_map holds numpy scalars)
k = list(src["slots"]).index(24)                      # slot 24 = output of decoder layer 23 = SafeTune layers=[23]
fold_cos = torch.nn.functional.cosine_similarity(src["en_folds"][:, k], src["en_full"][k][None], dim=-1)
vec = dict(slot=24, safetune_layer=23, delta_en=float(src["delta_en"][k]), alpha_over_delta=0.2,
           alpha=round(0.2 * float(src["delta_en"][k]), 2),
           r_en=src["en_full"][k].clone(), r_hi=src["hi_full"][k].clone(),          # unit vectors, all pairs
           r_en_folds=src["en_folds"][:, k].clone(), r_hi_folds=src["hi_folds"][:, k].clone(),
           fold_map={str(p): int(f) for p, f in dict(src["fold_map"]).items()})   # plain types -> loads with weights_only=True
for n in ("r_en", "r_hi"):
    assert abs(vec[n].norm().item() - 1) < 1e-4, n
print(f"alpha {vec['alpha']} | delta_en {vec['delta_en']:.2f} | cos(fold, full) r_en min {fold_cos.min():.4f}")

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
torch.save(vec, OUT / "steer_tinyaya_s24_v1.pt")
card = (REPO_ROOT / "cards" / "steer_model_README.md").read_text().replace("{FOLD_COS_MIN}", f"{fold_cos.min():.4f}")
(OUT / "README.md").write_text(card)
print("built", OUT)
if "--push" in sys.argv:
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=REPO, repo_type="model", commit_message="Ship: refusal steering vectors + card")
    print("pushed: https://huggingface.co/" + REPO)
