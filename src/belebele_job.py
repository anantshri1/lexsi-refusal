# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers==5.22.0", "accelerate", "huggingface_hub", "pandas", "datasets",
#                 "auditkit[transformers] @ git+https://github.com/Lexsi-Labs/AuditKIT@71316da",
#                 "safetune @ git+https://github.com/Lexsi-Labs/SafeTune@26f59e6"]
# ///
"""Phase 4 block 4.3 — UTILITY axis: Belebele reading comprehension (facebook/belebele; eng_Latn, ita_Latn, deu_Latn, hin_Deva;
900 questions each) via AuditKIT's NATIVE engine (MCQAdapter mcq_loglikelihood: one teacher-forced forward per choice; trap T3:
run_lmeval reloads weights and would lose hooks). Steering = SafeTune RefusalDirectionModel installed on the model object inside
AuditKIT's HFGenModel, same op as the full runs. Prompt = lm-eval's Belebele format (P:/Q:/A-D, targets ' A'..' D'), zero-shot, raw text.
Steering vector for utility = normalised mean of the 5 SafeTune fold vectors at the slot (Belebele has no pairs to cross-fit;
fold vectors agree at cos >= ~0.99). HF Job t4-small. Output: ART_REPO phase4/belebele/belebele_{cond}_v1.json."""
import argparse, json, time
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi
from auditkit import Sample, evaluate
from auditkit.adapter import MCQAdapter
from auditkit.model import AutoModel
from safetune.interventions.steer.refusal_direction import RefusalDirectionModel

MODEL_ID, ART_REPO = "CohereLabs/tiny-aya-global", "anantshri1/lexsi-refusal-artefacts"
LANGS = {"en": "eng_Latn", "it": "ita_Latn", "de": "deu_Latn", "hi": "hin_Deva"}
SLOTS = [18, 21, 24]
DIRS, STEER, OUT = "phase4/screen/safetune_dirs_v1.pt", "directions/dir_tinyaya_steer_v1.pt", "phase4/belebele/"
# id: (direction, slot, alpha/Delta) — base = no hook. F1/F2 = the full-run setting.
CONDS = {"base": None, "steer-ren-s24-a020": ("ren", 24, 0.2), "steer-rhi-s24-a020": ("rhi", 24, 0.2)}


def st_idx(slot):
    return slot - 1


def samples(lang, limit):
    ds = load_dataset("facebook/belebele", LANGS[lang], split="test")
    out = []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        p = (f"P: {r['flores_passage']}\nQ: {r['question']}\nA: {r['mc_answer1']}\nB: {r['mc_answer2']}\n"
             f"C: {r['mc_answer3']}\nD: {r['mc_answer4']}\nAnswer:")
        out.append(Sample(input=p, choices=[" A", " B", " C", " D"], target=int(r["correct_answer_num"]) - 1,
                          id=f"{lang}-{i}", task=f"belebele_{lang}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)          # smoke: N questions per language
    a = ap.parse_args()
    api = HfApi()
    D = torch.load(hf_hub_download(ART_REPO, DIRS, repo_type="dataset"), weights_only=False)
    V = torch.load(hf_hub_download(ART_REPO, STEER, repo_type="dataset"), weights_only=False)
    m = AutoModel.resolve(f"hf:{MODEL_ID}", dtype=torch.float16)
    m._ensure_pipeline()                                       # load once; hooks go on m._pipeline.model
    data = {L: samples(L, a.limit) for L in LANGS}
    tag = "smoke" if a.limit else "v1"
    for cid, spec in CONDS.items():
        name = f"belebele_{cid}_{tag}.json"
        if api.file_exists(ART_REPO, OUT + name, repo_type="dataset"):
            print("skip (exists):", name, flush=True); continue
        hook = None
        if spec:
            direction, slot, frac = spec
            j = SLOTS.index(slot)
            d = D["en_folds" if direction == "ren" else "hi_folds"][:, j].mean(0)
            d = d / d.norm()
            alpha = frac * V["delta_en"][j].item()
            hook = RefusalDirectionModel(m._pipeline.model, d, mode="steer", strength=alpha, layers=[st_idx(slot)]).install()
        res, t0 = {}, time.time()
        try:
            for L, ss in data.items():
                r = evaluate(ss, m, scorers="exact_match", adapter=MCQAdapter("mcq_loglikelihood"))
                own = [max(range(4), key=lambda i: p.choice_likelihoods[i]) == s.target for p, s in
                       zip(sorted(r.predictions, key=lambda p: int(p.sample_id.split("-")[1])), ss)]
                res[L] = dict(n=len(ss), auditkit_headline=r.headline, acc_recomputed=round(sum(own) / len(own), 4),
                              correct=[bool(x) for x in own])
                print(f"{cid} {L}: AuditKIT {r.headline} | recomputed acc {res[L]['acc_recomputed']}", flush=True)
        finally:
            if hook:
                hook.remove()
        api.upload_file(path_or_fileobj=json.dumps(dict(cond=cid, spec=spec, secs=round(time.time() - t0), langs=res)).encode(),
                        path_in_repo=OUT + name, repo_id=ART_REPO, repo_type="dataset", commit_message=f"Belebele {cid}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
