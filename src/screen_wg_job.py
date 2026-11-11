# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers==5.22.0", "accelerate", "huggingface_hub", "pandas", "sentencepiece", "protobuf", "tiktoken",
#                 "auditkit[transformers] @ git+https://github.com/Lexsi-Labs/AuditKIT@71316da"]
# ///
"""Phase 4 block 3.4b — SCREENING over-refusal (D11): AuditKIT GuardJudge('wildguard') — the path validated in step 2 — on the
165 BENIGN dev rows (all H_core twins) of base + the 14 screening conditions. HF Job, l4x1 (WildGuard fp16). One output file per
condition, uploaded when done; finished conditions are skipped on a rerun.
Output: ART_REPO phase4/screen/wg_{cond}_dev_v1.jsonl (uid, WildGuard fields, raw)."""
import hashlib, json, time
import pandas as pd, torch
from huggingface_hub import hf_hub_download, HfApi
from auditkit import GuardJudge, Sample

BENCH_REPO, ART_REPO = "anantshri1/lexsi-refusal-bench", "anantshri1/lexsi-refusal-artefacts"
BENCH_SHA = "9d727751d3cbbad74ad8e179e4e2b505f83681623febe9e2d424959b21a3871d"
OUT = "phase4/screen/"
SLOTS = [18, 21, 24]
CONDS = ["base", "steer-ren-s24-a025", "steer-ctrl-s24-a025", "steer-rand-s24-a025", "ablate-ren-s24-all", "steer-rhi-s24-a025"] + \
        [f"steer-ren-s{k}-a{round(100 * f):03d}" for k in SLOTS for f in (0.1, 0.2, 0.3)]


def main():
    api = HfApi()
    p = hf_hub_download(BENCH_REPO, "benchmark_v1.jsonl", repo_type="dataset")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == BENCH_SHA
    text = pd.read_json(p, lines=True).set_index("uid").text
    split = json.load(open(hf_hub_download(ART_REPO, OUT + "split_v1.json", repo_type="dataset")))
    benign = [u for u in split["dev_uids"] if "-b-" in u]
    g = GuardJudge("wildguard", judge_model="hf:allenai/wildguard", judge_model_args={"dtype": torch.float16})
    for cid in CONDS:
        name = f"wg_{cid}_dev_v1.jsonl"
        if api.file_exists(ART_REPO, OUT + name, repo_type="dataset"):
            print("skip (exists):", name, flush=True); continue
        gen = pd.read_json(hf_hub_download(ART_REPO, OUT + f"gen_tinyaya_{cid}_dev_v1.jsonl", repo_type="dataset"), lines=True)
        gen = gen.set_index("uid").loc[benign]
        out, t0 = [], time.time()
        for uid, r in gen.iterrows():
            s = g.score(Sample(input=text[uid], id=uid), r.response)
            m = s.metadata
            out.append(dict(uid=uid, condition=cid, wg_harmful_request=m.get("harmful_request"),
                            wg_response_refusal=m.get("response_refusal"), wg_harmful_response=m.get("harmful_response"),
                            wg_raw=m.get("raw")))
        api.upload_file(path_or_fileobj="\n".join(json.dumps(o, ensure_ascii=False) for o in out).encode(), path_in_repo=OUT + name,
                        repo_id=ART_REPO, repo_type="dataset", commit_message=f"Phase 4 screen WildGuard: {cid}")
        ref = sum(o["wg_response_refusal"] == "yes" for o in out); bad = sum(o["wg_response_refusal"] is None for o in out)
        print(f"{cid}: {len(out)} benign rows in {time.time()-t0:.0f}s | WildGuard refusal {ref} | unparsed {bad}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
