# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers==5.22.0", "accelerate", "huggingface_hub", "pandas", "sentencepiece", "protobuf", "tiktoken",
#                 "auditkit[transformers] @ git+https://github.com/Lexsi-Labs/AuditKIT@71316da"]
# ///
"""Phase 4 step 2.2 — AuditKIT GuardJudge('wildguard') on every Anant-labelled row (input built by src/wildguard_val.py prep).
HF Job on l4x1 (WildGuard 7B fp16 ~14.5 GB: no headroom on a T4; no 4-bit, so the judge we validate is the judge we would use).
Output: ART_REPO phase4/wildguard/wildguard_val_output_v1.jsonl (key, WildGuard fields, raw text). Scored on the Mac."""
import io, json, time
import torch
from huggingface_hub import hf_hub_download, HfApi
from auditkit import GuardJudge, Sample

ART_REPO = "anantshri1/lexsi-refusal-artefacts"
INP, OUT = "phase4/wildguard/wildguard_val_input_v1.jsonl", "phase4/wildguard/wildguard_val_output_v1.jsonl"


def push(rows, api):
    api.upload_file(path_or_fileobj="\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode(),
                    path_in_repo=OUT, repo_id=ART_REPO, repo_type="dataset", commit_message=f"WildGuard val: {len(rows)} rows")


def main():
    api = HfApi()
    rows = [json.loads(l) for l in open(hf_hub_download(ART_REPO, INP, repo_type="dataset"))]
    g = GuardJudge("wildguard", judge_model="hf:allenai/wildguard", judge_model_args={"dtype": torch.float16})
    print("GPU:", torch.cuda.get_device_name(0), "| rows:", len(rows), "| judge:", g.name, flush=True)
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        s = g.score(Sample(input=r["prompt"], id=r["key"]), r["response"])
        m = s.metadata
        out.append(dict(key=r["key"], wg_harmful_request=m.get("harmful_request"), wg_response_refusal=m.get("response_refusal"),
                        wg_harmful_response=m.get("harmful_response"), wg_value=s.value, wg_raw=m.get("raw")))
        if i < 3:
            print("example:", out[-1], flush=True)
        if (i + 1) % 100 == 0:
            push(out, api); print(f"{i+1} rows, {time.time()-t0:.0f}s", flush=True)
    push(out, api)
    unparsed = sum(o["wg_response_refusal"] is None for o in out)
    print(f"DONE {len(out)} rows in {time.time()-t0:.0f}s | response_refusal unparsed: {unparsed}", flush=True)


if __name__ == "__main__":
    main()
