# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers==5.22.0", "accelerate", "huggingface_hub", "pandas", "datasets",
#                 "sentencepiece", "protobuf", "tiktoken",
#                 "auditkit[transformers] @ git+https://github.com/Lexsi-Labs/AuditKIT@71316da"]
# ///
"""Phase 4 step 2.4 — WildGuard POSITIVE CONTROL / pipeline diagnostic (PREDICTIONS P4.4). HF Job, l4x1.
Why: in step 2.2 WildGuard said "harmful response: yes" on 0 of 334 rows. Can it fire at all through our pipeline?
Controls = WildGuard's OWN test set (allenai/wildguardmix, config wildguardtest; gated): 10 rows its authors labelled
harmful response + 10 labelled unharmful (English, non-adversarial, seed 0). No harmful text is written by us, printed, or re-uploaded.
Three scoring paths on the SAME weights (the model AuditKIT loads is reused for b/c -> no second 14.5 GB copy):
 (a) AuditKIT GuardJudge('wildguard') — exactly as in step 2.2
 (b) direct transformers, WildGuard model-card template, add_special_tokens=False  (official usage)
 (c) direct, add_special_tokens=True  (tests the double-<s> hypothesis: AuditKIT's template already starts with '<s>')
Plus (b) on OUR 111 harmful validation rows: if (b) fires where (a) never did, step 2.2's numbers came from a defective path.
Output: ART_REPO phase4/wildguard/wildguard_diag_v1.jsonl (ids, labels, parsed fields, WildGuard's short raw answer; no prompt/response text)."""
import json, time
import pandas as pd, torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi
from auditkit import GuardJudge, Sample
from auditkit.metrics.guard import _WILDGUARD_TEMPLATE, _parse_wildguard

ART_REPO = "anantshri1/lexsi-refusal-artefacts"
VAL_INP, OUT = "phase4/wildguard/wildguard_val_input_v1.jsonl", "phase4/wildguard/wildguard_diag_v1.jsonl"
N_EACH, SEED = 10, 0


def controls():
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test").to_pandas()
    print("wildguardmix columns:", list(ds.columns), "| rows:", len(ds), flush=True)
    ds = ds[(ds.adversarial == False) & ds.response.notna() & (ds.response.str.len() > 0)]
    ds["ctrl_id"] = ds.index.astype(str)
    pick = [ds[ds.response_harm_label == lab].sample(N_EACH, random_state=SEED).assign(ctrl=lab)
            for lab in ["harmful", "unharmful"]]
    return pd.concat(pick)


@torch.no_grad()
def direct(model, tok, prompt, response, special):
    text = _WILDGUARD_TEMPLATE.format(prompt=prompt, response=response)       # identical template text to (a)
    enc = tok([text], return_tensors="pt", add_special_tokens=special).to(model.device)
    n_bos = int((enc["input_ids"][0][:3] == tok.bos_token_id).sum())
    out = model.generate(**enc, max_new_tokens=32, do_sample=False)
    raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return raw, n_bos


def main():
    api, t0 = HfApi(), time.time()
    C = controls()
    ours = [json.loads(l) for l in open(hf_hub_download(ART_REPO, VAL_INP, repo_type="dataset"))]
    ours = [r for r in ours if r["polarity"] == "harmful"]
    g = GuardJudge("wildguard", judge_model="hf:allenai/wildguard", judge_model_args={"dtype": torch.float16})
    rows = []
    for _, r in C.iterrows():                                                  # (a) first: this loads the weights
        s = g.score(Sample(input=r.prompt, id=r.ctrl_id), r.response)
        rows.append(dict(src="wgm", id=r.ctrl_id, truth_harm=r.ctrl, truth_refusal=r.response_refusal_label, path="a",
                         **{k: s.metadata.get(k) for k in ("harmful_request", "response_refusal", "harmful_response")},
                         raw=s.metadata.get("raw"), n_bos=None))
    pipe = g._model._pipeline                                                  # reuse AuditKIT's loaded weights
    model, tok = pipe.model, pipe.tokenizer
    print("tokenizer bos:", tok.bos_token, "| adds BOS by default:", tok("x").input_ids[:1] == [tok.bos_token_id], flush=True)
    for path, special in [("b", False), ("c", True)]:
        for _, r in C.iterrows():
            raw, n_bos = direct(model, tok, r.prompt, r.response, special)
            _, f = _parse_wildguard(raw)
            rows.append(dict(src="wgm", id=r.ctrl_id, truth_harm=r.ctrl, truth_refusal=r.response_refusal_label, path=path,
                             **{k: f.get(k) for k in ("harmful_request", "response_refusal", "harmful_response")}, raw=raw[:120], n_bos=n_bos))
    for r in ours:                                                             # (b) on our harmful validation rows
        raw, n_bos = direct(model, tok, r["prompt"], r["response"], False)
        _, f = _parse_wildguard(raw)
        rows.append(dict(src="ours", id=r["key"], truth_harm="human_comp" if r["human_comp"] else "human_not_comp",
                         truth_refusal=r["human"], path="b", **{k: f.get(k) for k in ("harmful_request", "response_refusal", "harmful_response")},
                         raw=raw[:120], n_bos=n_bos))
    D = pd.DataFrame(rows)
    api.upload_file(path_or_fileobj=D.to_json(orient="records", lines=True, force_ascii=False).encode(), path_in_repo=OUT,
                    repo_id=ART_REPO, repo_type="dataset", commit_message="Phase 4 step 2.4: WildGuard diagnostic")
    w = D[D.src == "wgm"]
    print("\nwildguardmix controls: share 'harmful response: yes' by path x truth (n=10 each)")
    print(w.assign(yes=w.harmful_response == "yes").pivot_table(index="path", columns="truth_harm", values="yes", aggfunc="mean").round(2))
    print("BOS tokens at start by path:", w.groupby("path").n_bos.first().to_dict())
    print("(a) vs (c) identical raw answers:", (w[w.path == "a"].raw.values == w[w.path == "c"].raw.values).mean().round(2))
    o = D[D.src == "ours"]
    print("\nOUR harmful rows via (b): 'harmful response: yes' among human-compliant:",
          int(((o.truth_harm == "human_comp") & (o.harmful_response == "yes")).sum()), "/", int((o.truth_harm == "human_comp").sum()),
          "| response_refusal agreement with step 2.2 (a): to compare on the Mac")
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
