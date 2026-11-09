# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers==5.22.0", "accelerate", "huggingface_hub", "pandas"]
# ///
"""Phase 3 — residual stream at 2 positions x 37 slots (embeddings + 36 layers, PRE final norm), all benchmark rows.
Runs as an HF Job (single file, no repo). Constants duplicated from paths.py/generate.py -> checked on Mac by 3c."""
import argparse, hashlib, io, time
import torch, pandas as pd, transformers
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID   = "CohereLabs/tiny-aya-global"
BENCH_REPO = "anantshri1/lexsi-refusal-bench"
ART_REPO   = "anantshri1/lexsi-refusal-artefacts"
BENCH_SHA  = "9d727751d3cbbad74ad8e179e4e2b505f83681623febe9e2d424959b21a3871d"
POS = {"last": -1, "user": -5}     # START_RESPONSE | last user-content token (gap 4 on all 1770 rows, check_positions.py)
N_SLOTS = 37                       # 0 = embeddings; k = output of decoder layer k-1 (pre final norm)


def load_bench():
    p = hf_hub_download(BENCH_REPO, "benchmark_v1.jsonl", repo_type="dataset")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == BENCH_SHA, "wrong benchmark"
    return pd.read_json(p, lines=True)


def build_messages(text):          # must equal generate.build_messages(text, "base")
    return [{"role": "user", "content": text}]


class Grab:
    """Forward hooks keep ONLY the 2 positions we need -> GPU memory stays flat whatever the length."""
    def __init__(self, model):
        self.buf, base = {}, model.model
        self.handles = [base.embed_tokens.register_forward_hook(self._hook(0))]
        self.handles += [l.register_forward_hook(self._hook(k + 1)) for k, l in enumerate(base.layers)]
        assert len(self.handles) == N_SLOTS

    def _hook(self, slot):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out                    # [bs, seq, 2048]
            self.buf[slot] = torch.stack([h[:, POS["last"]], h[:, POS["user"]]], 1).to("cpu")  # [bs, 2, 2048] fp16
        return f


@torch.no_grad()
def run_batch(model, tok, texts, grab, check_hs=False):
    enc = tok.apply_chat_template([build_messages(t) for t in texts], add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt", padding=True).to(model.device)
    grab.buf.clear()
    out = model.model(**enc, use_cache=False, output_hidden_states=check_hs)  # base model: skips the big-vocab lm_head
    acts = torch.stack([grab.buf[s] for s in range(N_SLOTS)], 2)           # [bs, 2(pos), 37, 2048]
    if check_hs:   # smoke only: our hook slots vs HF hidden_states at the last position
        for s, h in enumerate(out.hidden_states):
            d = (h[:, -1].cpu().float() - acts[:, 0, s].float()).abs().max().item()
            print(f"  slot {s:2d}: max|hook - hidden_states| = {d:.3g}")
        print("  n hidden_states:", len(out.hidden_states))
    return acts, enc["attention_mask"].sum(1).tolist()

SHARDS = [(l, "orig") for l in ["en", "it", "de", "hi", "ml", "hi-Latn"]] + [(l, "para") for l in ["en", "hi", "ml"]]

def shard_file(lang, var, tag):
    return f"act_tinyaya_base_{tag}_{lang}_{var}.pt"


def batches(lens, budget):
    """Longest first (an OOM shows up immediately); grow a batch while bs * max_len <= budget tokens."""
    order, b = sorted(range(len(lens)), key=lambda i: -lens[i]), []
    for i in order:
        if b and (len(b) + 1) * lens[b[0]] > budget:
            yield b; b = []
        b.append(i)
    if b:
        yield b


def extract_shard(model, tok, rows, grab, budget, check_hs=False):
    lens = [len(tok.apply_chat_template(build_messages(t), add_generation_prompt=True, tokenize=True,
                                        return_dict=True)["input_ids"]) for t in rows.text]
    acts = torch.empty(len(rows), 2, N_SLOTS, 2048, dtype=torch.float16)
    seq = [0] * len(rows)
    for j, b in enumerate(batches(lens, budget)):
        a, L = run_batch(model, tok, rows.text.iloc[b].tolist(), grab, check_hs=check_hs and j == 0)
        acts[b] = a
        for i, n in zip(b, L):
            seq[i] = n
    assert seq == lens, "attention-mask length != tokenised length (padding problem)"
    assert torch.isfinite(acts).all(), "non-finite activations (fp16 overflow?)"
    return acts, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="all")          # e.g. "en_orig,ml_orig"
    ap.add_argument("--limit", type=int, default=None)  # smoke test: first N rows per shard
    ap.add_argument("--budget", type=int, default=16384)
    a = ap.parse_args()
    tag = "v1" if a.limit is None else "smoke"
    todo = SHARDS if a.shards == "all" else [tuple(s.rsplit("_", 1)) for s in a.shards.split(",")]

    api, bench = HfApi(), load_bench()
    tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda").eval()
    grab = Grab(model)
    meta = dict(model=MODEL_ID, model_sha=api.model_info(MODEL_ID).sha, transformers=transformers.__version__,
                torch=torch.__version__, bench_sha=BENCH_SHA, template="base", positions=POS,
                slots="0=embeddings; k=decoder layer k-1 output; pre final norm", dtype="float16")
    print(meta, flush=True)

    for lang, var in todo:
        name = shard_file(lang, var, tag)
        if api.file_exists(ART_REPO, f"activations/{name}", repo_type="dataset"):
            print("skip (exists):", name); continue
        rows = bench[(bench.lang == lang) & bench.uid.str.endswith("-" + var)].reset_index(drop=True)
        if a.limit:
            rows = rows.head(a.limit)
            else:
                assert len(rows) == (266 if var == "orig" else 58), (name, len(rows))   # 133 pairs x 2; para 29 x 2
        t0 = time.time()
        acts, lens = extract_shard(model, tok, rows, grab, a.budget, check_hs=a.limit is not None)
        buf = io.BytesIO()
        torch.save(dict(uids=rows.uid.tolist(), acts=acts, seq_len=lens, meta=meta), buf)
        api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=f"activations/{name}", repo_id=ART_REPO,
                        repo_type="dataset", commit_message=f"Phase 3 activations: {name}")
        print(f"{name}: {tuple(acts.shape)} in {time.time()-t0:.0f}s, max_len {max(lens)}", flush=True)


if __name__ == "__main__":
    main()