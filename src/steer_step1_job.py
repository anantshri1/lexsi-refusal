# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers==5.22.0", "accelerate", "huggingface_hub", "pandas",
#                 "safetune @ git+https://github.com/Lexsi-Labs/SafeTune@26f59e6"]
# ///
"""Phase 4 Step 1 — SafeTune cross-check + steering smoke tests. HF Job (single file, no repo; constants checked on the Mac
by src/test_step1.py). Needs ART_REPO directions/dir_tinyaya_steer_v1.pt (src/steer_prep.py). Outputs -> ART_REPO phase4/step1/.
 A  SafeTune extract_refusal_direction (pool last_token, normalize=False, pick_layer = SLOT 24 -> idx 23) on en H_core
    pair_keep harmful vs benign twins, target layers = our 3 slots; cosine vs our r_en full printed here, saved for the Mac.
 B  SafeTune's template path (string, then tokenise without special tokens) == our token ids.
 C  Hook mechanics: steer at idx 23 with strength a -> slot-24 START_RESPONSE state changes by EXACTLY a * d (slot 23 untouched).
 D  strength 0 with hooks == no hooks (greedy ids identical); vs Phase 2 base generations if that file is on the repo (info only).
 E  Timing: 16 longest en/it/de/hi prompts, 512 new tokens, steered (r_en slot 24, alpha/Delta 0.25). Saved, NOT judged."""
import hashlib, io, json, time
import torch, pandas as pd, transformers
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
try:                           # absent on the Mac, where src/test_step1.py only checks the constants
    from safetune.interventions.steer.refusal_direction import (RefusalDirectionConfig, RefusalDirectionModel,
                                                                 extract_refusal_direction)
except ImportError:
    RefusalDirectionConfig = RefusalDirectionModel = extract_refusal_direction = None

MODEL_ID   = "CohereLabs/tiny-aya-global"
BENCH_REPO = "anantshri1/lexsi-refusal-bench"
ART_REPO   = "anantshri1/lexsi-refusal-artefacts"
BENCH_SHA  = "9d727751d3cbbad74ad8e179e4e2b505f83681623febe9e2d424959b21a3871d"
STEER_FILE = "directions/dir_tinyaya_steer_v1.pt"
OUT        = "phase4/step1/"
SLOTS      = [18, 21, 24]
EN_DROP    = {"mj110"}        # en H_core pairs that are NOT pair_keep (r_en was trained without them); checked on the Mac
GEN_P2     = "generations/gen_tinyaya_base_v1.jsonl"


def st_idx(slot):              # SLOT k = output of decoder layer k-1 -> SafeTune layer index (PATHS.md, trap T1)
    return slot - 1


class NoHead(torch.nn.Module):
    """The decoder WITHOUT lm_head, for SafeTune's extractor. Its full-CausalLM forward computes logits at every position
    (16 x ~550 tok x 262k vocab fp16 = 4.3 GB -> T4 OOM, first step 1 run 2026-11-10). Same layer modules: SafeTune finds
    them via model.model.layers and hooks exactly what it would hook on the full model; only the unused logits are skipped."""
    def __init__(self, base):
        super().__init__()
        self.model = base

    def forward(self, **kw):
        return self.model(**kw, use_cache=False)


def load_bench():
    p = hf_hub_download(BENCH_REPO, "benchmark_v1.jsonl", repo_type="dataset")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == BENCH_SHA, "wrong benchmark"
    return pd.read_json(p, lines=True)


def build_messages(text):      # must equal generate.build_messages(text, "base")
    return [{"role": "user", "content": text}]


def upload(obj, name, api):
    buf = io.BytesIO()
    torch.save(obj, buf) if not isinstance(obj, (str, bytes)) else buf.write(obj.encode() if isinstance(obj, str) else obj)
    api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=OUT + name, repo_id=ART_REPO, repo_type="dataset",
                    commit_message=f"Phase 4 step 1: {name}")


def encode(tok, texts):
    return tok.apply_chat_template([build_messages(t) for t in texts], add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", padding=True).to("cuda")


@torch.no_grad()
def last_states(model, tok, texts, slots, steer=None):
    """START_RESPONSE state at each slot; steer = (direction, strength, slot) installed via SafeTune BEFORE our grab hooks,
    so our hooks see the steered output (PyTorch runs forward hooks in registration order, each on the previous result)."""
    st = RefusalDirectionModel(model, steer[0], mode="steer", strength=steer[1], layers=[st_idx(steer[2])]).install() if steer else None
    buf, hs = {}, []
    for s in slots:
        def f(mod, inp, out, s=s):
            buf[s] = (out[0] if isinstance(out, tuple) else out)[:, -1].float().cpu()
        hs.append(model.model.layers[st_idx(s)].register_forward_hook(f))
    try:
        model.model(**encode(tok, texts), use_cache=False)
    finally:
        for h in hs: h.remove()
        if st: st.remove()
    return buf


@torch.no_grad()
def greedy(model, tok, texts, max_new, steer=None):
    st = RefusalDirectionModel(model, steer[0], mode="steer", strength=steer[1], layers=[st_idx(steer[2])]).install() if steer else None
    try:
        enc = encode(tok, texts)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    finally:
        if st: st.remove()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    out = []
    for row in gen[:, enc["input_ids"].shape[1]:].tolist():
        cut = next((i for i, t in enumerate(row) if t in eos), None)
        out.append(row if cut is None else row[:cut])
    return out


def main():
    api, bench = HfApi(), load_bench()
    V = torch.load(hf_hub_download(ART_REPO, STEER_FILE, repo_type="dataset"), weights_only=False)
    assert V["slots"] == SLOTS and V["st_idx"] == [st_idx(k) for k in SLOTS], "steer file slot convention mismatch"
    tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda").eval()
    log = dict(model_sha=api.model_info(MODEL_ID).sha, transformers=transformers.__version__, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0))
    print(log, flush=True)
    orig = bench[bench.variant == "original"]

    # ---- A: SafeTune direction on exactly r_en's training rows ----
    en = orig[(orig.lang == "en") & (orig.tier == "H_core") & ~orig.pair_id.isin(EN_DROP)].sort_values("pair_id")
    harm, ben = en[en.polarity == "harmful"].text.tolist(), en[en.polarity == "benign"].text.tolist()
    cfg = RefusalDirectionConfig(target_layers=[st_idx(k) for k in SLOTS], pick_layer=st_idx(24),
                                 pool_method="last_token", normalize=False)
    t0 = time.time()
    d, layer, per = extract_refusal_direction(NoHead(model.model), tok, harm, ben, cfg)
    assert layer == st_idx(24), f"SafeTune picked {layer}, expected {st_idx(24)}"
    upload(dict(direction=d.cpu(), layer=layer, per_layer={k: v.cpu() for k, v in per.items()}, n_harm=len(harm),
                n_ben=len(ben), cfg=vars(cfg), note="per_layer keys = SafeTune idx = slot - 1"), "safetune_dir_en_v1.pt", api)
    cosA = {k: round(torch.nn.functional.cosine_similarity(per[st_idx(k)].float().cpu(), V["en_full"][j], dim=0).item(), 5)
            for j, k in enumerate(SLOTS)}
    log["A"] = dict(n=(len(harm), len(ben)), secs=round(time.time() - t0), cos_safetune_vs_r_en_by_slot=cosA,
                    raw_norm_slot24=round(per[st_idx(24)].norm().item(), 3), returned_norm=round(d.norm().item(), 3))
    print("A", log["A"], flush=True)

    # ---- B: template path ----
    smp = orig.groupby("lang").head(1).text.tolist()
    same = [tok(tok.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True),
                add_special_tokens=False).input_ids
            == list(tok.apply_chat_template(build_messages(t), add_generation_prompt=True, return_dict=True)["input_ids"])
            for t in smp]
    log["B"] = dict(template_ids_equal=same)
    print("B", log["B"], flush=True)

    # ---- C: hook mechanics at slot 24 ----
    hi8 = orig[(orig.lang == "hi") & (orig.tier == "H_core") & (orig.polarity == "harmful")].head(8).text.tolist()
    r24, a = V["en_full"][2], 0.25 * V["delta_en"][2].item()
    h0 = last_states(model, tok, hi8, [23, 24, 25])
    h1 = last_states(model, tok, hi8, [23, 24, 25], steer=(r24, a, 24))
    dd = {s: h1[s] - h0[s] for s in (23, 24, 25)}
    log["C"] = dict(alpha=round(a, 3),
                    slot23_max_abs_change=dd[23].abs().max().item(),
                    slot24_proj_change=[round(x, 3) for x in (dd[24] @ r24).tolist()],
                    slot24_resid_off_direction=round((dd[24] - (dd[24] @ r24)[:, None] * r24).norm(dim=-1).max().item(), 4),
                    slot25_proj_change=[round(x, 3) for x in (dd[25] @ r24).tolist()])
    print("C", log["C"], flush=True)

    # ---- D: strength 0 == no hooks; vs Phase 2 (info) ----
    ids0 = greedy(model, tok, hi8, 64)
    ids1 = greedy(model, tok, hi8, 64, steer=(r24, 0.0, 24))
    log["D"] = dict(strength0_identical=[x == y for x, y in zip(ids0, ids1)])
    try:
        p2 = pd.read_json(hf_hub_download(ART_REPO, GEN_P2, repo_type="dataset"), lines=True).set_index("uid")
        u8 = orig[(orig.lang == "hi") & (orig.tier == "H_core") & (orig.polarity == "harmful")].head(8).uid
        log["D"]["first64_equal_phase2"] = [list(p2.loc[u, "ids"][:64]) == x[:64] for u, x in zip(u8, ids0)]
    except Exception as e:
        log["D"]["first64_equal_phase2"] = f"skipped: {type(e).__name__}"
    print("D", log["D"], flush=True)

    # ---- E: timing, worst case (16 longest prompts), steered as in screening ----
    pool = orig[orig.lang.isin(["en", "it", "de", "hi"])].copy()
    pool["n_tok"] = [len(tok.apply_chat_template(build_messages(t), add_generation_prompt=True, return_dict=True)["input_ids"])
                     for t in pool.text]
    longest = pool.sort_values("n_tok", ascending=False).head(16)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    ids = greedy(model, tok, longest.text.tolist(), 512, steer=(r24, a, 24))
    log["E"] = dict(secs_16x512=round(time.time() - t0), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1),
                    max_prompt_tok=int(longest.n_tok.max()), n_new=[len(x) for x in ids])
    print("E", log["E"], flush=True)
    resp = [dict(uid=u, steer="r_en slot24 alpha/Delta0.25", alpha=a, n_new_tokens=len(x),
                 response=tok.decode(x, skip_special_tokens=True).strip()) for u, x in zip(longest.uid, ids)]
    upload("\n".join(json.dumps(r, ensure_ascii=False) for r in resp), "timing_responses_v1.jsonl", api)
    upload(json.dumps(log, indent=1), "step1_log_v1.json", api)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
