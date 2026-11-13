# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers==5.22.0", "accelerate", "huggingface_hub", "pandas",
#                 "safetune @ git+https://github.com/Lexsi-Labs/SafeTune@26f59e6"]
# ///
"""Phase 4 step 3 — SCREENING (PHASE4_PLAN.md). HF Job, t4-small, single file (constants checked on the Mac by src/test_screen.py).
  --stage extract   (block 3.2a) SafeTune extracts EVERY steering vector we use: cross-fitted folds of r_en and r_hi
                    (fold f = H_core train pairs of that language with fold != f) at slots 18/21/24, and the en food-travel
                    control. Each is checked against our dir_tinyaya_steer_v1.pt (cos >= 0.9999) and saved to
                    ART_REPO phase4/screen/safetune_dirs_v1.pt. Our file is the cross-check; SafeTune's vectors are what we steer with.
  --stage screen    (block 3.2b) condition loop on the 330 dev rows (CONDS below; kill-test first). Each prompt in fold f is
                    steered with the fold-f vector (never saw its pair). Greedy, 256 new tokens. One file per condition,
                    uploaded as soon as it is done; finished conditions are skipped on a rerun. --limit N = smoke (tag smoke).
  --stage full      (block 4.1b) F1/F2 at the SELECTED setting (PHASE4_PLAN section 6 -> steer-ren-s24-a020) on all 1032 full rows
                    (phase4/full/full_uids_v1.json), 512 new tokens (= baselines), same per-fold SafeTune vectors. -> phase4/full/.
Inputs: ART_REPO phase4/screen/split_v1.json (src/screen_split.py), directions/dir_tinyaya_steer_v1.pt, control/control_neutral_v1.jsonl."""
import argparse, hashlib, io, json, time
import torch, pandas as pd, transformers
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
try:                           # absent on the Mac, where src/test_screen.py only checks the constants
    from safetune.interventions.steer.refusal_direction import (RefusalDirectionConfig, RefusalDirectionModel,
                                                                 extract_refusal_direction)
except ImportError:
    RefusalDirectionConfig = RefusalDirectionModel = extract_refusal_direction = None

MODEL_ID   = "CohereLabs/tiny-aya-global"
BENCH_REPO = "anantshri1/lexsi-refusal-bench"
ART_REPO   = "anantshri1/lexsi-refusal-artefacts"
BENCH_SHA  = "9d727751d3cbbad74ad8e179e4e2b505f83681623febe9e2d424959b21a3871d"
CTRL_SHA   = "441cd2d412ac1d1a88977c398cf2f3a4b4c89e7a8f91451fbafe6f510b83ba15"
STEER_FILE = "directions/dir_tinyaya_steer_v1.pt"
CTRL_FILE  = "control/control_neutral_v1.jsonl"
OUT        = "phase4/screen/"
SPLIT_FILE = OUT + "split_v1.json"
DIRS_FILE  = OUT + "safetune_dirs_v1.pt"
SLOTS      = [18, 21, 24]
K_FOLD     = 5
COS_MIN    = 0.9999
MAX_NEW, BS = 256, 16
FULL_NEW, FULL_DIR, FULL_UIDS = 512, "phase4/full/", "phase4/full/full_uids_v1.json"
FULL_CONDS = [("steer-ren-s24-a020", "ren", 24, 0.2, "steer"), ("steer-rhi-s24-a020", "rhi", 24, 0.2, "steer")]   # F1, F2
HI_CONDS = [("steer-ren-s24-a030", "ren", 24, 0.3, "steer")]   # P4.9: hi only, pushed harder (tag fullhi)
# id, direction, slot, alpha/Delta, mode. Order = priority (kill-test + sanity first, then the grid).
CONDS = [("steer-ren-s24-a025", "ren", 24, 0.25, "steer"), ("steer-ctrl-s24-a025", "ctrl", 24, 0.25, "steer"),
         ("steer-rand-s24-a025", "rand", 24, 0.25, "steer"), ("ablate-ren-s24-all", "ren", 24, None, "ablate"),
         ("steer-rhi-s24-a025", "rhi", 24, 0.25, "steer")] + \
        [(f"steer-ren-s{k}-a{round(100 * f):03d}", "ren", k, f, "steer") for k in SLOTS for f in (0.1, 0.2, 0.3)]


def st_idx(slot):              # SLOT k = output of decoder layer k-1 -> SafeTune layer index (PATHS.md, trap T1)
    return slot - 1


class NoHead(torch.nn.Module):
    """Decoder WITHOUT lm_head for SafeTune's extractor (its full-LM forward OOMs a T4; trap T4, step 1). Same layer modules."""
    def __init__(self, base):
        super().__init__()
        self.model = base

    def forward(self, **kw):
        return self.model(**kw, use_cache=False)


def load_bench():
    p = hf_hub_download(BENCH_REPO, "benchmark_v1.jsonl", repo_type="dataset")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == BENCH_SHA, "wrong benchmark"
    return pd.read_json(p, lines=True)


def load_ctrl():
    p = hf_hub_download(ART_REPO, CTRL_FILE, repo_type="dataset")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == CTRL_SHA, "wrong control file"
    return pd.read_json(p, lines=True)


def build_messages(text):      # must equal generate.build_messages(text, "base")
    return [{"role": "user", "content": text}]


def upload(obj, name, api):
    buf = io.BytesIO()
    if isinstance(obj, str):
        buf.write(obj.encode())
    else:
        torch.save(obj, buf)
    api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=OUT + name, repo_id=ART_REPO, repo_type="dataset",
                    commit_message=f"Phase 4 screen: {name}")


def safetune_dir(model, tok, pos_texts, neg_texts):
    """SafeTune diff of means (pos - neg) at our 3 slots, last token, unnormalised -> unit vectors [3, 2048] (float32, CPU)."""
    cfg = RefusalDirectionConfig(target_layers=[st_idx(k) for k in SLOTS], pick_layer=st_idx(SLOTS[-1]),
                                 pool_method="last_token", normalize=False)
    _, layer, per = extract_refusal_direction(NoHead(model.model), tok, pos_texts, neg_texts, cfg)
    assert layer == st_idx(SLOTS[-1])
    d = torch.stack([per[st_idx(k)].float().cpu() for k in SLOTS])
    return d / d.norm(dim=-1, keepdim=True)


def stage_extract(model, tok, bench, split, V, api, log):
    orig = bench[(bench.variant == "original") & (bench.tier == "H_core")]
    fmap = split["fold_map"]
    out, cos = {}, {}
    for L in ["en", "hi"]:
        rows = orig[(orig.lang == L) & orig.pair_id.isin(split["train_pairs"][L])]
        folds = []
        for f in range(K_FOLD):
            r = rows[rows.pair_id.map(fmap) != f].sort_values("pair_id")
            h, b = r[r.polarity == "harmful"].text.tolist(), r[r.polarity == "benign"].text.tolist()
            assert len(h) == len(b), (L, f, len(h), len(b))
            d = safetune_dir(model, tok, h, b)
            c = (d * V[f"{L}_folds"][f]).sum(-1)                              # [3] cosine per slot vs our file
            cos[f"{L}_fold{f}"] = [round(x, 6) for x in c.tolist()]
            print(f"{L} fold {f}: n={len(h)} pairs | cos vs ours by slot {cos[f'{L}_fold{f}']}", flush=True)
            folds.append(d)
        out[f"{L}_folds"] = torch.stack(folds)                                # [5, 3, 2048]
    c = load_ctrl()
    c = c[c.lang == "en"].sort_values("pair_id")
    food, travel = c[c.side == "food"], c[c.side == "travel"]
    assert (food.pair_id.values == travel.pair_id.values).all() and len(food) == 64
    out["ctrl"] = safetune_dir(model, tok, food.text.tolist(), travel.text.tolist())   # "harmful"=food, "harmless"=travel: just a diff of means
    cos["ctrl"] = [round(x, 6) for x in (out["ctrl"] * V["ctrl"]).sum(-1).tolist()]
    print("ctrl: cos vs ours by slot", cos["ctrl"], flush=True)
    worst = min(min(v) for v in cos.values())
    log["extract"] = dict(cos=cos, worst=worst)
    assert worst >= COS_MIN, f"SafeTune vector disagrees with ours: worst cos {worst}"
    upload(dict(**out, slots=SLOTS, st_idx=[st_idx(k) for k in SLOTS], cos_vs_ours=cos, fold_map=fmap,
                note="unit vectors extracted by SafeTune extract_refusal_direction (normalize=False, then unit); [*, slot, 2048]"),
           "safetune_dirs_v1.pt", api)
    print(f"extract OK (worst cos {worst:.6f}) -> {OUT}safetune_dirs_v1.pt", flush=True)


def vector(D, V, direction, fold, j):
    """The unit vector a prompt in `fold` is steered with at slot index j. r_en / r_hi / ctrl = SafeTune-extracted (3.2a);
    rand = ours (seed 0; nothing to extract)."""
    return {"ren": lambda: D["en_folds"][fold, j], "rhi": lambda: D["hi_folds"][fold, j],
            "ctrl": lambda: D["ctrl"][j], "rand": lambda: V["rand"][j]}[direction]()


@torch.no_grad()
def generate(model, tok, texts, max_new=MAX_NEW):
    enc = tok.apply_chat_template([build_messages(t) for t in texts], add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt", padding=True).to("cuda")
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    out = []
    for row in gen[:, enc["input_ids"].shape[1]:].tolist():           # left padding -> prompts end at the same column
        cut = next((i for i, t in enumerate(row) if t in eos), None)
        ids = row if cut is None else row[:cut]
        out.append(dict(response=tok.decode(ids, skip_special_tokens=True).strip(), ids=ids,
                        n_new_tokens=len(ids), hit_eos=cut is not None))
    return out


def stage_screen(model, tok, bench, split, V, api, log, limit):
    run_conds(model, tok, bench, split, V, api, log, limit, split["dev_uids"], CONDS, MAX_NEW, OUT, "dev")


def stage_full(model, tok, bench, split, V, api, log, limit):
    uids = json.load(open(hf_hub_download(ART_REPO, FULL_UIDS, repo_type="dataset")))["uids"]
    assert len(uids) == 1032, len(uids)
    run_conds(model, tok, bench, split, V, api, log, limit, uids, FULL_CONDS, FULL_NEW, FULL_DIR, "full")


def stage_fullhi(model, tok, bench, split, V, api, log, limit):
    uids = [u for u in json.load(open(hf_hub_download(ART_REPO, FULL_UIDS, repo_type="dataset")))["uids"] if "-hi-" in u]
    assert len(uids) == 248, len(uids)
    run_conds(model, tok, bench, split, V, api, log, limit, uids, HI_CONDS, FULL_NEW, FULL_DIR, "fullhi")


def run_conds(model, tok, bench, split, V, api, log, limit, uid_list, conds, max_new, out_dir, run_tag):
    D = torch.load(hf_hub_download(ART_REPO, DIRS_FILE, repo_type="dataset"), weights_only=False)
    assert D["slots"] == SLOTS and min(min(v) for v in D["cos_vs_ours"].values()) >= COS_MIN
    rows = bench.set_index("uid").loc[uid_list].reset_index()
    rows["fold"] = rows.pair_id.map(split["fold_map"])
    rows["n_tok"] = [len(tok.apply_chat_template(build_messages(t), add_generation_prompt=True, return_dict=True)["input_ids"])
                     for t in rows.text]
    if limit:
        rows = rows.groupby("fold").head(max(1, limit // K_FOLD)).head(limit)
    tag = "smoke" if limit else run_tag
    log["conds"] = {}
    for cid, direction, slot, frac, mode in conds:
        name = f"gen_tinyaya_{cid}_{tag}_v1.jsonl"
        if api.file_exists(ART_REPO, out_dir + name, repo_type="dataset"):
            print("skip (exists):", name, flush=True); continue
        j = SLOTS.index(slot)
        alpha = None if mode == "ablate" else frac * V["delta_en"][j].item()
        out, t0 = [], time.time()
        for f, g in rows.groupby("fold"):                                   # one vector per fold (cross-fitting)
            d = vector(D, V, direction, f, j)
            hook = (RefusalDirectionModel(model, d, mode="ablate", strength=1.0, layers=None) if mode == "ablate" else
                    RefusalDirectionModel(model, d, mode="steer", strength=alpha, layers=[st_idx(slot)]))
            hook.install()
            try:
                g = g.sort_values("n_tok", ascending=False)                  # longest first: an OOM shows up at once
                for i in range(0, len(g), BS):
                    b = g.iloc[i:i + BS]
                    for (_, r), res in zip(b.iterrows(), generate(model, tok, b.text.tolist(), max_new)):
                        out.append(dict(uid=r.uid, model="tinyaya", condition=cid, max_new_tokens=max_new, **res,
                                        direction=direction, slot=slot, st_idx=None if mode == "ablate" else st_idx(slot),
                                        layers="all" if mode == "ablate" else [st_idx(slot)], alpha=alpha, frac=frac,
                                        mode=mode, fold=int(f), vec_source="ours" if direction == "rand" else "safetune"))
            finally:
                hook.remove()
        assert len(out) == len(rows) and len({o["uid"] for o in out}) == len(rows)
        api.upload_file(path_or_fileobj="\n".join(json.dumps(o, ensure_ascii=False) for o in out).encode(),
                        path_in_repo=out_dir + name, repo_id=ART_REPO, repo_type="dataset", commit_message=f"Phase 4: {name}")
        secs = round(time.time() - t0)
        log["conds"][cid] = dict(secs=secs, n=len(out), alpha=alpha, hit_eos=round(sum(o["hit_eos"] for o in out) / len(out), 3),
                                 mean_new=round(sum(o["n_new_tokens"] for o in out) / len(out), 1))
        print(cid, log["conds"][cid], flush=True)
        upload(json.dumps(log, indent=1), f"log_gen_{tag}_v1.json", api)   # progress log (phase4/screen/); tag dev|full|smoke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "screen", "full", "fullhi"], required=True)
    ap.add_argument("--limit", type=int, default=None)      # smoke: first N dev rows per condition, files tagged smoke
    a = ap.parse_args()
    api, bench = HfApi(), load_bench()
    split = json.load(open(hf_hub_download(ART_REPO, SPLIT_FILE, repo_type="dataset")))
    V = torch.load(hf_hub_download(ART_REPO, STEER_FILE, repo_type="dataset"), weights_only=False)
    assert V["slots"] == SLOTS and V["st_idx"] == [st_idx(k) for k in SLOTS], "steer file slot convention mismatch"
    tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda").eval()
    log = dict(stage=a.stage, model_sha=api.model_info(MODEL_ID).sha, transformers=transformers.__version__,
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    print(log, flush=True)
    t0 = time.time()
    if a.stage == "extract":
        stage_extract(model, tok, bench, split, V, api, log)
    elif a.stage == "screen":
        stage_screen(model, tok, bench, split, V, api, log, a.limit)
    elif a.stage == "full":
        stage_full(model, tok, bench, split, V, api, log, a.limit)
    else:
        stage_fullhi(model, tok, bench, split, V, api, log, a.limit)
    log["secs"] = round(time.time() - t0)
    upload(json.dumps(log, indent=1), f"log_{a.stage}_v1.json", api)
    print("DONE", log["secs"], "s", flush=True)


if __name__ == "__main__":
    main()
