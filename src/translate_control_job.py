# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers==5.22.0", "accelerate", "huggingface_hub", "pandas", "sentencepiece"]
# ///
"""Phase 3 control set — it/de translation with NLLB-200 3.3B, SAME settings as the benchmark
(notebooks/00_recce.ipynb translate/translate_split: sentence split, num_beams 4, max_new_tokens 256, fp16).
HF Job (single file). In:  ART_REPO control/control_neutral_en_v1.csv (pair_id, side, en)
                     Out: ART_REPO control/mt_ctrl_itde_nllb_v1.csv (pair_id, side, en, it, de)"""
import io, re, torch, pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ART_REPO = "anantshri1/lexsi-refusal-artefacts"
MT_ID = "facebook/nllb-200-3.3B"
TGT = {"it": "ita_Latn", "de": "deu_Latn"}
SPLIT = re.compile(r'(?<=[.!?।])\s+')

tok = AutoTokenizer.from_pretrained(MT_ID, src_lang="eng_Latn")
mt = AutoModelForSeq2SeqLM.from_pretrained(MT_ID, dtype=torch.float16, device_map="cuda", use_safetensors=True).eval()


def translate(texts, tgt, bs=16, max_new=256):                      # copied verbatim from 00_recce
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            gen = mt.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), num_beams=4, max_new_tokens=max_new)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def translate_split(texts, tgt):                                    # copied verbatim from 00_recce
    pieces, owner = [], []
    for i, t in enumerate(texts):
        for s in SPLIT.split(t.strip()):
            if s: pieces.append(s); owner.append(i)
    out = translate(pieces, tgt)
    joined = [[] for _ in texts]
    for i, o in zip(owner, out): joined[i].append(o.strip())
    return [" ".join(j) for j in joined]


src = pd.read_csv(hf_hub_download(ART_REPO, "control/control_neutral_en_v1.csv", repo_type="dataset"))
for lang, tgt in TGT.items():
    src[lang] = translate_split(src.en.tolist(), tgt)
    print(lang, "done |", src[lang].iloc[0], flush=True)
buf = io.BytesIO(); src.to_csv(buf, index=False)
HfApi().upload_file(path_or_fileobj=buf.getvalue(), path_in_repo="control/mt_ctrl_itde_nllb_v1.csv", repo_id=ART_REPO,
                    repo_type="dataset", commit_message="control set: NLLB it/de")
print(src.sample(4, random_state=0)[["en", "it", "de"]].to_string())
