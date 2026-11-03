"""B9 — assemble benchmark_v1.jsonl from reviewed interim files. Run once; refuses to overwrite."""
import json, hashlib, re
import pandas as pd
from paths import DATA_INTERIM as I, DATA_FROZEN as F

OUT = F / "benchmark_v1.jsonl"
assert not OUT.exists(), "frozen file exists — never overwrite; make v2"

sel = pd.read_csv(I / "multijail_selected_v1.csv").rename(columns={"id": "mj_id"})
tw  = pd.read_csv(I / "benign_twins_reviewed.csv")
meta = sel.set_index("mj_id")[["tier", "primary_tag"]]
DROP_PARA = {169}                                           # paraphrase weakened; see notes

def clean(t):                                                # whitespace + NLLB spacing artefacts
    t = " ".join(str(t).split())
    return re.sub(r"\s+([?.!,;:])", r"\1", t).replace(" 's", "'s")

rows = []
def add(mj, pol, lang, var, text, origin, validation, bt=None, notes=""):
    pid = f"mj{int(mj):03d}"
    rows.append(dict(
        uid=f"{pid}-{pol[0]}-{lang}-{'orig' if var=='original' else 'para'}",
        pair_id=pid, mj_id=int(mj) if pol == "harmful" else None, polarity=pol,
        tier=meta.loc[mj, "tier"], primary_tag=meta.loc[mj, "primary_tag"],
        lang=lang, variant=var, text=clean(text), origin=origin, validation=validation,
        bt_text=clean(bt) if isinstance(bt, str) else None, notes=notes))

# --- en original
for _, r in sel.iterrows():
    add(r.mj_id, "harmful", "en", "original", r.en, "multijail_en", "source")
for _, r in tw.iterrows():
    add(r.mj_id, "benign", "en", "original", r.en_benign, "authored:claude", "reviewed:anant")

# --- it/de original (NLLB, sentence-split) + pre-edit overrides
fx_itde = pd.read_csv(I / "mt_itde_nllb_fix.csv").set_index(["mj_id", "lang"])
for lang in ["it", "de"]:
    for _, r in pd.read_csv(I / f"mt_{lang}_nllb_v2.csv").iterrows():
        k = (r.mj_id, lang)
        if r.polarity == "harmful" and k in fx_itde.index:
            f = fx_itde.loc[k]
            add(r.mj_id, r.polarity, lang, "original", f.text, "mt:nllb-200-3.3b+preedit",
                "bt+llm_check", f.bt_text, f"preedit: {f.issue}")
        else:
            add(r.mj_id, r.polarity, lang, "original", r.text, "mt:nllb-200-3.3b",
                "bt+llm_check", r.bt_text, "acronym preedit" if r.preedited else "")

# --- hi/ml original (Google) + overrides
g = pd.read_csv(I / "mt_gsheets.csv")
fx_g = pd.read_csv(I / "mt_gsheets_fix.csv").set_index(["mj_id", "lang"])
GT = "mt:google-translate(2026-11-03)"
for lang in ["hi", "ml"]:
    for _, r in g.iterrows():
        k = (r.mj_id, lang)
        if r.polarity == "harmful" and k in fx_g.index:
            f = fx_g.loc[k]
            add(r.mj_id, r.polarity, lang, "original", f.text, GT + "+preedit", "bt+llm_check", f.bt_text, "preedit")
        else:
            add(r.mj_id, r.polarity, lang, "original", r[lang], GT, "bt+llm_check", r[f"{lang}_bt"])

# --- hi-Latn (transliteration of final hi)
for _, r in pd.read_csv(I / "hi_latn_v1.csv").iterrows():
    add(r.mj_id, r.polarity, "hi-Latn", "original", r.hi_latn, "translit:iast+unidecode(hi)", "derived",
        notes="formal romanisation (inherent vowels kept)")

# --- paraphrases: en (pivot) + hi/ml (Google)
vp = pd.read_csv(I / "variants_paraphrase.csv")
gp = pd.read_csv(I / "mt_gsheets_para.csv").set_index(["mj_id", "polarity"])
for _, r in vp.iterrows():
    if r.mj_id in DROP_PARA: continue
    add(r.mj_id, r.polarity, "en", "paraphrase", r.en_paraphrase, f"paraphrase:{r.origin}", "bt+llm_check")
    p = gp.loc[(r.mj_id, r.polarity)]
    for lang in ["hi", "ml"]:
        add(r.mj_id, r.polarity, lang, "paraphrase", p[lang], GT, "bt+llm_check", p[f"{lang}_bt"])

df = pd.DataFrame(rows)

# --- validation
assert df.uid.is_unique, "duplicate uid"
assert (df.text.str.len() > 0).all(), "empty text"
both = df.groupby(["pair_id", "lang", "variant"]).polarity.nunique()
assert (both == 2).all(), f"unpaired: {both[both != 2].index.tolist()[:5]}"
expected = 133 * 2 * 6 + 29 * 2 * 3
print(df.groupby(["lang", "variant"]).size().unstack(fill_value=0))
print("total:", len(df), "| expected:", expected)
assert len(df) == expected

# --- freeze
F.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for rec in df.to_dict("records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
print("frozen:", OUT, "\nsha256:", sha)