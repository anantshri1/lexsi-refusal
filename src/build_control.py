"""Phase 3 control set: neutral TOPIC contrast (food vs travel), pair-matched like the harmful/benign twins (Zhang 2026:
cross-language cosine is only meaningful against a non-safety contrast built with the same estimator and pipeline).
  python src/build_control.py prep   reviewed en -> control_neutral_en_v1.csv (long) + gsheets_ctrl_input.csv (hi, ml)
  python src/build_control.py build  en + NLLB it/de (HF Job) + Google hi/ml (Sheets) + translit hi-Latn
                                     -> DATA_FROZEN/control_neutral_v1.jsonl (+ sha256 printed)
Same MT engines as the benchmark (D3). No back-translation check (no harmful content to soften)."""
import sys, hashlib, json
import pandas as pd
from indic_transliteration import sanscript
from unidecode import unidecode
from paths import DATA_INTERIM as I, DATA_FROZEN as F

SIDE = {"food": "f", "travel": "t"}
LANGS = ["en", "it", "de", "hi", "ml", "hi-Latn"]
ORIGIN = {"en": "authored:claude+reviewed:anant", "it": "mt:nllb-200-3.3b", "de": "mt:nllb-200-3.3b",
          "hi": "mt:google-translate", "ml": "mt:google-translate", "hi-Latn": "translit:iast+unidecode(hi)"}


def to_latn(t):    # copied verbatim from src/hinglish.py (that module runs on import, so not importable)
    iast = sanscript.transliterate(t, sanscript.DEVANAGARI, sanscript.IAST)
    iast = iast.replace("ṃ", "n").replace("m̐", "n").replace("|", ".")
    iast = iast.replace("ch", "\x00").replace("c", "ch").replace("\x00", "chh")
    return unidecode(iast).replace("~", "n").replace("N", "n")


def prep():
    r = pd.read_csv(I / "control_neutral_en_reviewed.csv", dtype=str).fillna("")
    r = r[~r.keep.str.strip().str.lower().isin(["n", "no", "0", "false", "drop"])]      # blank = keep
    rows = [dict(pair_id=x.pair_id, side=side, en=(getattr(x, f"edit_{side}").strip() or getattr(x, f"{side}_en")))
            for x in r.itertuples() for side in SIDE]
    en = pd.DataFrame(rows)
    en.to_csv(I / "control_neutral_en_v1.csv", index=False)
    en.rename(columns={"en": "mt_input"}).to_csv(I / "gsheets_ctrl_input.csv", index=False)
    print(f"{en.pair_id.nunique()} pairs / {len(en)} rows -> control_neutral_en_v1.csv + gsheets_ctrl_input.csv")


def build():
    en = pd.read_csv(I / "control_neutral_en_v1.csv")
    nl = pd.read_csv(I / "mt_ctrl_itde_nllb_v1.csv")[["pair_id", "side", "it", "de"]]     # from the HF Job
    gs = pd.read_csv(I / "mt_gsheets_ctrl.csv")[["pair_id", "side", "hi", "ml"]]          # Sheets export (values, not formulas)
    m = en.merge(nl, on=["pair_id", "side"]).merge(gs, on=["pair_id", "side"])
    assert len(m) == len(en) and m.notna().all().all(), "missing translations"
    m["hi-Latn"] = m.hi.apply(to_latn)
    out = [dict(uid=f"{r.pair_id}-{SIDE[r.side]}-{lang}-orig", pair_id=r.pair_id, side=r.side, lang=lang,
                variant="original", text=r[lang].strip(), origin=ORIGIN[lang])
           for _, r in m.iterrows() for lang in LANGS]
    p = F / "control_neutral_v1.jsonl"
    p.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in sorted(out, key=lambda o: o["uid"])))
    print(len(out), "rows ->", p, "| sha256", hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    {"prep": prep, "build": build}[sys.argv[1]]()
