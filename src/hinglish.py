# B6.1 — transliterate final Hindi (Devanagari -> Latin) for all 266 rows
# pip install indic_transliteration unidecode
import pandas as pd
from indic_transliteration import sanscript
from unidecode import unidecode
from paths import DATA_INTERIM

g   = pd.read_csv(DATA_INTERIM / "mt_gsheets.csv")
fix = pd.read_csv(DATA_INTERIM / "mt_gsheets_fix.csv")
fix_hi = fix[fix.lang == "hi"].set_index("mj_id")["text"]          # only #218

g["hi_final"] = g["hi"]
mask = (g.polarity == "harmful") & g.mj_id.isin(fix_hi.index)
g.loc[mask, "hi_final"] = g.loc[mask, "mj_id"].map(fix_hi)
print("hi overrides applied:", mask.sum())                          # expect 1

def to_latn(t):
    iast = sanscript.transliterate(t, sanscript.DEVANAGARI, sanscript.IAST)
    iast = iast.replace("ṃ", "n").replace("m̐", "n").replace("|", ".")              # nasal marks -> n; danda -> .
    iast = iast.replace("ch", "\x00").replace("c", "ch").replace("\x00", "chh")   # c->ch, ch->chh (order matters)
    return unidecode(iast)

g["hi_latn"] = g["hi_final"].apply(to_latn)
out = g[["mj_id", "polarity", "hi_final", "hi_latn"]]
out.to_csv(DATA_INTERIM / "hi_latn_v1.csv", index=False)
print(out.sample(5, random_state=0).to_string())