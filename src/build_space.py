"""Ship: the write-up (README.md, the submission document) as a static HF Space (no model, no GPU: just the article, figures and links).
  python build_space.py          -> builds space/ (index.html, README.md, figures/) locally; open space/index.html to check
  python build_space.py --push   -> also creates the Space (public, sdk static) and uploads space/"""
import re, shutil, sys
import markdown                                   # pip install markdown
from huggingface_hub import HfApi
from paths import REPO_ROOT, FIGURES, HF_USER

SPACE_ID = f"{HF_USER}/lexsi-refusal"
OUT = REPO_ROOT / "space"
GH = "https://github.com/anantshri1/lexsi-refusal/blob/main/"
TITLE = "Steering a multilingual model's safety from the inside"

md = (REPO_ROOT / "README.md").read_text()
figs = sorted(set(re.findall(r"\]\(figures/([\w.-]+\.png)\)", md)))   # figures are served from the Space itself
md = re.sub(r"\]\((?!https?://|figures/|#)([^)]+)\)", lambda m: "](" + GH + m.group(1) + ")", md)   # repo files -> GitHub
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
body = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>', r'<pre class="mermaid">\1</pre>', body, flags=re.S)
MERMAID = ('<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";'
           'mermaid.initialize({startOnLoad: true});</script>')

CSS = """body{max-width:860px;margin:2rem auto;padding:0 1.2rem;font:17px/1.6 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2328}
h1{font-size:2rem;line-height:1.25}h2{margin-top:2.2rem;border-bottom:1px solid #d0d7de;padding-bottom:.3rem}
img{max-width:100%;height:auto;display:block;margin:1rem auto}
table{border-collapse:collapse;display:block;overflow-x:auto;margin:1rem 0;font-size:15px}
th,td{border:1px solid #d0d7de;padding:.35rem .6rem;text-align:left}th{background:#f6f8fa}
blockquote{margin:1rem 0;padding:.5rem 1rem;border-left:4px solid #d0d7de;color:#57606a;background:#f6f8fa}
code{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px;font-size:90%}a{color:#0969da}"""
html = (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{TITLE}</title>'
        f"<style>{CSS}</style>{MERMAID}</head><body>\n{body}\n</body></html>\n")

README = f"""---
title: Multilingual refusal steering
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: Steering Tiny Aya's English safety into Hindi, Italian, German
---
{TITLE}. Write-up for the Lexsi AI Safety Hackathon (Track 2). Code: https://github.com/anantshri1/lexsi-refusal
"""

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)
(OUT / "index.html").write_text(html)
(OUT / "README.md").write_text(README)
for f in figs:
    shutil.copy(FIGURES / f, OUT / "figures" / f)
print("built", OUT, "| figures:", figs, "| html bytes:", len(html))
assert figs and "](" not in html, "figures missing or a markdown link was not converted"

if "--push" in sys.argv:
    api = HfApi()
    api.create_repo(SPACE_ID, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=SPACE_ID, repo_type="space", commit_message="Ship: write-up Space")
    print("pushed: https://huggingface.co/spaces/" + SPACE_ID)
