"""Single source of truth for every path. Never write a path string anywhere else."""
import sys
from pathlib import Path

# Repo root = two levels up from this file (src/paths.py -> repo/).
# Works identically locally and in a Colab clone, because it's relative to the code.
REPO_ROOT = Path(__file__).resolve().parent.parent

ON_COLAB = "google.colab" in sys.modules

# Artefacts: on Colab, write to Drive so disconnects don't lose compute.
# Locally, keep them in the repo (git-ignored).
if ON_COLAB:
    ARTEFACT_ROOT = Path("/content/drive/MyDrive/lexsi-refusal")
else:
    ARTEFACT_ROOT = REPO_ROOT

# --- data (frozen benchmark always comes from the repo, so it's versioned) ---
DATA_RAW      = ARTEFACT_ROOT / "data" / "raw"
DATA_INTERIM  = ARTEFACT_ROOT / "data" / "interim"
DATA_FROZEN   = REPO_ROOT / "data" / "frozen"

# --- outputs ---
GENERATIONS   = ARTEFACT_ROOT / "outputs" / "generations"
JUDGEMENTS    = ARTEFACT_ROOT / "outputs" / "judgements"
ACTIVATIONS   = ARTEFACT_ROOT / "outputs" / "activations"
DIRECTIONS    = ARTEFACT_ROOT / "outputs" / "directions"
FIGURES       = REPO_ROOT / "figures"
COMPREHENSION = ARTEFACT_ROOT / "outputs" / "comprehension"

# --- ground-truth files ---
CANONICAL     = REPO_ROOT / "canonical_numbers.txt"

# Create artefact dirs (needed on Colab/Drive, where the .gitkeep skeleton doesn't exist)
for _p in [DATA_RAW, DATA_INTERIM, GENERATIONS, JUDGEMENTS, ACTIVATIONS, DIRECTIONS, FIGURES]:
    _p.mkdir(parents=True, exist_ok=True)

# --- Hugging Face Hub ids (sync layer + submission) ---
HF_USER          = "anantshri1"
HF_BENCH_REPO    = f"{HF_USER}/lexsi-refusal-bench"       # frozen benchmark only
HF_ARTEFACT_REPO = f"{HF_USER}/lexsi-refusal-artefacts"   # generations, judgements, directions, logs
MODEL_ID         = "CohereLabs/tiny-aya-global"           # D1; fallback "CohereLabs/aya-expanse-8b"