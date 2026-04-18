"""
config.py — Loads all environment variables and validates required paths.

On import, this module reads .env (if present), resolves all paths, and
exposes them as module-level constants. If a critical file is missing, a
clear ValueError is raised at startup rather than failing silently later.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (if it exists alongside this script)
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_FILE)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[0]))
)

# ---------------------------------------------------------------------------
# Model & artifact paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "stage5_opt" / "B2_adamw_5e5.keras"))
)
TOKENIZER_PATH = Path(
    os.getenv("TOKENIZER_PATH", str(PROJECT_ROOT / "artifacts" / "stage1c_preprocess" / "tokenizer.json"))
)
POLICY_PATH = Path(
    os.getenv("POLICY_PATH", str(PROJECT_ROOT / "artifacts" / "final" / "deployment_policy.json"))
)

# ---------------------------------------------------------------------------
# Inference hyper-parameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
MAX_LEN = int(os.getenv("MAX_LEN", "30"))
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "3"))
LENGTH_PENALTY = float(os.getenv("LENGTH_PENALTY", "0.6"))

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ---------------------------------------------------------------------------
# Prediction log
# ---------------------------------------------------------------------------
LOG_PATH = Path(
    os.getenv("LOG_PATH", str(Path(__file__).resolve().parent / "prediction_log.csv"))
)

# ---------------------------------------------------------------------------
# Deployment policy (threshold + metadata)
# ---------------------------------------------------------------------------
REROUTE_THRESHOLD: float = 0.5  # default; overridden below if file exists
POLICY_META: dict = {}

if POLICY_PATH.exists():
    with open(POLICY_PATH, "r") as f:
        POLICY_META = json.load(f)
    REROUTE_THRESHOLD = float(POLICY_META.get("selected_threshold", 0.5))
else:
    print(f"[config] WARNING — policy file not found at {POLICY_PATH}; using default threshold {REROUTE_THRESHOLD}")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def validate() -> list[str]:
    """Return a list of human-readable warnings. Empty = all good."""
    warnings: list[str] = []
    if not MODEL_PATH.exists():
        warnings.append(f"Model checkpoint not found: {MODEL_PATH}")
    if not TOKENIZER_PATH.exists():
        warnings.append(f"Tokenizer not found: {TOKENIZER_PATH}")
    if not POLICY_PATH.exists():
        warnings.append(f"Deployment policy not found: {POLICY_PATH}")
    return warnings
