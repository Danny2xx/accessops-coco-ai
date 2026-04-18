"""
download_artifacts.py — Downloads model and tokenizer from Hugging Face Hub
if they are not already present locally.
"""

import os
from pathlib import Path

HF_REPO = os.getenv("HF_REPO", "Danny2xx/accessops-coco-ai")
LOCAL_DIR = Path(__file__).parent / "hf_artifacts"


def ensure_artifacts() -> tuple[Path, Path]:
    """
    Download model + tokenizer from HF Hub if not already cached.
    Returns (model_path, tokenizer_path).
    """
    model_path = LOCAL_DIR / "B2_adamw_5e5.keras"
    tokenizer_path = LOCAL_DIR / "tokenizer.json"

    if model_path.exists() and tokenizer_path.exists():
        print(f"[download] Artifacts already present at {LOCAL_DIR}")
        return model_path, tokenizer_path

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Downloading artifacts from {HF_REPO} ...")

    try:
        from huggingface_hub import hf_hub_download

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

        if not model_path.exists():
            print("[download] Downloading model (~376 MB) ...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename="B2_adamw_5e5.keras",
                local_dir=str(LOCAL_DIR),
                token=hf_token,
            )
            print(f"[download] Model saved to {downloaded}")

        if not tokenizer_path.exists():
            print("[download] Downloading tokenizer ...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename="tokenizer.json",
                local_dir=str(LOCAL_DIR),
                token=hf_token,
            )
            print(f"[download] Tokenizer saved to {downloaded}")

    except Exception as e:
        print(f"[download] ERROR: {e}")
        raise RuntimeError(f"Failed to download artifacts from HF Hub: {e}")

    return model_path, tokenizer_path
