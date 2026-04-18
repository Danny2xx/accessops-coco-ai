"""
download_artifacts.py — Downloads model, tokenizer, and all artifacts from HF Hub.
"""

import os
from pathlib import Path

HF_REPO = os.getenv("HF_REPO", "Danny2xx/accessops-coco-ai")
LOCAL_DIR = Path(__file__).parent / "hf_artifacts"

ARTIFACT_FILES = [
    "artifacts/stage7/retrieval_candidates_topk.csv",
    "artifacts/final/final_metrics_summary.csv",
    "artifacts/final/deployment_policy.json",
    "artifacts/stage6/reroute_threshold_sweep.csv",
    "artifacts/stage8_ablation/tables/decode_ablation.csv",
    "artifacts/stage8_ablation/tables/train_optimizer_ablation.csv",
    "artifacts/final/error_analysis/overall_summary.csv",
    "artifacts/final/error_analysis/error_bucket_counts.csv",
    "artifacts/final/error_analysis/route_summary.csv",
    "artifacts/final/error_analysis/rag_usage_summary.csv",
    "artifacts/final/error_analysis/top20_best_base_bleu4.csv",
    "artifacts/final/error_analysis/top20_worst_base_bleu4.csv",
    "artifacts/final/error_analysis/top20_rag_improved.csv",
    "artifacts/final/error_analysis/top20_rag_degraded.csv",
]


def ensure_artifacts() -> tuple[Path, Path]:
    """Download all artifacts from HF Hub if not already cached."""
    model_path = LOCAL_DIR / "B2_adamw_5e5.keras"
    tokenizer_path = LOCAL_DIR / "tokenizer.json"

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

        if not model_path.exists():
            print("[download] Downloading model (~376 MB) ...")
            hf_hub_download(repo_id=HF_REPO, filename="B2_adamw_5e5.keras",
                            local_dir=str(LOCAL_DIR), token=hf_token)
            print(f"[download] Model saved to {model_path}")
        else:
            print(f"[download] Model already present")

        if not tokenizer_path.exists():
            print("[download] Downloading tokenizer ...")
            hf_hub_download(repo_id=HF_REPO, filename="tokenizer.json",
                            local_dir=str(LOCAL_DIR), token=hf_token)
            print(f"[download] Tokenizer saved")

        for hf_path in ARTIFACT_FILES:
            local_path = LOCAL_DIR / hf_path
            if local_path.exists():
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[download] Downloading {hf_path} ...")
            hf_hub_download(repo_id=HF_REPO, filename=hf_path,
                            local_dir=str(LOCAL_DIR), token=hf_token)

        print("[download] All artifacts ready")

    except Exception as e:
        print(f"[download] ERROR: {e}")
        raise RuntimeError(f"Failed to download artifacts from HF Hub: {e}")

    return model_path, tokenizer_path
