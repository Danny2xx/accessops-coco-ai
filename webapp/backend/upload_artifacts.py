"""
Run once locally to upload all required artifacts to HuggingFace Hub.
Usage: python3 upload_artifacts.py
"""
import os
from pathlib import Path
from huggingface_hub import HfApi

HF_REPO = "Danny2xx/accessops-coco-ai"
ROOT = Path(__file__).resolve().parents[2]  # project root

FILES = {
    # RAG corpus
    "artifacts/stage7/retrieval_candidates_topk.csv": ROOT / "artifacts/stage7/retrieval_candidates_topk.csv",
    # Benchmark
    "artifacts/final/final_metrics_summary.csv": ROOT / "artifacts/final/final_metrics_summary.csv",
    "artifacts/final/deployment_policy.json": ROOT / "artifacts/final/deployment_policy.json",
    "artifacts/stage6/reroute_threshold_sweep.csv": ROOT / "artifacts/stage6/reroute_threshold_sweep.csv",
    "artifacts/stage8_ablation/tables/decode_ablation.csv": ROOT / "artifacts/stage8_ablation/tables/decode_ablation.csv",
    "artifacts/stage8_ablation/tables/train_optimizer_ablation.csv": ROOT / "artifacts/stage8_ablation/tables/train_optimizer_ablation.csv",
    # Error analysis
    "artifacts/final/error_analysis/overall_summary.csv": ROOT / "artifacts/final/error_analysis/overall_summary.csv",
    "artifacts/final/error_analysis/error_bucket_counts.csv": ROOT / "artifacts/final/error_analysis/error_bucket_counts.csv",
    "artifacts/final/error_analysis/route_summary.csv": ROOT / "artifacts/final/error_analysis/route_summary.csv",
    "artifacts/final/error_analysis/rag_usage_summary.csv": ROOT / "artifacts/final/error_analysis/rag_usage_summary.csv",
    "artifacts/final/error_analysis/top20_best_base_bleu4.csv": ROOT / "artifacts/final/error_analysis/top20_best_base_bleu4.csv",
    "artifacts/final/error_analysis/top20_worst_base_bleu4.csv": ROOT / "artifacts/final/error_analysis/top20_worst_base_bleu4.csv",
    "artifacts/final/error_analysis/top20_rag_improved.csv": ROOT / "artifacts/final/error_analysis/top20_rag_improved.csv",
    "artifacts/final/error_analysis/top20_rag_degraded.csv": ROOT / "artifacts/final/error_analysis/top20_rag_degraded.csv",
}

api = HfApi()
for repo_path, local_path in FILES.items():
    if not local_path.exists():
        print(f"SKIP (not found): {local_path}")
        continue
    print(f"Uploading {repo_path} ...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=HF_REPO,
        repo_type="model",
    )
    print(f"  ✓ done")

print("\nAll artifacts uploaded.")
