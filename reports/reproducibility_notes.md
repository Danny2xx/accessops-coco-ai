# Reproducibility Notes

## Environment
- **Python:** 3.12.11
- **TensorFlow:** 2.16.2
- **GPU:** NVIDIA A100 (Lightning AI studio)
- **OS:** Ubuntu (Lightning AI cloud)

## Random Seeds
All notebooks set `SEED = 42` and call `np.random.seed(SEED)` before any stochastic operation.
TensorFlow's global seed is set via `tf.random.set_seed(42)` where applicable.

## Data
- **Dataset:** MS COCO 2017 (train + val splits)
- **Download:** Automated in `notebooks/01_data_sanity_and_splits.ipynb`
- **Splits:** Val2017 is deterministically split 50/50 into val and test by sorted image filename
- **No image-level leakage:** verified by assertion in notebook

## Tokeniser
- Word-level tokeniser fitted on training captions only
- Saved to `artifacts/stage1c_preprocess/tokenizer.json`
- Vocab size: 30,000 (observed: 29,078 unique words)
- Special tokens: `<unk>` (OOV), `<start>`, `<end>`

## Model Checkpoints
All model weights are saved in the `models/` directory:
- `models/stage2_mlp/mlp_keyword_baseline.keras`
- `models/stage2b_cnn/cnn_keyword_baseline_final.keras`
- `models/stage3_scratch_cnn_lstm/best_stage3_scratch.keras`
- `models/stage4_transfer/best_stage4_finetune.keras`
- `models/stage5_opt/B2_adamw_5e5.keras` (best overall)
- `models/stage6_rl/stage6_rl_epoch1.weights.h5`

## Metrics
- All BLEU scores computed using `nltk.translate.bleu_score`
- Test-set evaluation uses all 2,500 test images (one reference caption per image for BLEU)
- "Fast eval" in stage 5 used a subset; "full eval" used all 2,500 images
- See `artifacts/final/metric_notes.md` for details on evaluation methodology

## Known Limitations
- Original notebooks used hardcoded paths to `/teamspace/studios/this_studio/`
- These have been updated to use the `COCO_PROJECT_ROOT` environment variable pattern
- COCO image data (~19 GB) must be downloaded separately; it is not included in the repository

## How to Reproduce

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set project root: `export COCO_PROJECT_ROOT=/path/to/accessops_coco_ai`
4. Run notebooks 01 through 06 in order
5. Notebook 01 will download COCO data automatically (requires internet + ~20 GB disk)
6. GPU is required for stages 2–6 (training)
