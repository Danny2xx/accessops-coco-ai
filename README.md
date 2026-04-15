# AccessOps COCO AI — Image Captioning for Accessibility

An end-to-end COCO image captioning pipeline built for the **CMP7225 MSc AI** module, demonstrating progressive model improvement from scratch baselines through transfer learning, optimisation, RL fine-tuning, and a human-reroute deployment policy.

## Project Overview

This project addresses the real-world problem of generating **alt-text for images** to improve web accessibility for visually impaired users. Starting from the MS COCO 2017 dataset (118,287 training images, ~617k captions), it builds and evaluates a series of increasingly sophisticated captioning models:

| Stage | Model | Test BLEU-4 |
|---|---|---|
| 2a | MLP keyword baseline | F1 = 0.458 |
| 2b | CNN keyword baseline | F1 = 0.355 |
| 3 | Scratch CNN+LSTM captioner | 0.169 |
| 4 | Transfer learning (frozen+finetune) | 0.219 |
| 5 | Optimised (AdamW + beam search) | 0.247 |
| 6 | RL (SCST) + Human reroute policy | 0.222* |

*RL regression is analysed in the RL analysis discussion.

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on A100 / T4)
- ~20 GB disk space for COCO images + model checkpoints

### Installation

```bash
git clone <repo-url>
cd accessops_coco_ai
pip install -r requirements.txt
```

### Data Download

The first notebook (`notebooks/01_data_sanity_and_splits.ipynb`) handles downloading and extracting the COCO 2017 dataset automatically. Alternatively, download manually from [cocodataset.org](https://cocodataset.org/#download):

- `train2017.zip`
- `val2017.zip`
- `annotations_trainval2017.zip`

Place the extracted directories under `data/coco/`.

### Environment Variable

Set the project root if running from a non-standard location:

```bash
export COCO_PROJECT_ROOT=/path/to/accessops_coco_ai
```

## Running the Pipeline

Execute notebooks in order:

```
notebooks/01_data_sanity_and_splits.ipynb   # Stage 1a: Download, clean, split
notebooks/01b_eda.ipynb                      # Stage 1b: Exploratory data analysis
notebooks/01c_preprocess_spec.ipynb          # Stage 1c: Tokenisation & preprocessing
notebooks/02_mlp_baseline.ipynb              # Stage 2a: MLP keyword classifier
notebooks/02b_cnn_baseline.ipynb             # Stage 2b: CNN keyword classifier
notebooks/02c_stage2_charts.ipynb            # Stage 2c: Baseline comparison charts
notebooks/03_cnn_lstm_scratch_baseline.ipynb  # Stage 3: Scratch captioner
notebooks/04_transfer_learning_main_model.ipynb # Stage 4: Transfer learning
notebooks/04c_stage4b_optimization.ipynb     # Stage 5: Optimisation & beam search
notebooks/05_stage6_rl_scst.ipynb            # Stage 6: RL fine-tuning & human reroute
notebooks/06_error_analysis.ipynb            # Error analysis (post-training)
```

## Project Structure

```
accessops_coco_ai/
├── README.md
├── requirements.txt
├── notebooks/              # Ordered pipeline notebooks
├── artifacts/              # All stage outputs (metrics, CSVs, figures)
│   ├── captions_clean_with_splits.csv
│   ├── splits/             # Train/val/test image lists
│   ├── stage0_setup/
│   ├── stage1_report.json
│   ├── stage1b_eda/        # EDA figures and tables
│   ├── stage1c_preprocess/ # Tokeniser and config
│   ├── stage2/             # MLP baseline outputs
│   ├── stage2b/            # CNN baseline outputs
│   ├── stage3/             # Scratch captioner outputs
│   ├── stage4/             # Transfer learning outputs
│   ├── stage5/             # Optimisation outputs
│   ├── stage6/             # RL + reroute outputs
│   └── final/              # Final summary metrics and charts
├── models/                 # Saved model checkpoints (.keras, .h5)
└── reports/                # Report, figures, rubric mapping
```

## Key Results

- **Progressive BLEU-4 improvement:** 0.169 → 0.219 → 0.247 across scratch, transfer, and optimised stages (46% relative improvement).
- **Human reroute policy:** Top-50% confidence predictions achieve BLEU-4 = 0.281; low-confidence predictions are routed to human review.
- **RL finding:** SCST with BLEU-4 reward regressed performance after 1 epoch — analysed as a research insight into reward-metric mismatch (see `artifacts/stage6/rl_analysis_discussion.md`).

## Reproducibility

- Random seed: 42 (set in all notebooks)
- Tokeniser: saved as `artifacts/stage1c_preprocess/tokenizer.json`
- Model checkpoints: saved in `models/` per stage
- All metrics: saved as JSON/CSV in `artifacts/`
- See `reports/reproducibility_notes.md` for detailed reproduction instructions.

## License

This project is submitted as coursework for CMP7225 and is not licensed for redistribution.
