# Rubric Mapping (Stage 0)

## Goal
Build a high-grade COCO image-captioning project with clear evidence per criterion.

## Criteria Coverage Plan
- Understanding of image/text as data (4%):
  - Data sanity checks, caption cleaning rationale, tokenisation decisions.
- Vlog visuals (17%):
  - Clean architecture diagram, training curves, sample predictions table.
- Vlog narrative (13%):
  - Problem -> method -> results -> limitations -> next steps.
- Technical challenge (21%):
  - Scratch baseline + transfer learning + tuned experiments.
- Optimisation (13%):
  - Learning-rate scheduling, early stopping, ablations, threshold/decoding tuning.
- RL knowledge (8%):
  - Human-review routing policy stage and reward framing.
- Use case and impact (17%):
  - Accessibility/content-ops business use case and deployment framing.
- Evaluation and testing (8%):
  - BLEU metrics, qualitative examples, error analysis, reproducibility artifacts.

## Evidence Files to Produce
- artifacts/*/metrics.json
- artifacts/*/training_history.csv
- artifacts/*/sample_predictions.csv
- reports/figures/*.png
- reports/tables/*.csv
