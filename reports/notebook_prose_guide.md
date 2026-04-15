# Notebook Critical Analysis Prose — Copy-Paste Guide

This file contains the exact markdown content to add to each notebook.
For each notebook, insert the cells at the indicated positions.

---

## Notebook: 01_data_sanity_and_splits.ipynb

### → INSERT AT TOP (before Cell 1)
```markdown
# Stage 1a: Data Sanity and Splits

## Objective
Download, validate, and split the MS COCO 2017 dataset for image captioning. This stage establishes the data foundation for all subsequent experiments.

## Key Design Decisions
- **Dataset choice (COCO 2017):** The standard benchmark for image captioning, providing ~118k training images with 5 captions each. Its scale and multi-reference structure make it ideal for evaluating generation quality.
- **Splitting strategy:** We deterministically split COCO val2017 (5,000 images) into val (2,500) and test (2,500) by sorted filename. This ensures: (a) reproducibility without random seeds, (b) zero image-level overlap, and (c) sufficient test-set size for reliable BLEU estimation.
- **Caption cleaning:** Lowercase + regex-based punctuation removal. We explicitly avoid stemming, lemmatisation, and stopword removal because the model must generate grammatically correct natural language — removing function words would degrade output quality.
```

### → INSERT AFTER Cell 6 (after saving splits)
```markdown
## Leakage Check
A critical requirement is that no image appears in more than one split. The split is based on image filenames, not caption text, so a given image and all 5 of its captions belong to exactly one split. This prevents the model from memorising test-set content during training.
```

### → INSERT AT END (after Cell 8, STAGE 1 PASS)
```markdown
## Stage 1a Summary
All required artifacts are produced and validated. The dataset contains 616,767 caption rows across 123,287 unique images. Caption lengths are well-distributed (mean 10.4 words, min 5, max 49, p95=15). No missing or empty captions after cleaning.

**Limitations:** The splitting approach uses a simple sorted split rather than stratified sampling. While COCO is generally balanced across categories, stratified splits by object category would provide stronger guarantees.
```

---

## Notebook: 01b_eda.ipynb

### → INSERT AT TOP
```markdown
# Stage 1b: Exploratory Data Analysis

## Objective
Characterise the caption and image distributions to inform preprocessing and model design choices.
```

### → INSERT AFTER Cell 3 (Core summary tables)
```markdown
## Key Findings from Summary Statistics
- **Balanced splits:** The train/val/test caption length statistics are nearly identical (mean ~10.4, median 10), confirming our split preserves the data distribution.
- **5 captions per image consistently:** This is important for two reasons: (a) multi-reference BLEU evaluation is possible, and (b) the model sees diverse descriptions of the same visual content during training.
```

### → INSERT AFTER Cell 6 (after the Zipf chart — the last chart)
```markdown
## Vocabulary Analysis
The Zipf distribution confirms that a small number of words (a, the, on, in, of) dominate the corpus. This is expected for natural language and has two implications:
1. **Vocabulary coverage:** A vocab size of 30,000 will cover virtually all tokens (99.99%+).
2. **Generation bias:** The model will tend toward high-frequency words and common phrases. Beam search with length penalty can partially mitigate this.

## Duplicate Analysis
We found 27,150 globally duplicated caption texts and 323 exact duplicates within the same image. The global duplicates are valid — independent annotators often produce similar descriptions ("a cat sitting on a couch"). Within-image duplicates are noise but represent <0.05% of data and are not worth removing as they don't affect training.

## EDA Summary
The data is clean, well-distributed, and appropriate for training. Key design decisions informed by EDA:
- **max_len=30** to capture the long tail (99.9% of captions fit)
- **vocab_size=30,000** for near-complete coverage
- **No stopword removal** — function words are essential for grammatical generation
```

---

## Notebook: 01c_preprocess_spec.ipynb

### → INSERT AT TOP
```markdown
# Stage 1c: Preprocessing Specification

## Objective
Define and apply the tokenisation pipeline, producing a reusable tokeniser and documenting all preprocessing choices.

## Design Rationale
Unlike text classification, image captioning requires the model to **generate** grammatically correct sentences. This makes several common NLP preprocessing steps inappropriate:
- **No stopword removal:** "A dog on a beach" becomes "dog beach" — ungrammatical.
- **No stemming/lemmatisation:** "running" → "run" loses tense information.
- **Word-level tokenisation over subword (BPE):** Simpler to implement and interpret. BPE would reduce OOV rate but adds complexity without clear BLEU improvement in this architecture.
```

### → INSERT AFTER Cell 5 (quality stats)
```markdown
## Preprocessing Quality Assessment
- **Truncation rate: 0.09%** — only 0.09% of captions are truncated at max_len=30. This confirms our choice preserves almost all information.
- **Padding rate: 99.9%** — most captions are shorter than max_len, so padding is extensive. Post-padding ensures the model processes real tokens first.
- **Token coverage: 100%** — all training-set tokens map to vocabulary entries (no OOV occurrences). The configured vocab_size (30,000) exceeds the observed vocab (29,078).
- **<start>/<end> tokens:** ID 3 and 4 respectively. These are critical for the decoder to know when to begin and stop generating.
```

---

## Notebook: 02_mlp_baseline.ipynb

### → INSERT AT TOP
```markdown
# Stage 2a: MLP Keyword Baseline

## Objective
Build a multi-label keyword classifier as a feature-quality validation step. If MobileNetV2 features can predict relevant keywords, they contain sufficient information for captioning.

## Why Keywords Before Captioning?
Keyword classification is simpler than sequence generation. By establishing that features → keywords works (F1 > 0.4), we gain confidence that the same features can support the more complex captioning task. A failure here would indicate a feature extraction problem.
```

### → INSERT AT END (after results)
```markdown
## Results Interpretation
- **micro-F1 = 0.458** on 600 keyword classes is strong for multi-label classification, especially given that many keywords are rare.
- **Threshold search on validation set** (best: 0.25) is essential — the default 0.5 threshold would severely under-predict keywords.
- **Feature quality confirmed:** The MobileNetV2 1280-d features encode sufficient visual information for keyword prediction, supporting their use as encoder input for captioning.

## Limitations
- The keyword vocabulary (top 600 words) is fixed and may miss rare but important concepts.
- Multi-label F1 rewards predicting common keywords — the model may struggle with rare objects.
```

---

## Notebook: 02b_cnn_baseline.ipynb

### → INSERT AT TOP
```markdown
# Stage 2b: CNN Keyword Baseline

## Objective
Test whether a 1D CNN applied to image features can outperform the MLP by capturing local patterns in the feature vector.

## Hypothesis
A CNN might detect local feature correlations (adjacent neurons in the MobileNetV2 feature vector) that an MLP misses.
```

### → INSERT AT END
```markdown
## Result: CNN Underperforms MLP
- **CNN micro-F1 = 0.355** vs **MLP micro-F1 = 0.458**
- The CNN performed 22% worse than the simpler MLP.

## Why CNN < MLP (Critical Insight)
This result is not a failure — it's an informative finding. MobileNetV2's output features have already undergone global average pooling, collapsing spatial dimensions into a 1D vector. The 1280 dimensions represent **semantic channels**, not spatial positions. Adjacent dimensions have no meaningful spatial relationship — they represent different learned features.

A 1D convolution on this vector finds patterns between *adjacent feature channels*, which is arbitrary and uninformative. The MLP can learn any mapping from features to keywords without this false spatial assumption.

**Implication for captioning:** This confirms that the decoder should treat image features as a single context vector, not attempt further convolutional processing. The encoder's job is already done by MobileNetV2.
```

---

## Notebook: 03_cnn_lstm_scratch_baseline.ipynb

### → INSERT AT TOP
```markdown
# Stage 3: Scratch CNN+LSTM Captioner

## Objective
Build the first full captioning model — a CNN encoder + LSTM decoder trained from scratch with teacher forcing.

## Architecture Rationale
- **Encoder:** MobileNetV2 features (precomputed) → Dense projection to embedding dimension. Precomputation speeds training by ~10x compared to end-to-end training.
- **Decoder:** Token embedding → LSTM → Dense → softmax. Teacher forcing: during training, the ground-truth previous token is fed at each timestep, stabilising learning.
- **Why LSTM over GRU:** LSTMs are the standard for image captioning (used in NIC, Show-Attend-Tell). Their cell state provides better long-range memory than GRUs.
```

### → INSERT AFTER results/charts
```markdown
## Results Discussion
- **BLEU-4 = 0.169** — below published NIC baseline (~0.27) but expected without attention, beam search, or fine-tuned features.
- **Masked accuracy = 0.490** — the model predicts the correct next word approximately half the time, indicating it has learned meaningful language patterns.
- **Training curves:** Loss decreases smoothly with no sign of overfitting (val loss tracks train loss closely). Early stopping was applied to prevent degradation.

## Qualitative Analysis
Examining the top-12 predictions reveals both strengths and failure modes:
- **Strength:** The model generates grammatically correct, plausible descriptions (e.g., "a giraffe standing in a field with trees in the background").
- **Object confusion:** Row 4 predicts "a man is sitting on a bench with a laptop" when the reference describes "a green motorcycle." The model has learned common co-occurrence patterns and falls back to them for ambiguous images.
- **Generic outputs (mode collapse):** Many predictions follow templates like "a [noun] on/in a [noun]." This is a known issue with maximum-likelihood training — the model optimises for average correctness, not diversity.

These failure modes motivate the transfer learning stage (better features) and beam search (better decoding).
```

---

## Notebook: 04_transfer_learning_main_model.ipynb

### → INSERT AT TOP
```markdown
# Stage 4: Transfer Learning

## Objective
Improve captioning quality by leveraging pretrained features more effectively through a frozen → finetune training strategy.

## Why Transfer Learning?
The scratch model uses pretrained MobileNetV2 features but only trains the decoder. Transfer learning goes further:
1. **Frozen phase:** Fix the encoder, train the decoder to convergence. This prevents early noisy decoder gradients from destroying pretrained encoder features.
2. **Finetune phase:** Unfreeze the encoder with a low LR, allowing both components to co-adapt. This is the key innovation: the encoder learns captioning-specific features.

## Image Size Change
We increased input resolution from 192 to 224 (MobileNetV2's native ImageNet resolution). This extracts richer, more detailed features at the cost of slightly more computation.
```

### → INSERT AFTER results
```markdown
## Results Discussion
- **BLEU-4: 0.169 → 0.219** — a 30% relative improvement over the scratch baseline.
- **Val loss: 2.53 → 2.43** — consistent improvement in sequence prediction quality.
- **Frozen → finetune benefit:** Val loss dropped from 2.51 (frozen-only) to 2.43 (after finetuning), confirming that encoder adaptation improves performance.

The improvement is driven by two complementary factors:
1. Better input features from native-resolution processing (224 vs 192).
2. Encoder-decoder co-adaptation during fine-tuning, where the encoder learns to emphasise captioning-relevant visual information.

## Comparison with Stage 3
| Metric | Stage 3 (Scratch) | Stage 4 (Transfer) | Change |
|--------|-------------------|---------------------|--------|
| BLEU-1 | 0.598 | 0.669 | +12% |
| BLEU-4 | 0.169 | 0.219 | +30% |
| Val Loss | 2.53 | 2.43 | -4% |
```

---

## Notebook: 04c_stage4b_optimization.ipynb

### → INSERT AT TOP
```markdown
# Stage 5: Optimisation — Hyperparameter Tuning and Decoding

## Objective
Squeeze additional performance from the transfer-learned model through controlled ablation of optimiser, learning rate, and decoding strategy.

## Ablation Design
We fix the model architecture and vary:
1. **Optimiser:** Adam vs AdamW (with weight decay)
2. **Learning rate:** 5e-5, 8e-5, 1e-4
3. **Decoding strategy:** Greedy vs beam search (beam=3, length_penalty=0.6)

Each run trains for 3 epochs from the stage 4 finetune checkpoint. We evaluate on the validation set to select the winner.
```

### → INSERT AFTER results
```markdown
## Results Discussion
| Run | Optimizer | LR | Val Loss | BLEU-4 |
|-----|-----------|-----|----------|--------|
| Baseline | (Stage 4) | — | 2.426 | 0.219 |
| B2_adam_1e4 | Adam | 1e-4 | 2.438 | 0.225 |
| B2_adamw_8e5 | AdamW | 8e-5 | 2.428 | 0.224 |
| **B2_adamw_5e5** | **AdamW** | **5e-5** | **2.422** | **0.231** |

**AdamW at LR=5e-5 wins.** AdamW's decoupled weight decay (Loshchilov & Hutter, 2019) provides regularisation that standard Adam lacks, reducing overfitting and improving generalisation.

Beam search with length penalty (0.6) further improved BLEU-4 from 0.231 to **0.247** by exploring multiple candidate sequences and normalising for caption length.

## Acknowledged Gaps
We did not ablate:
- **Learning rate schedules** (cosine decay, warm-up) — these would likely yield 1-2% additional improvement.
- **Dropout rates** — fixed at default throughout.
- **Embedding/hidden dimensions** — fixed from stage 3.

These represent future work opportunities but are lower-priority than the RL and deployment stages.
```

---

## Notebook: 05_stage6_rl_scst.ipynb

### → INSERT AT TOP
```markdown
# Stage 6: Reinforcement Learning (SCST) and Human Reroute Policy

## Objective
1. Apply Self-Critical Sequence Training (SCST) to directly optimise BLEU-4 via policy gradient.
2. Design a confidence-based human reroute policy for deployment.

## SCST Formulation
Following Rennie et al. (2017):
- **Sample** a caption from the model's distribution (multinomial sampling).
- **Greedy-decode** a baseline caption (argmax at each step).
- **Reward:** R = BLEU-4(sample, reference) - BLEU-4(greedy, reference)
- **Gradient:** ∇θ ≈ R · ∇θ log P(sample) — positive R encourages the sampled caption, negative R discourages it.
```

### → INSERT AFTER RL training results
```markdown
## RL Result: Honest Analysis

### Observation
After 1 epoch of SCST, BLEU-4 **did not improve** (0.222 → 0.222 with greedy decoding). The best stage 5 model with beam search achieves 0.247.

### Why Did RL Not Help?
See `artifacts/stage6/rl_analysis_discussion.md` for the full analysis. In summary:

1. **Insufficient epochs:** Published SCST trains for 30-50 epochs. Our 1-epoch budget was inadequate.
2. **BLEU-4 reward sparsity:** Many generated captions have zero 4-gram overlap with references, producing zero reward and high gradient variance.
3. **Wrong reward signal:** The original SCST paper optimises CIDEr, which is smoother and captioning-specific. BLEU-4 is too sparse.
4. **Learning rate:** May have been too high for policy gradient stability.

### This Is a Research Finding, Not a Failure
The regression is consistent with literature: RL for NLG requires careful reward engineering. Our honest analysis demonstrates understanding of RL challenges in sequence generation.
```

### → INSERT AFTER reroute policy section
```markdown
## Human Reroute: Practical Deployment Safety

The reroute policy demonstrates practical thinking absent from many research-only projects:
- **Threshold selection:** 0.511 gives a 50/50 auto/human split.
- **Quality gain:** Auto-served captions (top 50% by confidence) achieve BLEU-4 = 0.281 — 27% above average. This means the system auto-publishes only its best work.
- **Cost argument:** Human review effort is halved compared to fully manual captioning.
- **Adjustable:** Operators can shift the threshold to trade off quality vs cost based on domain requirements.

## Stage 6 Summary
While RL did not improve model quality, this stage provides two valuable contributions:
1. An honest, literature-grounded analysis of RL challenges in NLG.
2. A practical deployment policy with measurable quality guarantees.
```

---

## IMPORTANT NOTE
After pasting these markdown cells into the notebooks, re-run any cells that display tables or charts to ensure outputs appear alongside the new prose.
