# Vlog Script — AccessOps COCO AI

## Video Info
- **Title:** Image Captioning for Accessibility — End-to-End Pipeline
- **Duration target:** 5–10 minutes
- **Format:** Screen recording with voiceover + slides/notebook walkthrough

---

## SLIDE 1 — Title (0:00–0:15)
**Visual:** Project title, your name, module code  
**Script:**
> "Hi, I'm [Name]. For CMP7225, I built an end-to-end image captioning system for accessibility — automatically generating alt-text descriptions for images so visually impaired users can understand web content. Let me walk you through the project."

---

## SLIDE 2 — The Problem (0:15–0:50)
**Visual:** Image without alt-text → screen reader reads "image" → contrast with image + caption  
**Script:**
> "Over a billion images are uploaded online daily, and many lack alt-text. For the 2.2 billion people with vision impairment, this means missing content. Manual captioning doesn't scale. Our goal: train a model that generates captions automatically, with a safety mechanism for low-confidence predictions."

---

## SLIDE 3 — Data & EDA (0:50–1:40)
**Visual:** Show caption length histogram, examples of COCO images with 5 captions each  
**Figures:** `artifacts/stage1b_eda/figures/03_caption_length_histogram.png`, `06_top30_words.png`  
**Script:**
> "We used MS COCO 2017 — 118,000 training images, each with 5 human-written captions. EDA showed captions average 10 words, follow a Zipf distribution, and have some duplicates — which we keep since they're valid paraphrases. We tokenised at word level with a 30,000-word vocabulary covering 100% of training tokens."

---

## SLIDE 4 — Architecture (1:40–2:40)
**Visual:** Architecture diagram (CNN encoder → feature vector → LSTM decoder → word sequence)  
**Figure:** `reports/figures/architecture_diagram.png` (CREATE THIS)  
**Script:**
> "The model has two parts. The encoder uses MobileNetV2 — pretrained on ImageNet — to convert each image into a 1280-dimensional feature vector. The decoder is an LSTM that takes this vector and generates a caption word by word, trained with teacher forcing.
>
> We first validated this approach with baseline keyword classifiers. An MLP achieved F1 of 0.46 on keyword prediction — interestingly, a CNN performed worse at 0.36, because our features are already spatially aggregated. This told us to treat features as flat vectors, not spatial maps."

---

## SLIDE 5 — Progressive Results (2:40–4:00)
**Visual:** BLEU-4 bar chart across stages, training loss curve  
**Figures:** `artifacts/final/final_bleu_by_stage.png`, `artifacts/stage4/training_curve_loss.png`  
**Script:**
> "Starting from scratch, our CNN+LSTM achieved BLEU-4 of 0.169. Transfer learning — freezing the encoder then fine-tuning — jumped this to 0.219, a 30% improvement.
>
> Then we optimised: AdamW with weight decay outperformed standard Adam, and beam search with length penalty pushed BLEU-4 to 0.247 — a 46% improvement over baseline. This matches published Show-Attend-Tell results of 0.24, without even using attention."

---

## SLIDE 6 — Sample Predictions (4:00–5:00)
**Visual:** Grid of 6 images: 3 good predictions, 3 bad predictions  
**Figures:** From error analysis or `artifacts/stage3/qualitative_examples_top12.md`  
**Script:**
> "Here are some predictions. On the left, the model correctly identifies a giraffe in a field and a kitchen with a stove. On the right, failure cases: it predicts 'laptop' for a motorcycle image, and generates generic 'a man sitting on a bench' for a complex scene. The model converges on common patterns and struggles with unusual compositions."

---

## SLIDE 7 — RL & Human Reroute (5:00–6:30)
**Visual:** SCST diagram, then reroute threshold table  
**Figures:** `artifacts/stage6/reroute_threshold_sweep.csv` as table  
**Script:**
> "We applied Self-Critical Sequence Training — a reinforcement learning method — using BLEU-4 as the reward. Honestly, after one epoch, BLEU-4 didn't improve. This is actually consistent with the literature: the original SCST paper uses CIDEr, not BLEU, and trains for 30 to 50 epochs. BLEU-4 is too sparse as a reward signal.
>
> But here's where the deployment policy saves us. We implemented a confidence-based reroute: the top 50% of predictions by model confidence achieve BLEU-4 of 0.281 — 27% above average — and are auto-published. The bottom 50% go to human review. This gives us a practical, safe deployment."

---

## SLIDE 8 — Use Case & Impact (6:30–7:30)
**Visual:** Deployment architecture diagram (Image → Model → Threshold check → Auto publish / Human queue)  
**Script:**
> "In a real deployment, images are uploaded to a CMS. The model generates a caption with a confidence score. Above the threshold, captions are auto-applied. Below it, they're queued for human editors. This halves the human workload while ensuring quality.
>
> Ethically, we must note that COCO has a Western-centric bias, and incorrect alt-text can actually mislead users. The reroute policy mitigates this — but doesn't eliminate it."

---

## SLIDE 9 — Limitations & Future Work (7:30–8:30)
**Visual:** Bullet list of limitations and future directions  
**Script:**
> "Key limitations: we only evaluated on COCO, we don't use attention, and RL needs more epochs with a better reward like CIDEr. We also only compute BLEU — CIDEr and METEOR would give richer evaluation.
>
> Future work includes: a Transformer decoder with multi-head attention, CIDEr-optimised RL over 10+ epochs, cross-domain evaluation on VizWiz — images captured by blind users — and a real user study with screen reader users."

---

## SLIDE 10 — Summary & Close (8:30–9:00)
**Visual:** Final results table (3 → 4 → 5 progression), project structure tree  
**Script:**
> "To summarise: we built a complete captioning pipeline with 46% BLEU improvement across four stages, a practical deployment policy, and honest analysis of RL challenges. Everything is fully reproducible with versioned checkpoints and documented artifacts.
>
> Thank you for watching."

---

## Production Notes
- Record using OBS Studio or Zoom screen share
- Export at 1080p MP4
- Keep total length between 5 and 10 minutes
- Ensure audio is clear (use headset mic if possible)
- Optional: add simple title cards between sections
