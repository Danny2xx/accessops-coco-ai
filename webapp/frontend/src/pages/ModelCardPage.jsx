import { Shield, Cpu, GitBranch, AlertTriangle, CheckCircle, Info, Layers } from "lucide-react";

/* ================================================================ */
export default function ModelCardPage() {
  return (
    <div className="w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

      {/* ── Header ──────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center gap-3 mb-3">
          <div className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[var(--color-accent)]/15 text-[var(--color-accent-2)] border border-[var(--color-accent)]/30">
            Model Card v1.0
          </div>
          <div className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            Inference-only
          </div>
        </div>
        <h1 className="text-2xl font-bold">AccessOps COCO AI — Image Captioning</h1>
        <p className="mt-2 text-sm text-[var(--color-text-dim)] leading-relaxed max-w-2xl">
          A CNN+LSTM image captioning system trained on MS COCO 2017, optimised through seven development stages,
          and deployed with a confidence-based human reroute policy for safe production use.
        </p>
      </div>

      {/* ── Quick facts ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Task",           value: "Image Captioning"   },
          { label: "Dataset",        value: "MS COCO 2017"       },
          { label: "Train images",   value: "118,287"            },
          { label: "Test images",    value: "2,500"              },
          { label: "Architecture",   value: "MobileNetV2 + LSTM" },
          { label: "Vocabulary",     value: "30,000 tokens"      },
          { label: "Best BLEU-4",    value: "0.2473 (beam)"      },
          { label: "Version",        value: "v1.0"               },
        ].map(({ label, value }) => (
          <div key={label} className="stat-card">
            <p className="section-label mb-1">{label}</p>
            <p className="text-sm font-semibold">{value}</p>
          </div>
        ))}
      </div>

      {/* ── Model pipeline ──────────────────────────────────────── */}
      <Card icon={<Layers size={16} />} title="Model Architecture">
        <div className="space-y-4">
          <p className="text-sm text-[var(--color-text-dim)] leading-relaxed">
            The pipeline follows a standard encoder–decoder design. MobileNetV2 encodes spatial image
            features; an LSTM decoder generates tokens auto-regressively, conditioned on both the
            image features and the previous token at each step.
          </p>

          {/* Pipeline diagram */}
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono overflow-x-auto py-2">
            {[
              { label: "Image\n224×224", bg: "bg-[var(--color-surface-3)]" },
              { arrow: true },
              { label: "MobileNetV2\n(encoder)", bg: "bg-[var(--color-accent)]/15 border border-[var(--color-accent)]/30" },
              { arrow: true },
              { label: "7×7×1280\nfeature map", bg: "bg-[var(--color-surface-3)]" },
              { arrow: true },
              { label: "LSTM\n(decoder)", bg: "bg-purple-500/15 border border-purple-500/30" },
              { arrow: true },
              { label: "Beam Search\nk=3, λ=0.6", bg: "bg-[var(--color-surface-3)]" },
              { arrow: true },
              { label: "Caption\n+ confidence", bg: "bg-emerald-500/15 border border-emerald-500/30" },
            ].map((item, i) => (
              item.arrow
                ? <span key={i} className="text-[var(--color-text-dim)]">→</span>
                : (
                  <div
                    key={i}
                    className={`px-3 py-2 rounded-lg text-center whitespace-pre-line leading-tight ${item.bg}`}
                  >
                    {item.label}
                  </div>
                )
            ))}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
            <SpecTable title="Encoder" rows={[
              ["Base model",    "MobileNetV2 (ImageNet)"],
              ["Input size",    "224 × 224 × 3"],
              ["Output",        "7 × 7 × 1280 spatial map"],
              ["Phase 1",       "Frozen (all layers)"],
              ["Phase 2",       "Last 40 layers unfrozen"],
            ]} />
            <SpecTable title="Decoder" rows={[
              ["Architecture",  "LSTM (2 layers)"],
              ["Embedding dim", "384"],
              ["Hidden dim",    "512"],
              ["Vocabulary",    "30,000 tokens"],
              ["Max length",    "30 tokens"],
            ]} />
          </div>
        </div>
      </Card>

      {/* ── Training stages ─────────────────────────────────────── */}
      <Card icon={<GitBranch size={16} />} title="Training Stages">
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse" role="table">
            <thead>
              <tr className="border-b border-[var(--color-border)]">
                {["Stage", "Description", "Key Result", "BLEU-4"].map((h) => (
                  <th key={h} className="text-left py-3 px-3 section-label" scope="col">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {STAGES.map((s, i) => (
                <tr
                  key={i}
                  className="border-b border-[var(--color-border)]/50 hover:bg-[var(--color-surface-3)]/40 transition-colors"
                >
                  <td className="py-3 px-3 font-medium text-[var(--color-accent-2)] whitespace-nowrap">
                    {s.stage}
                  </td>
                  <td className="py-3 px-3 text-[var(--color-text-dim)] text-xs leading-relaxed max-w-xs">
                    {s.desc}
                  </td>
                  <td className="py-3 px-3 text-xs">{s.result}</td>
                  <td className="py-3 px-3 tabular-nums font-semibold">{s.bleu4}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-[var(--color-text-dim)] mt-3">
          * Stage 4D (cross-attention ablation): BLEU-4 ≈ 0.0001 due to incomplete Phase 1 training
          (interrupted at epoch 3/6 due to compute constraints).
          Fine-tune destabilised partially-trained attention weights (val_loss 4.48→5.17).
        </p>
      </Card>

      {/* ── Deployment policy ───────────────────────────────────── */}
      <Card icon={<Shield size={16} />} title="Deployment Policy">
        <div className="space-y-4">
          <p className="text-sm text-[var(--color-text-dim)] leading-relaxed">
            Every prediction is assigned a confidence score and routed by a threshold policy.
            The threshold was calibrated on 2,500 test images to achieve a 50/50 auto/human split,
            balancing quality guarantees with operational throughput.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <PolicyBlock
              title="AUTO-PUBLISH"
              color="green"
              items={[
                "Confidence ≥ 51.15%",
                "~50% of all predictions",
                "Avg BLEU-4: 0.281",
                "No human review required",
              ]}
            />
            <PolicyBlock
              title="HUMAN REVIEW"
              color="amber"
              items={[
                "Confidence < 51.15%",
                "~50% of all predictions",
                "Avg BLEU-4: 0.163",
                "Flagged for human correction",
              ]}
            />
            <PolicyBlock
              title="Threshold Selection"
              color="indigo"
              items={[
                "Swept 7 threshold values",
                "20%→80% auto rate",
                "Selected balanced 50/50",
                "Threshold: 0.5115",
              ]}
            />
          </div>

          <div className="p-4 rounded-xl bg-[var(--color-surface-3)]/60 text-xs text-[var(--color-text-dim)] space-y-1.5">
            <p className="font-semibold text-[var(--color-text)]">Confidence Formula</p>
            <p className="font-mono bg-[var(--color-surface-2)] px-3 py-2 rounded-lg inline-block">
              confidence = sigmoid(mean_log_prob − 1.0)
            </p>
            <p className="leading-relaxed mt-2">
              Log-probability is averaged across generated tokens. The offset of −1.0 centres
              the sigmoid so that typical beam-search log-probs map to a useful [0, 1] range.
              Values below the threshold indicate high uncertainty or visual ambiguity.
            </p>
          </div>
        </div>
      </Card>

      {/* ── Evaluation ──────────────────────────────────────────── */}
      <Card icon={<Cpu size={16} />} title="Evaluation Methodology">
        <div className="space-y-4 text-sm text-[var(--color-text-dim)] leading-relaxed">
          <p>
            All captioning metrics are computed using corpus-level BLEU (NLTK) with individual
            sentence-level smoothing (method1) to avoid zero-scores for short captions.
            Each test image has multiple reference captions (5 per COCO image).
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <SpecTable title="BLEU Metrics" rows={[
              ["BLEU-1", "Unigram precision (recall proxy)"],
              ["BLEU-2", "Bigram precision"],
              ["BLEU-3", "Trigram precision"],
              ["BLEU-4", "4-gram precision (primary metric)"],
            ]} />
            <SpecTable title="Eval Variants" rows={[
              ["full_2500",    "All 2,500 test images, greedy decode"],
              ["fast_subset",  "500 images, beam k=3, λ=0.6"],
              ["Beam k=3 λ=0.6", "Best decode config (BLEU-4 0.247)"],
            ]} />
          </div>

          <p className="text-xs">
            Note: fast_subset scores (beam decode) are not directly comparable to full_2500 scores (greedy decode).
            The honest full-to-full improvement from Stage 3 scratch to Stage 5 optimised is
            <span className="font-semibold text-[var(--color-text)]"> +32% </span>
            (0.169 → 0.222).
          </p>
        </div>
      </Card>

      {/* ── Limitations & ethics ────────────────────────────────── */}
      <Card icon={<AlertTriangle size={16} />} title="Limitations &amp; Ethical Considerations">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">

          <div className="space-y-3">
            <p className="section-label">Known Limitations</p>
            {[
              "BLEU-4 has known weaknesses for evaluating caption diversity and semantics.",
              "Model was trained/evaluated on English COCO captions only — no multilingual support.",
              "RL (SCST) training regressed performance after 1 epoch; a single BLEU-4 reward is sparse and noisy.",
              "RAG retrieval marginally degraded full-set BLEU-4 (−0.001); retrieval similarity threshold tuning helps but retrieval quality is limited by embedding alignment.",
              "Cross-attention ablation (Stage 4D) failed due to compute constraints — incomplete Phase 1 training led to degenerate attention weights.",
            ].map((item, i) => (
              <div key={i} className="flex gap-2 text-sm text-[var(--color-text-dim)]">
                <AlertTriangle size={13} className="text-amber-400 mt-0.5 shrink-0" aria-hidden="true" />
                {item}
              </div>
            ))}
          </div>

          <div className="space-y-3">
            <p className="section-label">Ethical Considerations</p>
            {[
              "MS COCO 2017 reflects Western/English-speaking photographic contexts and may underperform on underrepresented visual content.",
              "Auto-published captions are not perfect; the 50/50 threshold means half of all captions still require human review.",
              "The system is designed as an assistive tool — not a replacement for human accessibility work.",
              "Confidence scores are sigmoid-transformed log-probabilities, not calibrated probability estimates.",
              "This is a v1.0 prototype. Production deployment would require additional safety evaluation, bias auditing, and accessibility testing.",
            ].map((item, i) => (
              <div key={i} className="flex gap-2 text-sm text-[var(--color-text-dim)]">
                <Info size={13} className="text-[var(--color-accent-2)] mt-0.5 shrink-0" aria-hidden="true" />
                {item}
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* ── What went well ──────────────────────────────────────── */}
      <Card icon={<CheckCircle size={16} />} title="Key Findings">
        {[
          ["Transfer learning was the single biggest gain", "Freezing MobileNetV2 and fine-tuning on COCO lifted BLEU-4 from 0.169 (scratch) to 0.219 — a 30% improvement with far less compute."],
          ["Beam search decisively outperforms greedy", "Beam k=3 with length penalty λ=0.6 achieved BLEU-4 0.247 vs 0.231 greedy on the same 500-image subset — a 7% relative improvement."],
          ["RL (SCST) did not improve beyond 1 epoch", "Sparse BLEU-4 reward, greedy baseline noise, and insufficient epochs (only 1) caused slight regression. CIDEr reward and multi-epoch training are recommended for future work."],
          ["Human reroute policy is effective", "Routing top-50% confidence predictions to auto-publish achieves BLEU-4 0.281 vs 0.222 overall — 27% quality uplift for auto-published content."],
          ["RAG retrieval is selective but marginally helpful", "Only 8.3% of test images triggered retrieval, and the average delta was −0.001 full-set BLEU-4. RAG improved 62.8% of triggered cases but hurt the 3.9% with poor retrieval matches."],
        ].map(([title, body], i) => (
          <div key={i} className={`py-4 ${i > 0 ? "border-t border-[var(--color-border)]" : ""}`}>
            <p className="font-semibold text-sm mb-1">{title}</p>
            <p className="text-sm text-[var(--color-text-dim)] leading-relaxed">{body}</p>
          </div>
        ))}
      </Card>

    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Static data                                                        */
/* ------------------------------------------------------------------ */
const STAGES = [
  {
    stage:  "Stage 2a",
    desc:   "MLP keyword baseline on COCO category labels",
    result: "F1 = 0.458",
    bleu4:  "—",
  },
  {
    stage:  "Stage 2b",
    desc:   "CNN feature extraction + keyword classification",
    result: "F1 = 0.355",
    bleu4:  "—",
  },
  {
    stage:  "Stage 3",
    desc:   "CNN+LSTM trained from scratch (img_size=192)",
    result: "First end-to-end captioning baseline",
    bleu4:  "0.169",
  },
  {
    stage:  "Stage 4",
    desc:   "MobileNetV2 encoder with ImageNet weights; frozen then fine-tuned (img_size=224)",
    result: "+30% vs scratch via transfer learning",
    bleu4:  "0.219",
  },
  {
    stage:  "Stage 4D*",
    desc:   "Cross-attention ablation: MultiHeadAttention (4 heads, key_dim=128) over 7×7 spatial tokens",
    result: "Attention weights collapsed — Phase 1 interrupted at epoch 3/6",
    bleu4:  "0.0001",
  },
  {
    stage:  "Stage 5",
    desc:   "Optimizer & LR sweep (AdamW/Adam, 3 LRs) + beam search ablation",
    result: "B2_adamw_5e5 best training run; beam k=3 λ=0.6 best decode",
    bleu4:  "0.231 (fast) / 0.222 (full)",
  },
  {
    stage:  "Stage 6",
    desc:   "SCST reinforcement learning (1 epoch) + human reroute threshold sweep",
    result: "RL did not improve; reroute policy selected at threshold 0.511",
    bleu4:  "0.222 (RL) / 0.281 (auto subset)",
  },
  {
    stage:  "Stage 7",
    desc:   "RAG caption refinement with FAISS retrieval over COCO caption embeddings",
    result: "8.3% retrieval rate; marginal −0.001 full-set delta",
    bleu4:  "0.221",
  },
];

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */
function Card({ icon, title, children }) {
  return (
    <section className="glass p-6 space-y-5" aria-label={title}>
      <div className="flex items-center gap-2 text-[var(--color-accent-2)]">
        {icon}
        <h2 className="text-base font-semibold text-[var(--color-text)]">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function SpecTable({ title, rows }) {
  return (
    <div>
      <p className="section-label mb-2">{title}</p>
      <table className="w-full text-xs" role="table">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b border-[var(--color-border)]/40">
              <td className="py-1.5 pr-3 text-[var(--color-text-dim)] whitespace-nowrap">{k}</td>
              <td className="py-1.5 text-[var(--color-text)]">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PolicyBlock({ title, color, items }) {
  const styles = {
    green:  "border-emerald-500/30 bg-emerald-500/5",
    amber:  "border-amber-500/30 bg-amber-500/5",
    indigo: "border-[var(--color-accent)]/30 bg-[var(--color-accent)]/5",
  };
  const titleStyles = {
    green:  "text-emerald-400",
    amber:  "text-amber-400",
    indigo: "text-[var(--color-accent-2)]",
  };
  return (
    <div className={`p-4 rounded-xl border ${styles[color]}`}>
      <p className={`text-xs font-bold mb-3 ${titleStyles[color]}`}>{title}</p>
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="text-xs text-[var(--color-text-dim)] flex items-start gap-1.5">
            <span className={`mt-1 w-1 h-1 rounded-full shrink-0 ${
              color === "green" ? "bg-emerald-400" : color === "amber" ? "bg-amber-400" : "bg-[var(--color-accent)]"
            }`} />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
