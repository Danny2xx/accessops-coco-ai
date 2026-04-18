import { useState, useEffect } from "react";
import API_BASE from "../api.js";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { AlertCircle, CheckCircle, ArrowUpRight, ArrowDownRight, FileImage } from "lucide-react";

const TOOLTIP_STYLE = {
  background: "#161b22",
  border: "1px solid #30363d",
  borderRadius: "10px",
  color: "#e6edf3",
  fontSize: "12px",
  padding: "10px 14px",
};

const TABS = [
  { id: "best",        label: "Best BLEU",     icon: CheckCircle,    key: "top_best"         },
  { id: "worst",       label: "Worst BLEU",    icon: AlertCircle,    key: "top_worst"        },
  { id: "rag_up",      label: "RAG Improved",  icon: ArrowUpRight,   key: "top_rag_improved" },
  { id: "rag_down",    label: "RAG Degraded",  icon: ArrowDownRight, key: "top_rag_degraded" },
];

const BUCKET_COLORS = {
  strong_caption:              "#10b981",
  medium_or_ambiguous:         "#6366f1",
  high_confidence_low_quality: "#f59e0b",
  rag_helped_strongly:         "#818cf8",
  rag_hurt_strongly:           "#ef4444",
};

/* ================================================================ */
export default function ErrorAnalysisPage() {
  const [data,     setData]    = useState(null);
  const [tab,      setTab]     = useState("best");
  const [loading,  setLoading] = useState(true);
  const [fetchErr, setFetchErr] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/analysis/error`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setFetchErr(true))
      .finally(() => setLoading(false));
  }, []);

  if (loading)              return <PageSkeleton />;
  if (fetchErr || !data?.available) return <UnavailableCard />;

  const overall = data.overall ?? {};
  const nImages    = parseInt(overall.n_images      ?? 2500);
  const autoRate   = parseFloat(overall.rag_used_rate ?? 0);
  const improvedR  = parseFloat(overall.improved_rate ?? 0);
  const degradedR  = parseFloat(overall.degraded_rate ?? 0);

  /* Bucket chart data */
  const bucketData = (data.error_buckets ?? []).map((b) => ({
    name:  b.error_bucket?.replace(/_/g, " ") ?? "?",
    raw:   b.error_bucket ?? "",
    count: parseInt(b.count) || 0,
  })).sort((a, b) => b.count - a.count);

  /* Route quality data */
  const routeData = (data.route_summary ?? []).map((r) => ({
    route:      r.route_policy ?? "?",
    n:          parseInt(r.n) || 0,
    confidence: parseFloat(r.mean_confidence || 0),
    bleu4:      parseFloat(r.base_bleu4_mean || 0),
    ragBleu4:   parseFloat(r.rag_bleu4_mean || 0),
  }));

  /* Current tab items */
  const currentTab = TABS.find((t) => t.id === tab);
  const items = data[currentTab?.key] ?? [];

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

      {/* ── Page header ─────────────────────────────────────────── */}
      <div>
        <h1 className="text-2xl font-bold">Error Analysis</h1>
        <p className="mt-1 text-sm text-[var(--color-text-dim)]">
          Post-hoc quality breakdown across {nImages.toLocaleString()} MS COCO test images
        </p>
      </div>

      {/* ── Overview stats ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Test images"      value={nImages.toLocaleString()} color="indigo" />
        <StatCard
          label="Strong captions"
          value={bucketData.find((b) => b.raw === "strong_caption")?.count.toLocaleString() ?? "—"}
          sub="BLEU-4 ≥ 0.5"
          color="green"
        />
        <StatCard
          label="RAG retrieval used"
          value={`${(autoRate * 100).toFixed(1)}%`}
          sub="of test images"
          color="purple"
        />
        <StatCard
          label="RAG improved"
          value={`${(improvedR * 100).toFixed(1)}%`}
          sub={`degraded ${(degradedR * 100).toFixed(1)}%`}
          color="amber"
        />
      </div>

      {/* ── Two-column: bucket chart + route table ───────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Bucket chart */}
        <section className="glass p-6" aria-label="Error bucket distribution">
          <h2 className="text-base font-semibold mb-1">Error Bucket Distribution</h2>
          <p className="text-xs text-[var(--color-text-dim)] mb-4">
            Classification of 2,500 test captions by quality category
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={bucketData}
              layout="vertical"
              margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
            >
              <XAxis
                type="number"
                tick={{ fill: "#8b949e", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={160}
                tick={{ fill: "#8b949e", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(v) => [v.toLocaleString(), "images"]}
              />
              <Bar dataKey="count" radius={[0, 5, 5, 0]} maxBarSize={28}>
                {bucketData.map((b, i) => (
                  <Cell key={i} fill={BUCKET_COLORS[b.raw] ?? "#6366f1"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        {/* Route quality table */}
        <section className="glass p-6" aria-label="Route quality breakdown">
          <h2 className="text-base font-semibold mb-1">Route Quality</h2>
          <p className="text-xs text-[var(--color-text-dim)] mb-4">
            AUTO-published vs human-reviewed caption quality
          </p>

          {routeData.length > 0 ? (
            <table className="w-full text-sm" role="table">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  {["Route", "Count", "Avg Conf.", "BLEU-4"].map((h) => (
                    <th key={h} className="text-left py-2 px-2 section-label" scope="col">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {routeData.map((r) => (
                  <tr
                    key={r.route}
                    className="border-b border-[var(--color-border)]/40 hover:bg-[var(--color-surface-3)]/40 transition-colors"
                  >
                    <td className="py-3 px-2">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-semibold
                        ${r.route === "AUTO"
                          ? "bg-emerald-500/15 text-emerald-400"
                          : "bg-amber-500/15 text-amber-400"}`}>
                        {r.route}
                      </span>
                    </td>
                    <td className="py-3 px-2 tabular-nums text-[var(--color-text-dim)]">
                      {r.n.toLocaleString()}
                    </td>
                    <td className="py-3 px-2 tabular-nums text-[var(--color-text-dim)]">
                      {(r.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-2 tabular-nums font-semibold text-[var(--color-accent-2)]">
                      {r.bleu4.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-[var(--color-text-dim)] py-8 text-center">No route data available</p>
          )}

          {/* RAG usage summary */}
          {data.rag_usage?.length > 0 && (
            <div className="mt-5 pt-4 border-t border-[var(--color-border)]">
              <p className="section-label mb-3">RAG Retrieval Usage</p>
              <div className="space-y-2">
                {data.rag_usage.map((r, i) => {
                  const used = r.rag_used_retrieval === "1" || r.rag_used_retrieval === "True";
                  return (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="text-[var(--color-text-dim)]">
                        {used ? "Retrieval used" : "No retrieval"}
                        {" · "}{parseInt(r.n).toLocaleString()} images
                      </span>
                      <span className="font-semibold tabular-nums">
                        BLEU-4 {parseFloat(r.base_bleu4_mean || r.rag_bleu4_mean || 0).toFixed(4)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      </div>

      {/* ── Caption gallery tabs ─────────────────────────────────── */}
      <section aria-label="Caption gallery">
        <h2 className="text-base font-semibold mb-4">Caption Gallery</h2>

        {/* Tab bar */}
        <div className="tab-bar mb-6" role="tablist" aria-label="Caption categories">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              role="tab"
              aria-selected={tab === id}
              aria-controls={`panel-${id}`}
              onClick={() => setTab(id)}
              className={`tab-btn ${tab === id ? "active" : ""}`}
            >
              <span className="flex items-center justify-center gap-1.5">
                <Icon size={12} aria-hidden="true" />
                {label}
              </span>
            </button>
          ))}
        </div>

        {/* Gallery grid */}
        <div
          id={`panel-${tab}`}
          role="tabpanel"
          aria-label={currentTab?.label}
        >
          {items.length === 0 ? (
            <div className="glass p-12 text-center">
              <FileImage size={32} className="mx-auto mb-3 text-[var(--color-text-dim)]" aria-hidden="true" />
              <p className="text-sm text-[var(--color-text-dim)]">No examples available for this category.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
              {items.map((item, i) => (
                <CaptionCard key={i} item={item} tabId={tab} />
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Caption card                                                       */
/* ------------------------------------------------------------------ */
function CaptionCard({ item, tabId }) {
  const bleu4    = parseFloat(item.base_bleu4 || 0);
  const ragBleu4 = parseFloat(item.rag_bleu4  || 0);
  const ragDelta = parseFloat(item.rag_delta_bleu4 || 0);
  const conf     = parseFloat(item.confidence || 0);
  const isRagTab = tabId === "rag_up" || tabId === "rag_down";

  const bleuColor =
    bleu4 > 0.4 ? "text-emerald-400" :
    bleu4 > 0.2 ? "text-[var(--color-accent-2)]" :
    bleu4 > 0.1 ? "text-amber-400" : "text-red-400";

  const bucketColors = {
    strong_caption:              "bg-emerald-500/15 text-emerald-400",
    medium_or_ambiguous:         "bg-[var(--color-accent)]/15 text-[var(--color-accent-2)]",
    high_confidence_low_quality: "bg-amber-500/15 text-amber-400",
    rag_helped_strongly:         "bg-purple-500/15 text-purple-400",
    rag_hurt_strongly:           "bg-red-500/15 text-red-400",
  };
  const bucketClass = bucketColors[item.error_bucket] ?? "bg-[var(--color-surface-3)] text-[var(--color-text-dim)]";

  return (
    <article className="glass p-4 space-y-3 hover:border-[var(--color-accent)]/40 transition-colors duration-200">
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <FileImage size={12} className="text-[var(--color-text-dim)] shrink-0" aria-hidden="true" />
          <span className="text-[11px] text-[var(--color-text-dim)] truncate font-mono">
            {item.image_name ?? "unknown"}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className={`text-[11px] font-semibold px-1.5 py-0.5 rounded-full
            ${item.route_policy === "AUTO"
              ? "bg-emerald-500/15 text-emerald-400"
              : "bg-amber-500/15 text-amber-400"}`}>
            {item.route_policy ?? "?"}
          </span>
        </div>
      </div>

      {/* Generated caption */}
      <div>
        <p className="section-label mb-1">Generated</p>
        <p className="text-sm leading-relaxed">"{item.base_caption ?? "—"}"</p>
      </div>

      {/* Reference caption */}
      {item.ref_1 && (
        <div>
          <p className="section-label mb-1">Reference</p>
          <p className="text-xs text-[var(--color-text-dim)] leading-relaxed italic">"{item.ref_1}"</p>
        </div>
      )}

      {/* RAG caption for RAG tabs */}
      {isRagTab && item.rag_caption && item.rag_caption !== item.base_caption && (
        <div>
          <p className="section-label mb-1">RAG Caption</p>
          <p className="text-xs text-purple-300 leading-relaxed">"{item.rag_caption}"</p>
        </div>
      )}

      {/* Scores */}
      <div className="flex items-center gap-3 pt-1 flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="section-label">BLEU-4</span>
          <span className={`text-sm font-bold tabular-nums ${bleuColor}`}>
            {bleu4.toFixed(4)}
          </span>
        </div>

        {isRagTab && (
          <div className="flex items-center gap-1">
            <span className="section-label">→</span>
            <span className={`text-sm font-bold tabular-nums ${ragDelta >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {ragDelta >= 0 ? "+" : ""}{ragDelta.toFixed(4)}
            </span>
          </div>
        )}

        <div className="flex items-center gap-1.5 ml-auto">
          <span className="section-label">Conf.</span>
          <span className="text-xs tabular-nums text-[var(--color-text-dim)]">
            {(conf * 100).toFixed(0)}%
          </span>
        </div>

        {item.error_bucket && (
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${bucketClass}`}>
            {item.error_bucket.replace(/_/g, " ")}
          </span>
        )}
      </div>
    </article>
  );
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */
function StatCard({ label, value, sub, color }) {
  const colors = {
    indigo: "border-[var(--color-accent)]/30 text-[var(--color-accent-2)]",
    green:  "border-emerald-500/30 text-emerald-400",
    purple: "border-purple-500/30 text-purple-400",
    amber:  "border-amber-500/30 text-amber-400",
  };
  return (
    <div className={`stat-card border ${colors[color] ?? colors.indigo}`}>
      <p className="section-label mb-1">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
      {sub && <p className="text-xs opacity-60 mt-0.5">{sub}</p>}
    </div>
  );
}

function UnavailableCard() {
  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-16 text-center">
      <AlertCircle size={32} className="mx-auto mb-4 text-[var(--color-warning)]" aria-hidden="true" />
      <h2 className="text-lg font-semibold mb-2">Error Analysis</h2>
      <p className="text-sm text-[var(--color-text-dim)]">
        Could not load error analysis data. Ensure the backend is running and artifacts/final/error_analysis/ is present.
      </p>
    </div>
  );
}

function PageSkeleton() {
  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-8 space-y-6" aria-label="Loading" aria-busy="true">
      <div className="skeleton h-8 w-56" />
      <div className="skeleton h-4 w-80" />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => <div key={i} className="skeleton h-24 rounded-xl" />)}
      </div>
      <div className="grid grid-cols-2 gap-6">
        <div className="skeleton h-64 rounded-2xl" />
        <div className="skeleton h-64 rounded-2xl" />
      </div>
      <div className="skeleton h-12 rounded-xl" />
      <div className="grid grid-cols-3 gap-4">
        {[...Array(6)].map((_, i) => <div key={i} className="skeleton h-52 rounded-2xl" />)}
      </div>
    </div>
  );
}
