import { useState, useEffect } from "react";
import API_BASE from "../api.js";
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { TrendingUp, Cpu, GitBranch, Shield, AlertTriangle } from "lucide-react";

/* ── Chart theme tokens ─────────────────────────────────────────── */
const T = {
  grid:    "#30363d",
  tick:    "#8b949e",
  accent:  "#6366f1",
  accent2: "#818cf8",
  success: "#10b981",
  warning: "#f59e0b",
  surface: "#161b22",
  border:  "#30363d",
  text:    "#e6edf3",
};

const TOOLTIP_STYLE = {
  background: T.surface,
  border: `1px solid ${T.border}`,
  borderRadius: "10px",
  color: T.text,
  fontSize: "12px",
  padding: "10px 14px",
};

/* ── Stage display names ────────────────────────────────────────── */
const STAGE_NAMES = {
  "Stage 3 Scratch":    "S3 Scratch",
  "Stage 4 Transfer":   "S4 Transfer",
  "Stage 5 Optimized":  "S5 Optim.",
  "Stage 6 RL+Reroute": "S6 RL",
  "Stage 7 RAG":        "S7 RAG",
};

/* ================================================================ */
export default function BenchmarkPage() {
  const [summary,  setSummary]  = useState(null);
  const [reroute,  setReroute]  = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [fetchErr, setFetchErr] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/metrics/summary`).then((r) => r.json()),
      fetch(`${API_BASE}/policy/reroute`).then((r) => r.json()),
    ])
      .then(([s, r]) => { setSummary(s); setReroute(r); })
      .catch(() => setFetchErr(true))
      .finally(() => setLoading(false));
  }, []);

  if (loading)  return <PageSkeleton />;
  if (fetchErr) return <UnavailableCard title="Benchmark Dashboard" />;

  /* ── transform data ─────────────────────────────────────────── */
  const stageData = (summary?.stages ?? [])
    .filter((s) => s.bleu4 && parseFloat(s.bleu4) > 0.01)
    .map((s) => ({
      name:   STAGE_NAMES[s.stage] ?? s.stage?.replace("Stage ", "S") ?? "?",
      full:   s.stage ?? "",
      bleu4:  round4(s.bleu4),
      bleu1:  round4(s.bleu1),
      method: s.bleu4_method ?? "",
    }));

  const decodeData = (summary?.decode_ablation ?? [])
    .filter((d) => d.run_id && d.bleu4_fast)
    .map((d) => ({
      name:  d.run_id ?? "",
      label: `beam=${d.beam_size ?? "?"} λ=${d.length_penalty ?? "?"}`,
      bleu4: round4(d.bleu4_fast),
      bleu1: round4(d.bleu1_fast),
    }))
    .sort((a, b) => b.bleu4 - a.bleu4);

  const trainData = (summary?.train_ablation ?? [])
    .filter((d) => d.run_id && d.bleu4_rank && d.run_id !== "B0_baseline_stage4A_metrics")
    .map((d) => ({
      name:  d.run_id ?? "",
      opt:   d.optimizer ?? "—",
      lr:    d.lr ?? "—",
      bleu4: round4(d.bleu4_rank),
    }))
    .sort((a, b) => b.bleu4 - a.bleu4);

  const rerouteData = (reroute?.sweep ?? [])
    .map((r) => ({
      auto:      round4(r.auto_rate),
      autoBleu4: round4(r.auto_bleu4),
      threshold: round4(r.threshold),
    }))
    .sort((a, b) => a.auto - b.auto);

  const selectedThreshold = round4(reroute?.selected?.selected_threshold);
  const selectedAuto      = round4(reroute?.selected?.auto_rate);

  /* ── headline metrics ──────────────────────────────────────── */
  const bestDecode  = decodeData[0]?.bleu4 ?? null;
  const fullBest    = stageData.find((s) => s.full === "Stage 5 Optimized")?.bleu4
                   ?? stageData.find((s) => s.full?.includes("5"))?.bleu4 ?? null;
  const scratch     = stageData.find((s) => s.full === "Stage 3 Scratch")?.bleu4 ?? null;
  const autoPub     = round4(reroute?.selected?.auto_bleu4);
  const improvement = scratch && fullBest ? (((fullBest - scratch) / scratch) * 100).toFixed(0) : null;

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">

      {/* ── Page header ─────────────────────────────────────────── */}
      <div>
        <h1 className="text-2xl font-bold">Benchmark Dashboard</h1>
        <p className="mt-1 text-sm text-[var(--color-text-dim)]">
          End-to-end evaluation across all training stages · MS COCO 2017 · 2,500 test images
        </p>
      </div>

      {/* ── Headline metrics ────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={<TrendingUp size={16} />}
          label="Best BLEU-4 (beam decode)"
          value={bestDecode ? bestDecode.toFixed(4) : "—"}
          sub="beam k=3, λ=0.6, 500-image subset"
          accent="indigo"
        />
        <MetricCard
          icon={<Cpu size={16} />}
          label="Full-set BLEU-4"
          value={fullBest ? fullBest.toFixed(4) : "—"}
          sub="Stage 5 · 2,500 test images"
          accent="purple"
        />
        <MetricCard
          icon={<Shield size={16} />}
          label="Auto-publish BLEU-4"
          value={autoPub ? autoPub.toFixed(4) : "—"}
          sub={`top ${selectedAuto ? Math.round(selectedAuto * 100) : 50}% by confidence`}
          accent="green"
        />
        <MetricCard
          icon={<GitBranch size={16} />}
          label="Improvement vs Scratch"
          value={improvement ? `+${improvement}%` : "—"}
          sub="Stage 3 → Stage 5 full-set"
          accent="amber"
        />
      </div>

      {/* ── Stage BLEU-4 progression ────────────────────────────── */}
      <ChartSection
        title="BLEU-4 Progression by Training Stage"
        desc="Each stage adds optimisations on top of the previous. Stage 4D (attention ablation, BLEU-4≈0) excluded for scale."
      >
        {stageData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={stageData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.grid} vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fill: T.tick, fontSize: 12 }}
                axisLine={{ stroke: T.grid }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 0.32]}
                tick={{ fill: T.tick, fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                cursor={{ fill: "rgba(255,255,255,0.04)" }}
                formatter={(v, name) => [v.toFixed(4), name.toUpperCase()]}
                labelFormatter={(l, payload) => payload?.[0]?.payload?.full ?? l}
              />
              <Legend
                wrapperStyle={{ color: T.tick, fontSize: "12px", paddingTop: "12px" }}
              />
              <Bar dataKey="bleu4" name="BLEU-4" radius={[5, 5, 0, 0]} maxBarSize={56}>
                {stageData.map((_, i) => (
                  <Cell key={i} fill={i === stageData.length - 1 ? T.success : T.accent} />
                ))}
              </Bar>
              <Bar dataKey="bleu1" name="BLEU-1" fill={T.accent2} radius={[5, 5, 0, 0]} opacity={0.6} maxBarSize={56} />
            </BarChart>
          </ResponsiveContainer>
        ) : <NoData />}
      </ChartSection>

      {/* ── Ablation grids ──────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Decode strategy */}
        <ChartSection
          title="Decoding Strategy Ablation"
          desc="All strategies use the same B2_adamw_5e5 checkpoint. Evaluated on 500-image fast subset."
          compact
        >
          {decodeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={decodeData} margin={{ top: 10, right: 10, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={T.grid} vertical={false} />
                <XAxis
                  dataKey="label"
                  tick={{ fill: T.tick, fontSize: 10 }}
                  axisLine={{ stroke: T.grid }}
                  tickLine={false}
                  angle={-25}
                  textAnchor="end"
                />
                <YAxis
                  domain={[0.20, 0.27]}
                  tick={{ fill: T.tick, fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => v.toFixed(3)}
                />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  formatter={(v) => [v.toFixed(4), "BLEU-4"]}
                  labelFormatter={(_, p) => p?.[0]?.payload?.name ?? ""}
                />
                <Bar dataKey="bleu4" name="BLEU-4" radius={[5, 5, 0, 0]} maxBarSize={48}>
                  {decodeData.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? T.success : T.accent} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartSection>

        {/* Training run ablation */}
        <ChartSection
          title="Optimizer & LR Ablation"
          desc="3-epoch fine-tune runs from Stage 4 baseline. Ranked by BLEU-4 on fast subset."
          compact
        >
          {trainData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={trainData} margin={{ top: 10, right: 10, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={T.grid} vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: T.tick, fontSize: 10 }}
                  axisLine={{ stroke: T.grid }}
                  tickLine={false}
                  angle={-25}
                  textAnchor="end"
                />
                <YAxis
                  domain={[0.21, 0.245]}
                  tick={{ fill: T.tick, fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => v.toFixed(3)}
                />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  formatter={(v) => [v.toFixed(4), "BLEU-4"]}
                  labelFormatter={(_, p) => {
                    const d = p?.[0]?.payload;
                    return d ? `${d.opt} lr=${d.lr}` : "";
                  }}
                />
                <Bar dataKey="bleu4" name="BLEU-4" radius={[5, 5, 0, 0]} maxBarSize={48}>
                  {trainData.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? T.success : T.accent} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartSection>
      </div>

      {/* ── Reroute threshold tradeoff ──────────────────────────── */}
      <ChartSection
        title="Human Reroute Threshold Tradeoff"
        desc="Higher auto-publish rate → lower average quality. Selected threshold (▲) balances 50/50 auto/human split."
      >
        {rerouteData.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={rerouteData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.grid} />
              <XAxis
                dataKey="auto"
                tick={{ fill: T.tick, fontSize: 12 }}
                axisLine={{ stroke: T.grid }}
                tickLine={false}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={{ value: "Auto-publish rate", fill: T.tick, fontSize: 11, position: "insideBottom", offset: -10 }}
              />
              <YAxis
                tick={{ fill: T.tick, fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(3)}
                label={{ value: "BLEU-4 (auto subset)", fill: T.tick, fontSize: 11, angle: -90, position: "insideLeft", offset: 12 }}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(v, n) => [v.toFixed(4), n === "autoBleu4" ? "BLEU-4 (auto)" : n]}
                labelFormatter={(v) => `Auto-rate: ${Math.round(v * 100)}%`}
              />
              {selectedAuto && (
                <ReferenceLine
                  x={selectedAuto}
                  stroke={T.warning}
                  strokeDasharray="5 3"
                  label={{ value: "▲ selected", fill: T.warning, fontSize: 11, position: "top" }}
                />
              )}
              <Line
                type="monotone"
                dataKey="autoBleu4"
                name="BLEU-4 (auto)"
                stroke={T.accent}
                strokeWidth={2.5}
                dot={{ fill: T.accent, r: 4, strokeWidth: 0 }}
                activeDot={{ r: 6, fill: T.accent2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : <NoData />}

        {/* Selected policy card */}
        {reroute?.selected?.selected_threshold && (
          <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Threshold",     value: selectedThreshold?.toFixed(4) ?? "—" },
              { label: "Auto rate",     value: `${Math.round(selectedAuto * 100)}%` },
              { label: "Auto BLEU-4",   value: autoPub?.toFixed(4) ?? "—" },
              { label: "Overall BLEU-4",value: round4(reroute?.selected?.overall_bleu4_generated)?.toFixed(4) ?? "—" },
            ].map(({ label, value }) => (
              <div key={label} className="stat-card text-center">
                <p className="section-label mb-1">{label}</p>
                <p className="text-lg font-bold text-[var(--color-text)]">{value}</p>
              </div>
            ))}
          </div>
        )}
      </ChartSection>

      {/* ── Full ablation table ─────────────────────────────────── */}
      {stageData.length > 0 && (
        <ChartSection title="Full Stage Comparison" desc="BLEU scores across all evaluated variants.">
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse" role="table">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  {["Stage", "BLEU-1", "BLEU-4", "Eval Method", "Notes"].map((h) => (
                    <th
                      key={h}
                      className="text-left py-3 px-3 section-label font-semibold"
                      scope="col"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(summary?.stages ?? []).map((s, i) => {
                  const hasBLEU = s.bleu4 && parseFloat(s.bleu4) > 0;
                  return (
                    <tr
                      key={i}
                      className="border-b border-[var(--color-border)]/50 hover:bg-[var(--color-surface-3)]/40 transition-colors"
                    >
                      <td className="py-3 px-3 font-medium">{s.stage ?? "—"}</td>
                      <td className="py-3 px-3 tabular-nums text-[var(--color-text-dim)]">
                        {s.bleu1 ? parseFloat(s.bleu1).toFixed(4) : "—"}
                      </td>
                      <td className="py-3 px-3 tabular-nums">
                        {hasBLEU ? (
                          <span className="font-semibold text-[var(--color-accent-2)]">
                            {parseFloat(s.bleu4).toFixed(4)}
                          </span>
                        ) : "—"}
                      </td>
                      <td className="py-3 px-3 text-xs text-[var(--color-text-dim)]">{s.bleu4_method ?? "—"}</td>
                      <td className="py-3 px-3 text-xs text-[var(--color-text-dim)] max-w-xs truncate">{s.notes ?? ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </ChartSection>
      )}

    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */
function MetricCard({ icon, label, value, sub, accent }) {
  const colors = {
    indigo: "from-indigo-500/20 to-indigo-500/5 border-indigo-500/30 text-[var(--color-accent-2)]",
    purple: "from-purple-500/20 to-purple-500/5 border-purple-500/30 text-purple-400",
    green:  "from-emerald-500/20 to-emerald-500/5 border-emerald-500/30 text-emerald-400",
    amber:  "from-amber-500/20 to-amber-500/5 border-amber-500/30 text-amber-400",
  };
  return (
    <div className={`rounded-2xl border bg-gradient-to-br p-5 ${colors[accent]}`}>
      <div className="flex items-center gap-2 mb-3 opacity-80">{icon}<p className="section-label">{label}</p></div>
      <p className="text-3xl font-bold">{value}</p>
      <p className="text-xs mt-1 opacity-60">{sub}</p>
    </div>
  );
}

function ChartSection({ title, desc, children, compact }) {
  return (
    <section className="glass p-6 space-y-4" aria-label={title}>
      <div>
        <h2 className="text-base font-semibold">{title}</h2>
        {desc && <p className="text-xs text-[var(--color-text-dim)] mt-1">{desc}</p>}
      </div>
      <div className={compact ? "" : ""}>{children}</div>
    </section>
  );
}

function NoData() {
  return (
    <div className="flex items-center justify-center h-40 text-sm text-[var(--color-text-dim)]">
      <AlertTriangle size={14} className="mr-2" aria-hidden="true" /> Data unavailable
    </div>
  );
}

function UnavailableCard({ title }) {
  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-16 text-center">
      <AlertTriangle size={32} className="mx-auto mb-4 text-[var(--color-warning)]" aria-hidden="true" />
      <h2 className="text-lg font-semibold mb-2">{title}</h2>
      <p className="text-sm text-[var(--color-text-dim)]">
        Could not load benchmark data. Make sure the backend is running and artifacts are present.
      </p>
    </div>
  );
}

function PageSkeleton() {
  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-8 space-y-6" aria-label="Loading benchmark data" aria-busy="true">
      <div className="skeleton h-8 w-64" />
      <div className="skeleton h-4 w-96" />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => <div key={i} className="skeleton h-28 rounded-2xl" />)}
      </div>
      <div className="skeleton h-80 rounded-2xl" />
      <div className="grid grid-cols-2 gap-6">
        <div className="skeleton h-64 rounded-2xl" />
        <div className="skeleton h-64 rounded-2xl" />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */
function round4(v) {
  const n = parseFloat(v);
  return isNaN(n) ? null : Math.round(n * 10000) / 10000;
}
