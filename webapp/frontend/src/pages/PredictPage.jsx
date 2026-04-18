import { useState, useCallback, useRef, useEffect } from "react";
import { Upload, Sparkles, ChevronDown, ChevronUp, Info, Clock, Database, ArrowRight } from "lucide-react";

import API_BASE from "../api.js";
const API = API_BASE;

async function apiPredict(file, useRag = false) {
  const form = new FormData();
  form.append("file", file);
  const url = `${API}/predict?use_rag=${useRag}`;
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

async function apiPolicy() {
  const res = await fetch(`${API}/policy`);
  if (!res.ok) throw new Error("policy unavailable");
  return res.json();
}

/* ------------------------------------------------------------------ */
export default function PredictPage() {
  const [file, setFile]         = useState(null);
  const [preview, setPreview]   = useState(null);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);
  const [history, setHistory]   = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [policy, setPolicy]     = useState(null);
  const [useRag, setUseRag]     = useState(false);
  const [ragAvailable, setRagAvailable] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    apiPolicy().then(setPolicy).catch(() => {});
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => setRagAvailable(h.rag_available ?? false))
      .catch(() => {});
  }, []);

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please upload a valid image file (PNG, JPG, WEBP).");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, [preview]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handlePredict = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiPredict(file, useRag);
      setResult(data);
      setHistory((prev) =>
        [{ ...data, preview, fileName: file.name }, ...prev].slice(0, 20)
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [file, preview, useRag]);

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col lg:flex-row gap-8">

      {/* ── Left column ── */}
      <div className="flex-1 flex flex-col gap-5 min-w-0">

        {/* Upload */}
        <UploadArea
          preview={preview}
          dragOver={dragOver}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClickBrowse={() => inputRef.current?.click()}
        />
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          aria-label="Choose image file"
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {/* RAG toggle */}
        <div className="flex items-center justify-between glass px-4 py-3">
          <div className="flex items-center gap-2.5">
            <Database size={14} className="text-purple-400 shrink-0" aria-hidden="true" />
            <div>
              <p className="text-sm font-medium leading-tight">RAG Refinement</p>
              <p className="text-xs text-[var(--color-text-dim)] leading-tight mt-0.5">
                {ragAvailable
                  ? "Retrieval-augmented caption (Stage 7 corpus)"
                  : "Corpus unavailable — toggle disabled"}
              </p>
            </div>
          </div>
          <button
            role="switch"
            aria-checked={useRag}
            aria-label="Toggle RAG retrieval refinement"
            disabled={!ragAvailable}
            onClick={() => setUseRag((v) => !v)}
            className={`relative w-11 h-6 rounded-full transition-colors duration-200 shrink-0
                        focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]
                        disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer
                        ${useRag ? "bg-purple-500" : "bg-[var(--color-surface-3)]"}`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow
                          transition-transform duration-200
                          ${useRag ? "translate-x-5" : "translate-x-0"}`}
            />
          </button>
        </div>

        {/* Generate */}
        <button
          onClick={handlePredict}
          disabled={!file || loading}
          aria-label={loading ? "Generating caption, please wait" : "Generate caption"}
          className="w-full py-3.5 rounded-xl font-semibold text-white text-base
                     bg-gradient-to-r from-[var(--color-accent)] to-purple-500
                     hover:from-indigo-500 hover:to-purple-400
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all duration-300 cursor-pointer
                     shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <Spinner /> Generating caption…
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <Sparkles size={16} aria-hidden="true" /> Generate Caption
            </span>
          )}
        </button>

        {/* Error */}
        {error && (
          <div
            role="alert"
            className="glass px-4 py-3 border-red-500/40! text-red-400 text-sm animate-fade-in-up flex items-center gap-2"
          >
            <Info size={14} aria-hidden="true" /> {error}
          </div>
        )}

        {/* Result */}
        {result && <ResultCard result={result} policy={policy} />}

        {/* Policy info strip */}
        {policy && !result && <PolicyStrip policy={policy} />}
      </div>

      {/* ── Right column ── */}
      <aside className="w-full lg:w-80 shrink-0" aria-label="Prediction history">
        <HistoryPanel history={history} />
      </aside>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Upload area                                                        */
/* ------------------------------------------------------------------ */
function UploadArea({ preview, dragOver, onDragOver, onDragLeave, onDrop, onClickBrowse }) {
  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload image — click to browse or drag and drop"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClickBrowse}
      onKeyDown={(e) => e.key === "Enter" && onClickBrowse()}
      className={`glass cursor-pointer transition-all duration-300 overflow-hidden
                  ${dragOver ? "border-[var(--color-accent)]! shadow-lg shadow-indigo-500/20 scale-[1.01]" : ""}
                  ${preview ? "p-0" : "p-10"}`}
    >
      {preview ? (
        <div className="relative group">
          <img
            src={preview}
            alt="Uploaded image preview"
            className="w-full max-h-[400px] object-contain bg-black/30"
          />
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100
                          flex items-center justify-center transition-opacity duration-300">
            <span className="text-sm font-medium text-white/90 flex items-center gap-2">
              <Upload size={16} aria-hidden="true" /> Click to change image
            </span>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 text-center">
          <div className="w-16 h-16 rounded-2xl bg-[var(--color-surface-3)] flex items-center justify-center">
            <Upload size={28} className="text-[var(--color-text-dim)]" aria-hidden="true" />
          </div>
          <p className="text-sm font-medium">
            Drop an image here or{" "}
            <span className="text-[var(--color-accent-2)] underline underline-offset-2">browse</span>
          </p>
          <p className="text-xs text-[var(--color-text-dim)]">PNG, JPG, WEBP — any size</p>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Result card                                                        */
/* ------------------------------------------------------------------ */
function ResultCard({ result, policy }) {
  const [whyOpen, setWhyOpen] = useState(false);
  const [ragOpen, setRagOpen] = useState(true);
  const isAuto = result.route === "AUTO";
  const pct = Math.round(result.confidence * 100);
  const threshold = policy?.threshold ?? 0.5115;
  const thresholdPct = Math.round(threshold * 100);

  const barColor = isAuto
    ? "bg-gradient-to-r from-emerald-500 to-teal-400"
    : "bg-gradient-to-r from-amber-500 to-orange-400";

  const badgeClass = isAuto
    ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
    : "bg-amber-500/15 text-amber-400 border border-amber-500/30";

  return (
    <article
      className="glass p-6 space-y-5 animate-fade-in-up"
      aria-label="Prediction result"
    >
      {/* Demo badge */}
      {result.demo_mode && (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/20">
          <span className="text-xs">⚡</span>
          <span className="text-xs text-[var(--color-accent-2)]">
            Demo mode — sample COCO caption (model not loaded locally)
          </span>
        </div>
      )}

      {/* Caption */}
      <div>
        <p className="section-label mb-1.5">Generated Caption</p>
        <p className="text-lg font-medium leading-relaxed">"{result.caption}"</p>
      </div>

      {/* Confidence */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <p className="section-label">Model Confidence</p>
          <span className="text-sm font-semibold tabular-nums">{pct}%</span>
        </div>
        <div className="confidence-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
          <div className={`confidence-fill ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
      </div>

      {/* Route badge */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${badgeClass}`}>
          {isAuto ? "✓ AUTO-PUBLISH" : "⚑ HUMAN REVIEW"}
        </span>
        <span className="text-xs text-[var(--color-text-dim)] flex-1 min-w-0">{result.rationale}</span>
      </div>

      {/* RAG panel — shown when RAG was requested */}
      {(result.rag_caption !== undefined && result.rag_caption !== null) && (
        <div className="border-t border-[var(--color-border)] pt-4">
          <button
            onClick={() => setRagOpen((o) => !o)}
            aria-expanded={ragOpen}
            aria-controls="rag-panel"
            className="flex items-center gap-2 text-xs font-medium text-purple-400
                       hover:text-purple-300 transition-colors cursor-pointer w-full"
          >
            <Database size={13} aria-hidden="true" />
            RAG Refinement
            <span className={`ml-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold
              ${result.rag_used
                ? "bg-purple-500/20 text-purple-300"
                : "bg-[var(--color-surface-3)] text-[var(--color-text-dim)]"}`}>
              {result.rag_used ? "applied" : "not triggered"}
            </span>
            <span className="ml-auto">
              {ragOpen ? <ChevronUp size={13} aria-hidden="true" /> : <ChevronDown size={13} aria-hidden="true" />}
            </span>
          </button>

          {ragOpen && (
            <div
              id="rag-panel"
              className="mt-3 p-4 rounded-xl bg-purple-500/5 border border-purple-500/20 space-y-3 animate-fade-in-up"
              role="region"
              aria-label="RAG refinement details"
            >
              {result.rag_used ? (
                <>
                  {/* Before → After */}
                  <div className="space-y-2">
                    <div className="p-3 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)]">
                      <p className="section-label mb-1">Base caption</p>
                      <p className="text-xs text-[var(--color-text-dim)]">"{result.caption}"</p>
                    </div>
                    <div className="flex justify-center">
                      <ArrowRight size={14} className="text-purple-400" aria-hidden="true" />
                    </div>
                    <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/30">
                      <p className="section-label mb-1 text-purple-400">RAG caption</p>
                      <p className="text-xs text-purple-200">"{result.rag_caption}"</p>
                    </div>
                  </div>

                  {/* Similarity score */}
                  {result.retrieval_sim !== null && (
                    <p className="text-[11px] text-[var(--color-text-dim)]">
                      Retrieval similarity:{" "}
                      <span className="font-semibold text-purple-300">
                        {(result.retrieval_sim * 100).toFixed(1)}%
                      </span>
                      {" "}(Jaccard) · gate: sim ≥ 60% &amp; conf &lt; 40%
                    </p>
                  )}

                  {/* Retrieved captions list */}
                  {result.retrieved_captions?.length > 0 && (
                    <div>
                      <p className="section-label mb-2">Top retrieved captions</p>
                      <ol className="space-y-1">
                        {result.retrieved_captions.map((c, i) => (
                          <li key={i} className="flex gap-2 text-[11px] text-[var(--color-text-dim)]">
                            <span className="text-purple-400 shrink-0 tabular-nums">{i + 1}.</span>
                            <span className="italic">"{c}"</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}
                </>
              ) : (
                <div className="space-y-2">
                  <p className="text-xs text-[var(--color-text-dim)] leading-relaxed">
                    RAG gate not triggered. Retrieval is only applied when both conditions are met:
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div className="p-2 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)] text-center">
                      <p className="text-[var(--color-text-dim)]">Retrieval sim</p>
                      <p className="font-semibold mt-0.5">
                        {result.retrieval_sim !== null
                          ? `${(result.retrieval_sim * 100).toFixed(1)}% / 60% needed`
                          : "N/A"}
                      </p>
                    </div>
                    <div className="p-2 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)] text-center">
                      <p className="text-[var(--color-text-dim)]">Confidence</p>
                      <p className="font-semibold mt-0.5">
                        {pct}% / must be &lt;40%
                      </p>
                    </div>
                  </div>
                  {result.retrieved_captions?.length > 0 && (
                    <details className="text-[11px] text-[var(--color-text-dim)]">
                      <summary className="cursor-pointer hover:text-[var(--color-text)] transition-colors">
                        Show retrieved captions anyway
                      </summary>
                      <ol className="mt-2 space-y-1 pl-2">
                        {result.retrieved_captions.map((c, i) => (
                          <li key={i} className="flex gap-2">
                            <span className="text-purple-400 shrink-0">{i + 1}.</span>
                            <span className="italic">"{c}"</span>
                          </li>
                        ))}
                      </ol>
                    </details>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Why this decision */}
      <div className="border-t border-[var(--color-border)] pt-4">
        <button
          onClick={() => setWhyOpen((o) => !o)}
          aria-expanded={whyOpen}
          aria-controls="why-panel"
          className="flex items-center gap-2 text-xs font-medium text-[var(--color-accent-2)]
                     hover:text-white transition-colors cursor-pointer"
        >
          <Info size={13} aria-hidden="true" />
          Why this decision?
          {whyOpen ? <ChevronUp size={13} aria-hidden="true" /> : <ChevronDown size={13} aria-hidden="true" />}
        </button>

        {whyOpen && (
          <div
            id="why-panel"
            className="mt-3 p-4 rounded-xl bg-[var(--color-surface-3)]/60 space-y-3 text-xs text-[var(--color-text-dim)] animate-fade-in-up"
            role="region"
            aria-label="Decision explanation"
          >
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)]">
                <p className="section-label mb-1">Your confidence</p>
                <p className="text-xl font-bold text-[var(--color-text)]">{pct}%</p>
              </div>
              <div className="p-3 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)]">
                <p className="section-label mb-1">Policy threshold</p>
                <p className="text-xl font-bold text-[var(--color-text)]">{thresholdPct}%</p>
              </div>
            </div>

            <p className="leading-relaxed">
              Since {pct}%{" "}
              <span className={`font-semibold ${isAuto ? "text-emerald-400" : "text-amber-400"}`}>
                {isAuto ? "≥" : "<"}
              </span>{" "}
              {thresholdPct}%, this caption was routed to{" "}
              <span className={`font-semibold ${isAuto ? "text-emerald-400" : "text-amber-400"}`}>
                {isAuto ? "AUTO-PUBLISH" : "HUMAN REVIEW"}.
              </span>
            </p>

            <div className="grid grid-cols-2 gap-2 pt-1">
              <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-center">
                <p className="text-emerald-400 font-semibold">BLEU-4 0.281</p>
                <p className="text-[10px] mt-0.5">auto-published avg</p>
              </div>
              <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-center">
                <p className="text-amber-400 font-semibold">BLEU-4 0.163</p>
                <p className="text-[10px] mt-0.5">human-reviewed avg</p>
              </div>
            </div>

            <p className="text-[10px] leading-relaxed opacity-70">
              Threshold selected for 50/50 auto/human split across 2,500 COCO test images.
              Beam search (k=3, λ=0.6) · MobileNetV2 + LSTM · AdamW lr=5e-5
            </p>
          </div>
        )}
      </div>

      {/* Meta */}
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-[var(--color-text-dim)] pt-1">
        <span>ID: {result.request_id}</span>
        <span>Decode: {result.decode_method}</span>
        <span>{new Date(result.timestamp).toLocaleTimeString()}</span>
      </div>
    </article>
  );
}

/* ------------------------------------------------------------------ */
/*  Policy strip (shown before first prediction)                      */
/* ------------------------------------------------------------------ */
function PolicyStrip({ policy }) {
  const thresholdPct = Math.round((policy?.threshold ?? 0.5115) * 100);
  return (
    <div className="glass px-4 py-3 flex flex-wrap items-center gap-4 text-xs text-[var(--color-text-dim)]">
      <span className="flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)]" />
        Beam search k={policy?.beam_size ?? 3} λ={policy?.length_penalty ?? 0.6}
      </span>
      <span className="flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
        Auto-publish threshold {thresholdPct}%
      </span>
      <span className="flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-text-dim)]" />
        Max caption length {policy?.max_len ?? 30} tokens
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  History panel                                                      */
/* ------------------------------------------------------------------ */
function HistoryPanel({ history }) {
  return (
    <div className="glass p-5 h-fit lg:sticky lg:top-24">
      <h2 className="text-sm font-semibold mb-4 flex items-center gap-2">
        <Clock size={14} className="text-[var(--color-text-dim)]" aria-hidden="true" />
        History
        {history.length > 0 && (
          <span className="ml-auto text-xs text-[var(--color-text-dim)] tabular-nums">{history.length}</span>
        )}
      </h2>

      {history.length === 0 ? (
        <p className="text-xs text-[var(--color-text-dim)] text-center py-8">No predictions yet</p>
      ) : (
        <ul className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
          {history.map((h, i) => (
            <li key={i}>
              <div className="p-3 rounded-xl bg-[var(--color-surface-3)]/50 hover:bg-[var(--color-surface-3)]
                              transition-colors duration-200 flex items-start gap-2.5">
                {h.preview && (
                  <img src={h.preview} alt="" className="w-10 h-10 rounded-lg object-cover shrink-0" aria-hidden="true" />
                )}
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-[var(--color-text)] line-clamp-2 leading-relaxed">"{h.caption}"</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full
                      ${h.route === "AUTO"
                        ? "bg-emerald-500/15 text-emerald-400"
                        : "bg-amber-500/15 text-amber-400"}`}>
                      {h.route === "AUTO" ? "AUTO" : "REVIEW"}
                    </span>
                    <span className="text-[10px] text-[var(--color-text-dim)] tabular-nums">
                      {Math.round(h.confidence * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Spinner                                                            */
/* ------------------------------------------------------------------ */
function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
