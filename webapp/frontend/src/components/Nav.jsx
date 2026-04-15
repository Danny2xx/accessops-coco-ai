import { useState, useEffect } from "react";
import { Sparkles, BarChart2, AlertCircle, BookOpen, Menu, X } from "lucide-react";

const PAGES = [
  { id: "predict",   label: "Predict",        icon: Sparkles     },
  { id: "benchmark", label: "Benchmark",       icon: BarChart2    },
  { id: "error",     label: "Error Analysis",  icon: AlertCircle  },
  { id: "model",     label: "Model Card",      icon: BookOpen     },
];

export default function Nav({ page, setPage }) {
  const [status, setStatus] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus({ status: "offline" }));
  }, []);

  const st = status?.status ?? "…";
  const dotColor =
    st === "ok"       ? "#10b981" :
    st === "demo"     ? "#818cf8" :
    st === "degraded" ? "#f59e0b" : "#ef4444";
  const dotLabel =
    st === "ok"       ? "model loaded" :
    st === "demo"     ? "demo mode" :
    st === "degraded" ? "degraded" :
    st === "offline"  ? "offline" : "…";

  function navigate(id) {
    setPage(id);
    setMenuOpen(false);
  }

  return (
    <header
      className="border-b border-[var(--color-border)] bg-[var(--color-surface)]/90 backdrop-blur-md sticky top-0 z-50"
      role="banner"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">

        {/* Logo */}
        <div className="flex items-center gap-3 shrink-0">
          <div
            className="w-9 h-9 rounded-lg bg-gradient-to-br from-[var(--color-accent)] to-cyan-400
                       flex items-center justify-center text-xs font-bold shadow-lg shadow-sky-500/30 select-none"
            aria-hidden="true"
          >
            AC
          </div>
          <div className="hidden sm:block leading-tight">
            <p className="text-sm font-semibold">AccessOps COCO AI</p>
            <p className="text-[11px] text-[var(--color-text-dim)]">Image Captioning Pipeline</p>
          </div>
        </div>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-1" role="navigation" aria-label="Main navigation">
          {PAGES.map(({ id, label, icon: Icon }) => {
            const active = page === id;
            return (
              <button
                key={id}
                onClick={() => navigate(id)}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                            transition-all duration-200 cursor-pointer
                            ${active
                              ? "bg-[var(--color-accent)]/15 text-[var(--color-accent-2)]"
                              : "text-[var(--color-text-dim)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-3)]"
                            }`}
              >
                <Icon size={14} aria-hidden="true" />
                {label}
              </button>
            );
          })}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Status */}
          <div
            className="hidden sm:flex items-center gap-1.5 text-xs text-[var(--color-text-dim)]"
            role="status"
            aria-live="polite"
            aria-label={`API status: ${dotLabel}`}
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: dotColor, boxShadow: `0 0 6px ${dotColor}` }}
            />
            {dotLabel}
          </div>

          {/* Hamburger */}
          <button
            className="md:hidden p-2 rounded-lg text-[var(--color-text-dim)]
                       hover:text-[var(--color-text)] hover:bg-[var(--color-surface-3)] transition-colors cursor-pointer"
            onClick={() => setMenuOpen((o) => !o)}
            aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={menuOpen}
            aria-controls="mobile-nav"
          >
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div
          id="mobile-nav"
          className="md:hidden border-t border-[var(--color-border)] bg-[var(--color-surface-2)] px-4 py-3 space-y-1"
          role="navigation"
          aria-label="Mobile navigation"
        >
          {PAGES.map(({ id, label, icon: Icon }) => {
            const active = page === id;
            return (
              <button
                key={id}
                onClick={() => navigate(id)}
                aria-current={active ? "page" : undefined}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                            transition-colors cursor-pointer
                            ${active
                              ? "bg-[var(--color-accent)]/15 text-[var(--color-accent-2)]"
                              : "text-[var(--color-text-dim)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-3)]"
                            }`}
              >
                <Icon size={16} aria-hidden="true" />
                {label}
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
}
