import { useState } from "react";
import Nav from "./components/Nav";
import PredictPage from "./pages/PredictPage";
import BenchmarkPage from "./pages/BenchmarkPage";
import ErrorAnalysisPage from "./pages/ErrorAnalysisPage";
import ModelCardPage from "./pages/ModelCardPage";

export default function App() {
  const [page, setPage] = useState("predict");

  return (
    <div className="min-h-screen flex flex-col">
      <Nav page={page} setPage={setPage} />

      <main className="flex-1" id="main-content">
        {page === "predict"   && <PredictPage />}
        {page === "benchmark" && <BenchmarkPage />}
        {page === "error"     && <ErrorAnalysisPage />}
        {page === "model"     && <ModelCardPage />}
      </main>

      <footer className="text-center text-xs text-[var(--color-text-dim)] py-6 border-t border-[var(--color-border)] mt-auto">
        AccessOps COCO AI · Image Captioning Pipeline · v1.0
      </footer>
    </div>
  );
}
