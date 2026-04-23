import { useState, useRef } from "react";

const API = "http://localhost:8000";

const MODELS = [
  { id: "xgb",      label: "XGBoost",         sub: "Fast · CPU · k-mer features" },
  { id: "dnabert",  label: "DNABERT-2",        sub: "Deep · GPU · sequence model" },
  { id: "ensemble", label: "Ensemble",          sub: "XGB 40% + DNABERT 60%" },
];

const EXAMPLE = {
  chrom: "1", position: "925952", ref: "G", alt: "A",
  ref_seq: "AGCTGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
};

function ProbBar({ prob, prediction }) {
  const pct = Math.round(prob * 100);
  const isPath = prediction === "Pathogenic";
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontFamily: "'Space Mono', monospace", fontSize: 11, letterSpacing: 1 }}>
        <span style={{ color: "#4ade80" }}>BENIGN</span>
        <span style={{ color: "#f87171" }}>PATHOGENIC</span>
      </div>
      <div style={{ height: 8, background: "#1e2433", borderRadius: 4, overflow: "hidden", position: "relative" }}>
        <div style={{
          position: "absolute", left: 0, top: 0, height: "100%",
          width: `${pct}%`,
          background: isPath
            ? `linear-gradient(90deg, #f87171, #ef4444)`
            : `linear-gradient(90deg, #4ade80, #22c55e)`,
          borderRadius: 4,
          transition: "width 0.8s cubic-bezier(0.16,1,0.3,1)",
          boxShadow: isPath ? "0 0 12px #ef444466" : "0 0 12px #22c55e66",
        }} />
        <div style={{
          position: "absolute", left: `${pct}%`, top: "50%",
          transform: "translate(-50%,-50%)",
          width: 12, height: 12,
          background: "#fff",
          borderRadius: "50%",
          boxShadow: "0 0 8px rgba(255,255,255,0.6)",
          transition: "left 0.8s cubic-bezier(0.16,1,0.3,1)",
        }} />
      </div>
      <div style={{ textAlign: "center", marginTop: 8, fontFamily: "'Space Mono', monospace", fontSize: 28, fontWeight: 700,
        color: isPath ? "#f87171" : "#4ade80",
        textShadow: isPath ? "0 0 20px #ef444488" : "0 0 20px #22c55e88",
      }}>
        {pct}%
      </div>
    </div>
  );
}

function ResultCard({ result, onClose }) {
  const isPath = result.prediction === "Pathogenic";
  return (
    <div style={{
      background: "linear-gradient(135deg, #0d1117 0%, #161b27 100%)",
      border: `1px solid ${isPath ? "#ef444433" : "#22c55e33"}`,
      borderRadius: 16,
      padding: 28,
      position: "relative",
      boxShadow: isPath ? "0 8px 40px #ef444422" : "0 8px 40px #22c55e22",
      animation: "slideUp 0.4s cubic-bezier(0.16,1,0.3,1)",
    }}>
      <button onClick={onClose} style={{
        position: "absolute", top: 16, right: 16,
        background: "none", border: "none", color: "#4b5563",
        cursor: "pointer", fontSize: 18, lineHeight: 1,
      }}>✕</button>

      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{
          width: 10, height: 10, borderRadius: "50%",
          background: isPath ? "#ef4444" : "#22c55e",
          boxShadow: isPath ? "0 0 10px #ef4444" : "0 0 10px #22c55e",
        }} />
        <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 11, letterSpacing: 2, color: "#6b7280", textTransform: "uppercase" }}>
          {result.model}
        </span>
        <span style={{ marginLeft: "auto", fontFamily: "'Space Mono', monospace", fontSize: 10, color: "#374151" }}>
          {result.latency_ms}ms
        </span>
      </div>

      <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 32, fontWeight: 800,
        color: isPath ? "#f87171" : "#4ade80",
        letterSpacing: -1, marginBottom: 4,
      }}>
        {result.prediction}
      </div>
      <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
        Confidence: <span style={{ color: "#9ca3af" }}>{result.confidence}</span>
      </div>

      <ProbBar prob={result.probability} prediction={result.prediction} />

      <div style={{ marginTop: 20, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {[
          ["Chrom", result.chrom],
          ["Position", result.position.toLocaleString()],
          ["REF", result.ref],
          ["ALT", result.alt],
        ].map(([k, v]) => (
          <div key={k} style={{
            background: "#0a0e18", borderRadius: 8, padding: "8px 12px",
            border: "1px solid #1e2433",
          }}>
            <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 9, color: "#4b5563", letterSpacing: 1, marginBottom: 2 }}>{k}</div>
            <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 13, color: "#e5e7eb" }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [form, setForm]       = useState({ chrom: "", position: "", ref: "", alt: "", ref_seq: "", alt_seq: "" });
  const [model, setModel]     = useState("ensemble");
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);
  const formRef               = useRef(null);

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }));

  const loadExample = () => {
    setForm({ ...EXAMPLE, alt_seq: "" });
    setResult(null);
    setError(null);
  };

  const predict = async () => {
    const { chrom, position, ref, alt } = form;
    if (!chrom || !position || !ref || !alt) {
      setError("chrom, position, ref, alt are required.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const body = {
        chrom,
        position : parseInt(position),
        ref      : ref.toUpperCase(),
        alt      : alt.toUpperCase(),
        ref_seq  : form.ref_seq || null,
        alt_seq  : form.alt_seq || null,
      };

      const res  = await fetch(`${API}/predict/${model}`, {
        method  : "POST",
        headers : { "Content-Type": "application/json" },
        body    : JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const inputStyle = {
    width: "100%", background: "#0a0e18",
    border: "1px solid #1e2433", borderRadius: 8,
    padding: "10px 14px", color: "#e5e7eb",
    fontFamily: "'Space Mono', monospace", fontSize: 13,
    outline: "none", boxSizing: "border-box",
    transition: "border-color 0.2s",
  };

  const labelStyle = {
    fontFamily: "'Space Mono', monospace", fontSize: 10,
    letterSpacing: 1.5, color: "#4b5563",
    textTransform: "uppercase", marginBottom: 6, display: "block",
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #070b12; }
        input:focus { border-color: #3b82f6 !important; }
        input::placeholder { color: #374151; }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #070b12; }
        ::-webkit-scrollbar-thumb { background: #1e2433; border-radius: 2px; }
      `}</style>

      <div style={{
        minHeight: "100vh", background: "#070b12",
        display: "flex", alignItems: "flex-start",
        justifyContent: "center",
        padding: "40px 16px",
      }}>
        <div style={{ width: "100%", maxWidth: 520 }}>

          {/* Header */}
          <div style={{ marginBottom: 36 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <div style={{
                width: 8, height: 8, borderRadius: "50%",
                background: "#3b82f6",
                boxShadow: "0 0 12px #3b82f6",
                animation: "pulse 2s infinite",
              }} />
              <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 10, letterSpacing: 2, color: "#3b5280", textTransform: "uppercase" }}>
                Splice Variant Classifier
              </span>
            </div>
            <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: 38, fontWeight: 800, color: "#f1f5f9", letterSpacing: -1.5, lineHeight: 1.1 }}>
              Pathogenicity<br />
              <span style={{ color: "#3b82f6" }}>Predictor</span>
            </h1>
            <p style={{ marginTop: 10, fontFamily: "'Space Mono', monospace", fontSize: 11, color: "#374151", lineHeight: 1.7 }}>
              XGBoost · DNABERT-2 · Ensemble
            </p>
          </div>

          {/* Model selector */}
          <div style={{ marginBottom: 24 }}>
            <div style={labelStyle}>Model</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              {MODELS.map(m => (
                <button key={m.id} onClick={() => setModel(m.id)} style={{
                  background: model === m.id ? "#111827" : "#0a0e18",
                  border: `1px solid ${model === m.id ? "#3b82f6" : "#1e2433"}`,
                  borderRadius: 10, padding: "10px 8px",
                  cursor: "pointer", transition: "all 0.2s",
                  boxShadow: model === m.id ? "0 0 16px #3b82f633" : "none",
                }}>
                  <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 12, color: model === m.id ? "#93c5fd" : "#6b7280" }}>
                    {m.label}
                  </div>
                  <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 9, color: "#374151", marginTop: 3, lineHeight: 1.4 }}>
                    {m.sub}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Form */}
          <div ref={formRef} style={{ display: "flex", flexDirection: "column", gap: 14 }}>

            {/* Chrom + Position */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 10 }}>
              <div>
                <label style={labelStyle}>Chrom</label>
                <input style={inputStyle} value={form.chrom} onChange={set("chrom")} placeholder="1" />
              </div>
              <div>
                <label style={labelStyle}>Position</label>
                <input style={inputStyle} type="number" value={form.position} onChange={set("position")} placeholder="925952" />
              </div>
            </div>

            {/* REF + ALT */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div>
                <label style={labelStyle}>REF Allele</label>
                <input style={inputStyle} value={form.ref} onChange={set("ref")} placeholder="G" maxLength={50} />
              </div>
              <div>
                <label style={labelStyle}>ALT Allele</label>
                <input style={inputStyle} value={form.alt} onChange={set("alt")} placeholder="A" maxLength={50} />
              </div>
            </div>

            {/* REF SEQ */}
            <div>
              <label style={labelStyle}>REF Sequence (optional, ±200bp window)</label>
              <input style={{ ...inputStyle, fontFamily: "'Space Mono', monospace", fontSize: 11 }}
                value={form.ref_seq} onChange={set("ref_seq")}
                placeholder="Genomic window around variant..." />
            </div>

            {/* ALT SEQ */}
            <div>
              <label style={labelStyle}>ALT Sequence (optional, auto-built if omitted)</label>
              <input style={{ ...inputStyle, fontFamily: "'Space Mono', monospace", fontSize: 11 }}
                value={form.alt_seq} onChange={set("alt_seq")}
                placeholder="Leave blank to auto-compute..." />
            </div>

            {/* Buttons */}
            <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
              <button onClick={predict} disabled={loading} style={{
                flex: 1, padding: "14px 0",
                background: loading ? "#1e3a5f" : "linear-gradient(135deg, #1d4ed8, #2563eb)",
                border: "none", borderRadius: 10,
                color: "#fff", fontFamily: "'Syne', sans-serif",
                fontWeight: 700, fontSize: 14, letterSpacing: 0.5,
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.2s",
                boxShadow: loading ? "none" : "0 4px 20px #2563eb44",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
              }}>
                {loading ? (
                  <>
                    <div style={{ width: 14, height: 14, border: "2px solid #93c5fd33", borderTopColor: "#93c5fd", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                    Predicting...
                  </>
                ) : "Predict →"}
              </button>
              <button onClick={loadExample} style={{
                padding: "14px 18px",
                background: "#0a0e18", border: "1px solid #1e2433",
                borderRadius: 10, color: "#4b5563",
                fontFamily: "'Space Mono', monospace", fontSize: 11,
                cursor: "pointer", transition: "all 0.2s",
              }}>
                Example
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              marginTop: 16, padding: "12px 16px",
              background: "#1a0a0a", border: "1px solid #ef444433",
              borderRadius: 10, color: "#f87171",
              fontFamily: "'Space Mono', monospace", fontSize: 11,
            }}>
              ⚠ {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div style={{ marginTop: 20 }}>
              <ResultCard result={result} onClose={() => setResult(null)} />
            </div>
          )}

          {/* Footer */}
          <div style={{ marginTop: 40, textAlign: "center", fontFamily: "'Space Mono', monospace", fontSize: 10, color: "#1e2433", letterSpacing: 1 }}>
            API: {API} · XGBoost + DNABERT-2-117M
          </div>
        </div>
      </div>
    </>
  );
}