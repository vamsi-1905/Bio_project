import { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000";

const MODES = [
  { id: "xgb",          label: "XGBoost",       sub: "k-mer · CPU · fast" },
  { id: "dnabert",      label: "DNABERT-2",      sub: "sequence · GPU" },
  { id: "ensemble",     label: "Ensemble",       sub: "XGB 40% + DNA 60%" },
  { id: "splice_sites", label: "Splice Sites",   sub: "PWM · cryptic · donor/acceptor" },
];

const EX = {
  chrom:"1", position:"925952", ref:"G", alt:"A",
  ref_seq:"AGCTGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
};

/* ─── theme ─────────────────────────────────────────────────────── */
const T = {
  bg:      "#06080f",
  panel:   "#0c0f1a",
  border:  "#151929",
  border2: "#1e2540",
  text:    "#cdd6f4",
  muted:   "#45475a",
  muted2:  "#6c7086",
  accent:  "#89b4fa",
  red:     "#f38ba8",
  green:   "#a6e3a1",
  yellow:  "#f9e2af",
  orange:  "#fab387",
  teal:    "#94e2d5",
  mauve:   "#cba6f7",
};

/* ─── helpers ───────────────────────────────────────────────────── */
const mono = { fontFamily: "'JetBrains Mono', 'Fira Code', monospace" };
const sans = { fontFamily: "'Outfit', sans-serif" };

const SIG_COLOR = (s = "") =>
  s.startsWith("Strong") ? T.red
  : s.startsWith("Moderate") ? T.orange
  : s.startsWith("Weak") || s.startsWith("Low") ? T.yellow
  : T.green;

const MUT_COLOR = (t = "") =>
  t === "SNV" ? T.accent : t === "Deletion" ? T.red : t === "Insertion" ? T.orange : T.mauve;

/* ─── Animated number ────────────────────────────────────────────── */
function AnimNum({ value, suffix = "%", decimals = 0 }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const target = parseFloat(value);
    const step = target / 40;
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setDisplay(target); clearInterval(timer); }
      else setDisplay(start);
    }, 16);
    return () => clearInterval(timer);
  }, [value]);
  return <>{display.toFixed(decimals)}{suffix}</>;
}

/* ─── Gauge ──────────────────────────────────────────────────────── */
function Gauge({ prob, isPath, thresh = 0.5 }) {
  const pct    = Math.round(prob * 100);
  const tpct   = Math.round(thresh * 100);
  const color  = isPath ? T.red : T.green;
  const glow   = isPath ? "#f38ba855" : "#a6e3a155";

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ ...mono, fontSize: 10, letterSpacing: 2, color: T.green }}>BENIGN</span>
        <span style={{ ...mono, fontSize: 10, letterSpacing: 2, color: T.red }}>PATHOGENIC</span>
      </div>
      <div style={{ position: "relative", height: 10, background: T.border, borderRadius: 5, overflow: "visible" }}>
        <div style={{
          position: "absolute", left: 0, top: 0, height: "100%",
          width: `${pct}%`, borderRadius: 5,
          background: `linear-gradient(90deg, ${color}88, ${color})`,
          boxShadow: `0 0 16px ${glow}`,
          transition: "width 0.9s cubic-bezier(0.16,1,0.3,1)",
        }} />
        {/* threshold marker */}
        <div style={{
          position: "absolute", left: `${tpct}%`, top: -5,
          transform: "translateX(-50%)",
          width: 2, height: 20, background: T.yellow,
          borderRadius: 1, zIndex: 3,
        }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: 10 }}>
        <span style={{ ...mono, fontSize: 9, color: T.yellow }}>▲ threshold {tpct}%</span>
        <div style={{
          ...mono, fontSize: 42, fontWeight: 700,
          color, textShadow: `0 0 30px ${glow}`,
          lineHeight: 1,
        }}>
          <AnimNum value={pct} />
        </div>
      </div>
    </div>
  );
}

/* ─── Mutation badge ─────────────────────────────────────────────── */
function MutBadge({ type }) {
  return (
    <span style={{
      ...mono, fontSize: 9, letterSpacing: 1,
      padding: "3px 8px", borderRadius: 4,
      background: `${MUT_COLOR(type)}22`,
      border: `1px solid ${MUT_COLOR(type)}44`,
      color: MUT_COLOR(type),
      textTransform: "uppercase",
    }}>{type}</span>
  );
}

/* ─── Prediction card ────────────────────────────────────────────── */
function PredCard({ r, onClose }) {
  const isPath = r.prediction === "Pathogenic";
  const color  = isPath ? T.red : T.green;
  const glow   = isPath ? "#f38ba822" : "#a6e3a122";

  return (
    <div style={{
      background: `linear-gradient(135deg, ${T.panel} 0%, #0f1323 100%)`,
      border: `1px solid ${color}33`, borderRadius: 16, padding: 24,
      position: "relative", boxShadow: `0 8px 48px ${glow}`,
      animation: "slideUp 0.45s cubic-bezier(0.16,1,0.3,1)",
    }}>
      <button onClick={onClose} style={{
        position: "absolute", top: 16, right: 16,
        background: "none", border: "none", color: T.muted,
        cursor: "pointer", fontSize: 18, lineHeight: 1,
        transition: "color 0.2s",
      }}>✕</button>

      {/* header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
        <div style={{
          width: 10, height: 10, borderRadius: "50%",
          background: color, boxShadow: `0 0 12px ${color}`,
          animation: "pulse 2s infinite",
        }} />
        <span style={{ ...mono, fontSize: 10, letterSpacing: 2, color: T.muted2, textTransform: "uppercase" }}>
          {r.model}
        </span>
        <span style={{ marginLeft: "auto", ...mono, fontSize: 10, color: T.border2 }}>
          {r.latency_ms}ms
        </span>
      </div>

      <div style={{ ...sans, fontSize: 36, fontWeight: 700, color, letterSpacing: -1, marginBottom: 4 }}>
        {r.prediction}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 4 }}>
        <MutBadge type={r.mutation_type} />
        <span style={{ ...mono, fontSize: 10, color: T.muted2 }}>
          Confidence: <span style={{ color: T.text }}>{r.confidence}</span>
        </span>
      </div>

      <Gauge prob={r.probability} isPath={isPath} thresh={r.threshold_used} />

      {/* mechanism reasoning */}
      {r.mechanism && (
        <div style={{
          marginTop: 18, padding: "12px 14px",
          background: "#0a0d1a", borderRadius: 10,
          border: `1px solid ${T.border2}`,
          borderLeft: `3px solid ${color}`,
        }}>
          <div style={{ ...mono, fontSize: 9, letterSpacing: 1.5, color: T.muted2, marginBottom: 6 }}>
            MOLECULAR MECHANISM
          </div>
          <div style={{ ...mono, fontSize: 11, color: T.text, lineHeight: 1.7 }}>
            {r.mechanism}
          </div>
        </div>
      )}

      {/* variant info grid */}
      <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 7 }}>
        {[["CHROM", r.chrom], ["POSITION", r.position?.toLocaleString()],
          ["REF", r.ref], ["ALT", r.alt]].map(([k, v]) => (
          <div key={k} style={{
            background: T.bg, borderRadius: 8, padding: "8px 12px",
            border: `1px solid ${T.border}`,
          }}>
            <div style={{ ...mono, fontSize: 9, color: T.muted, letterSpacing: 1, marginBottom: 3 }}>{k}</div>
            <div style={{ ...mono, fontSize: 13, color: T.text }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Site row ───────────────────────────────────────────────────── */
function SiteRow({ s }) {
  const [open, setOpen] = useState(false);
  const tag   = s.disrupted ? "DISRUPTED" : s.created ? "CREATED" : s.cryptic ? "CRYPTIC" : "CHANGED";
  const col   = s.disrupted ? T.red : s.created ? T.green : s.cryptic ? T.orange : T.muted2;
  const pos   = `${s.position >= 0 ? "+" : ""}${s.position}`;

  return (
    <div style={{
      background: T.bg, border: `1px solid ${col}33`,
      borderRadius: 10, overflow: "hidden",
      transition: "border-color 0.2s",
    }}>
      <div
        onClick={() => setOpen(v => !v)}
        style={{
          padding: "10px 14px", cursor: "pointer",
          display: "grid", gridTemplateColumns: "80px 1fr auto",
          gap: 10, alignItems: "center",
        }}
      >
        {/* type + tag */}
        <div>
          <div style={{ ...mono, fontSize: 9, fontWeight: 700, color: col, letterSpacing: 1 }}>{tag}</div>
          <div style={{ ...mono, fontSize: 9, color: T.muted2, marginTop: 2 }}>{s.type} · pos {pos}</div>
        </div>

        {/* kmer */}
        <div>
          <div style={{ ...mono, fontSize: 9, color: T.muted }}>
            REF: <span style={{ color: T.muted2 }}>{s.ref_kmer || "—"}</span>
          </div>
          <div style={{ ...mono, fontSize: 9, color: T.muted, marginTop: 2 }}>
            ALT: <span style={{ color: col }}>{s.alt_kmer || "—"}</span>
          </div>
        </div>

        {/* score delta */}
        <div style={{ textAlign: "right" }}>
          <div style={{ ...mono, fontSize: 9, color: T.muted }}>
            {s.ref_score.toFixed(2)} → {s.alt_score.toFixed(2)}
          </div>
          <div style={{
            ...mono, fontSize: 14, fontWeight: 700,
            color: s.delta < 0 ? T.red : T.green,
            marginTop: 2,
          }}>
            {s.delta > 0 ? "+" : ""}{s.delta.toFixed(3)}
          </div>
          <div style={{ ...mono, fontSize: 9, color: T.muted2, marginTop: 2 }}>
            {open ? "▲" : "▼"}
          </div>
        </div>
      </div>

      {/* reasoning panel */}
      {open && (
        <div style={{
          padding: "10px 14px 14px",
          borderTop: `1px solid ${T.border}`,
          background: "#080a14",
          animation: "slideUp 0.2s ease",
        }}>
          <div style={{ ...mono, fontSize: 9, letterSpacing: 1.5, color: T.muted2, marginBottom: 6 }}>
            REASONING
          </div>
          <div style={{ ...mono, fontSize: 11, color: T.text, lineHeight: 1.75 }}>
            {s.reasoning}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Stat pill ──────────────────────────────────────────────────── */
function Pill({ label, val, color }) {
  const c = val > 0 ? (color || T.red) : T.muted;
  return (
    <div style={{
      background: T.bg, border: `1px solid ${val > 0 ? c + "44" : T.border}`,
      borderRadius: 8, padding: "8px 12px", textAlign: "center", minWidth: 80, flex: 1,
    }}>
      <div style={{ ...mono, fontSize: 20, fontWeight: 700, color: c }}>{val}</div>
      <div style={{ ...mono, fontSize: 9, color: T.muted, marginTop: 3, lineHeight: 1.4 }}>{label}</div>
    </div>
  );
}

/* ─── Splice card ────────────────────────────────────────────────── */
function SpliceCard({ r, onClose }) {
  const sigColor = SIG_COLOR(r.pathogenicity_signal);
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? r.sites : r.sites.slice(0, 5);

  return (
    <div style={{
      background: `linear-gradient(135deg, ${T.panel} 0%, #0f1323 100%)`,
      border: `1px solid ${sigColor}33`, borderRadius: 16, padding: 24,
      position: "relative", boxShadow: `0 8px 48px ${sigColor}22`,
      animation: "slideUp 0.45s cubic-bezier(0.16,1,0.3,1)",
    }}>
      <button onClick={onClose} style={{
        position: "absolute", top: 16, right: 16,
        background: "none", border: "none", color: T.muted, cursor: "pointer", fontSize: 18,
      }}>✕</button>

      {/* header */}
      <div style={{ ...mono, fontSize: 9, letterSpacing: 2, color: T.muted2, marginBottom: 6 }}>
        SPLICE SITE ANALYSIS · {r.latency_ms}ms
      </div>
      <div style={{ ...sans, fontSize: 22, fontWeight: 700, color: sigColor, lineHeight: 1.2, marginBottom: 6 }}>
        {r.pathogenicity_signal}
      </div>

      {/* mutation info */}
      <div style={{
        marginBottom: 16, padding: "12px 14px",
        background: "#080a14", borderRadius: 10,
        border: `1px solid ${T.border2}`,
        borderLeft: `3px solid ${MUT_COLOR(r.mutation_type)}`,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <MutBadge type={r.mutation_type} />
          <span style={{ ...mono, fontSize: 10, color: T.muted2 }}>{r.mutation_detail}</span>
        </div>
        <div style={{ ...mono, fontSize: 11, color: T.text, lineHeight: 1.7 }}>
          {r.mutation_mechanism}
        </div>
      </div>

      {/* summary */}
      <div style={{ ...mono, fontSize: 11, color: T.muted2, marginBottom: 16 }}>{r.summary}</div>

      {/* stat pills */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 20 }}>
        <Pill label="Disrupted Donors"    val={r.disrupted_donors}    color={T.red} />
        <Pill label="Disrupted Acceptors" val={r.disrupted_acceptors} color={T.red} />
        <Pill label="New Donors"          val={r.created_donors}      color={T.orange} />
        <Pill label="New Acceptors"       val={r.created_acceptors}   color={T.orange} />
        <Pill label="Cryptic Donors"      val={r.cryptic_donors}      color={T.yellow} />
        <Pill label="Cryptic Acceptors"   val={r.cryptic_acceptors}   color={T.yellow} />
      </div>

      {/* max disruption bar */}
      {r.max_disruption > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
            <span style={{ ...mono, fontSize: 9, color: T.muted2, letterSpacing: 1 }}>MAX SCORE CHANGE</span>
            <span style={{ ...mono, fontSize: 9, color: sigColor }}>
              {(r.max_disruption * 100).toFixed(0)}%
            </span>
          </div>
          <div style={{ height: 5, background: T.border, borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: `${Math.min(r.max_disruption * 100, 100)}%`,
              background: `linear-gradient(90deg, ${sigColor}88, ${sigColor})`,
              transition: "width 0.8s cubic-bezier(0.16,1,0.3,1)",
            }} />
          </div>
        </div>
      )}

      {/* sites */}
      {r.sites.length > 0 ? (
        <div>
          <div style={{ ...mono, fontSize: 9, color: T.muted2, letterSpacing: 1, marginBottom: 10 }}>
            SITES ({r.sites_found} found) — click a row to see reasoning
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {visible.map((s, i) => <SiteRow key={i} s={s} />)}
          </div>
          {r.sites.length > 5 && (
            <button onClick={() => setShowAll(v => !v)} style={{
              marginTop: 10, width: "100%", background: "none",
              border: `1px solid ${T.border2}`, borderRadius: 8,
              color: T.muted2, ...mono, fontSize: 10, padding: "8px",
              cursor: "pointer",
            }}>
              {showAll ? "Show less" : `Show all ${r.sites.length} sites`}
            </button>
          )}
        </div>
      ) : (
        <div style={{ ...mono, fontSize: 11, color: T.muted, textAlign: "center", padding: 24 }}>
          No significant splice site changes detected
        </div>
      )}

      {/* variant grid */}
      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 7 }}>
        {[["CHROM", r.chrom], ["POSITION", r.position?.toLocaleString()],
          ["REF", r.ref], ["ALT", r.alt]].map(([k, v]) => (
          <div key={k} style={{
            background: T.bg, borderRadius: 8, padding: "8px 12px",
            border: `1px solid ${T.border}`,
          }}>
            <div style={{ ...mono, fontSize: 9, color: T.muted, letterSpacing: 1, marginBottom: 3 }}>{k}</div>
            <div style={{ ...mono, fontSize: 13, color: T.text }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Main App ───────────────────────────────────────────────────── */
export default function App() {
  const [form,    setForm]    = useState({ chrom:"", position:"", ref:"", alt:"", ref_seq:"", alt_seq:"" });
  const [mode,    setMode]    = useState("xgb");
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);
  const [health,  setHealth]  = useState(null);

  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }));

  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(setHealth).catch(() => {});
  }, []);

  const run = async () => {
    const { chrom, position, ref, alt } = form;
    if (!chrom || !position || !ref || !alt) {
      setError("chrom, position, ref, alt are required."); return;
    }
    if (mode === "splice_sites" && !form.ref_seq) {
      setError("ref_seq is required for splice site analysis."); return;
    }
    setLoading(true); setError(null); setResult(null);
    try {
      const body = {
        chrom, position: parseInt(position),
        ref: ref.toUpperCase(), alt: alt.toUpperCase(),
        ref_seq: form.ref_seq || null, alt_seq: form.alt_seq || null,
      };
      const res  = await fetch(`${API}/predict/${mode}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `HTTP ${res.status}`); }
      const data = await res.json();
      setResult({ ...data, _mode: mode });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const inp = {
    width: "100%", background: T.bg,
    border: `1px solid ${T.border}`, borderRadius: 8,
    padding: "11px 14px", color: T.text,
    ...mono, fontSize: 12, outline: "none",
    boxSizing: "border-box", transition: "border-color 0.2s",
  };
  const lbl = {
    ...mono, fontSize: 9, letterSpacing: 1.5, color: T.muted2,
    textTransform: "uppercase", marginBottom: 6, display: "block",
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        body{background:${T.bg};min-height:100vh}
        input:focus{border-color:${T.accent}!important;box-shadow:0 0 0 3px ${T.accent}11}
        input::placeholder{color:${T.muted}}
        button:hover{opacity:0.88}
        @keyframes slideUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:${T.bg}}
        ::-webkit-scrollbar-thumb{background:${T.border2};border-radius:2px}
      `}</style>

      <div style={{
        minHeight: "100vh", background: T.bg,
        display: "flex", justifyContent: "center",
        padding: "44px 16px 80px",
      }}>
        <div style={{ width: "100%", maxWidth: 580 }}>

          {/* ── Header ── */}
          <div style={{ marginBottom: 40, animation: "fadeIn 0.6s ease" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <div style={{
                width: 8, height: 8, borderRadius: "50%",
                background: health?.xgb ? T.green : T.muted,
                boxShadow: health?.xgb ? `0 0 10px ${T.green}` : "none",
                animation: "pulse 2s infinite",
              }} />
              <span style={{ ...mono, fontSize: 9, letterSpacing: 2, color: T.muted2 }}>
                SPLICE VARIANT CLASSIFIER
              </span>
              {health && (
                <span style={{ marginLeft: "auto", ...mono, fontSize: 9, color: T.muted }}>
                  {health.xgb ? "XGB ✓" : "XGB ✗"}  {health.dnabert ? "DNA ✓" : "DNA ✗"}
                </span>
              )}
            </div>
            <h1 style={{
              ...sans, fontSize: 42, fontWeight: 700,
              color: T.text, letterSpacing: -1.5, lineHeight: 1.05, marginBottom: 10,
            }}>
              Pathogenicity<br />
              <span style={{ color: T.accent }}>+ Splice Analysis</span>
            </h1>
            <p style={{ ...mono, fontSize: 10, color: T.muted, lineHeight: 1.9 }}>
              XGBoost k-mer · DNABERT-2 · Donor/Acceptor PWM · Cryptic Site Detection
            </p>
          </div>

          {/* ── Mode selector ── */}
          <div style={{ marginBottom: 24 }}>
            <div style={lbl}>Analysis Mode</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 7 }}>
              {MODES.map(m => (
                <button key={m.id} onClick={() => setMode(m.id)} style={{
                  background: mode === m.id ? "#121629" : T.panel,
                  border: `1px solid ${mode === m.id ? T.accent : T.border}`,
                  borderRadius: 10, padding: "10px 6px", cursor: "pointer",
                  transition: "all 0.2s",
                  boxShadow: mode === m.id ? `0 0 20px ${T.accent}22` : "none",
                }}>
                  <div style={{
                    ...sans, fontWeight: 600, fontSize: 11,
                    color: mode === m.id ? T.accent : T.muted2,
                  }}>{m.label}</div>
                  <div style={{
                    ...mono, fontSize: 8, color: T.muted,
                    marginTop: 3, lineHeight: 1.5,
                  }}>{m.sub}</div>
                </button>
              ))}
            </div>
          </div>

          {/* ── Form ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 10 }}>
              <div>
                <label style={lbl}>Chrom</label>
                <input style={inp} value={form.chrom} onChange={set("chrom")} placeholder="1" />
              </div>
              <div>
                <label style={lbl}>Position</label>
                <input style={inp} type="number" value={form.position} onChange={set("position")} placeholder="925952" />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div>
                <label style={lbl}>REF Allele</label>
                <input style={inp} value={form.ref} onChange={set("ref")} placeholder="G" maxLength={50} />
              </div>
              <div>
                <label style={lbl}>ALT Allele</label>
                <input style={inp} value={form.alt} onChange={set("alt")} placeholder="A" maxLength={50} />
              </div>
            </div>

            <div>
              <label style={lbl}>
                REF Sequence ±200bp
                {mode === "splice_sites" && <span style={{ color: T.orange }}> (required)</span>}
                {mode !== "splice_sites" && <span style={{ color: T.muted }}> (optional — improves accuracy)</span>}
              </label>
              <input
                style={{ ...inp, fontSize: 10, letterSpacing: 0.3 }}
                value={form.ref_seq}
                onChange={set("ref_seq")}
                placeholder="Genomic window around variant…"
              />
            </div>

            <div>
              <label style={lbl}>ALT Sequence <span style={{ color: T.muted }}>(auto-built from REF if blank)</span></label>
              <input
                style={{ ...inp, fontSize: 10, letterSpacing: 0.3 }}
                value={form.alt_seq}
                onChange={set("alt_seq")}
                placeholder="Leave blank to auto-compute…"
              />
            </div>

            <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
              <button onClick={run} disabled={loading} style={{
                flex: 1, padding: "14px 0",
                background: loading ? T.panel : `linear-gradient(135deg, #1a3a6b, #1e4d94)`,
                border: `1px solid ${loading ? T.border : T.accent + "66"}`,
                borderRadius: 10, color: loading ? T.muted : T.text,
                ...sans, fontWeight: 600, fontSize: 14,
                cursor: loading ? "not-allowed" : "pointer",
                boxShadow: loading ? "none" : `0 4px 24px ${T.accent}22`,
                display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                transition: "all 0.2s",
              }}>
                {loading ? (
                  <>
                    <div style={{
                      width: 14, height: 14,
                      border: `2px solid ${T.accent}33`,
                      borderTopColor: T.accent,
                      borderRadius: "50%",
                      animation: "spin 0.7s linear infinite",
                    }} />
                    Analysing…
                  </>
                ) : "Analyse →"}
              </button>
              <button
                onClick={() => { setForm({ ...EX, alt_seq: "" }); setResult(null); setError(null); }}
                style={{
                  padding: "14px 18px", background: T.panel,
                  border: `1px solid ${T.border}`, borderRadius: 10,
                  color: T.muted2, ...mono, fontSize: 10, cursor: "pointer",
                  transition: "all 0.2s",
                }}
              >
                Example
              </button>
            </div>
          </div>

          {/* ── Error ── */}
          {error && (
            <div style={{
              marginTop: 16, padding: "12px 16px",
              background: "#1a080d", border: `1px solid ${T.red}44`,
              borderRadius: 10, color: T.red,
              ...mono, fontSize: 11, animation: "fadeIn 0.3s ease",
            }}>
              ⚠ {error}
            </div>
          )}

          {/* ── Result ── */}
          {result && (
            <div style={{ marginTop: 22 }}>
              {result._mode === "splice_sites"
                ? <SpliceCard r={result} onClose={() => setResult(null)} />
                : <PredCard   r={result} onClose={() => setResult(null)} />}
            </div>
          )}

          {/* ── Footer ── */}
          <div style={{
            marginTop: 48, textAlign: "center",
            ...mono, fontSize: 9, color: T.border2, letterSpacing: 1,
          }}>
            {API} · XGBoost + DNABERT-2-117M · PWM splice scoring
          </div>
        </div>
      </div>
    </>
  );
}