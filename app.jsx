import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8000";

/* ─── DESIGN TOKENS ──────────────────────────────────────────────── */
const C = {
  bg:        "#050709",
  surface:   "#0a0c12",
  surface2:  "#0f1219",
  surface3:  "#141820",
  border:    "#1a1f2e",
  border2:   "#232a3d",
  border3:   "#2d3550",
  text:      "#e2e8f0",
  textMuted: "#64748b",
  textDim:   "#94a3b8",
  accent:    "#3b82f6",
  accentGlow:"#3b82f633",
  red:       "#ef4444",
  redSoft:   "#7f1d1d",
  orange:    "#f97316",
  yellow:    "#eab308",
  green:     "#22c55e",
  teal:      "#14b8a6",
  violet:    "#8b5cf6",
  pink:      "#ec4899",
  cyan:      "#06b6d4",
};

const MODES = [
  { id:"xgb",          label:"XGBoost",      sub:"k-mer · CPU",       icon:"⬡" },
  { id:"dnabert",      label:"DNABERT-2",    sub:"sequence · GPU",    icon:"◈" },
  { id:"ensemble",     label:"Ensemble",     sub:"XGB 40% + DNA 60%", icon:"◎" },
  { id:"splice_sites", label:"Splice Sites", sub:"PWM · cryptic",     icon:"⟨⟩" },
];

const EX = {
  chrom:"1", position:"925952", ref:"G", alt:"A",
  ref_seq:"AGCTGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
};

/* ─── HELPERS ────────────────────────────────────────────────────── */
const mono = "'JetBrains Mono','Fira Code','Cascadia Code',monospace";
const sans = "'Sora','DM Sans','Inter',sans-serif";

function pathColor(isPath) { return isPath ? C.red : C.green; }
function sigColor(s="") {
  if (s.startsWith("Strong")) return C.red;
  if (s.startsWith("Moderate")) return C.orange;
  if (s.startsWith("Weak") || s.startsWith("Low")) return C.yellow;
  return C.green;
}
function mutColor(t="") {
  return t==="SNV"?C.accent:t==="Deletion"?C.red:t==="Insertion"?C.orange:C.violet;
}

/* ─── ANIMATED COUNTER ───────────────────────────────────────────── */
function Counter({ to, dur=900, dec=1 }) {
  const [v,setV] = useState(0);
  useEffect(() => {
    let s=Date.now(), from=0;
    const tick=()=>{
      const p=Math.min((Date.now()-s)/dur,1);
      const ease=1-Math.pow(1-p,4);
      setV(from+(to-from)*ease);
      if(p<1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  },[to]);
  return <>{v.toFixed(dec)}</>;
}

/* ─── TOPOLOGY BAR ───────────────────────────────────────────────── */
function TopoBar({ val, max=1, color, height=4, animate=true }) {
  const [w,setW]=useState(0);
  useEffect(()=>{ requestAnimationFrame(()=>setW(val/max*100)); },[val]);
  return (
    <div style={{ height, background:C.border, borderRadius:2, overflow:"hidden" }}>
      <div style={{
        height:"100%", width:`${animate?w:val/max*100}%`,
        background:color, borderRadius:2,
        transition:"width 0.8s cubic-bezier(0.16,1,0.3,1)",
      }}/>
    </div>
  );
}

/* ─── GAUGE RING ─────────────────────────────────────────────────── */
function GaugeRing({ prob, isPath, size=120 }) {
  const r=48, circ=2*Math.PI*r;
  const offset=circ*(1-prob);
  const color=pathColor(isPath);
  return (
    <div style={{ position:"relative", width:size, height:size }}>
      <svg width={size} height={size} viewBox="0 0 110 110" style={{ transform:"rotate(-90deg)" }}>
        <circle cx={55} cy={55} r={r} fill="none" stroke={C.border2} strokeWidth={8}/>
        <circle cx={55} cy={55} r={r} fill="none" stroke={color} strokeWidth={8}
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition:"stroke-dashoffset 1.1s cubic-bezier(0.16,1,0.3,1)" }}/>
      </svg>
      <div style={{
        position:"absolute",inset:0,display:"flex",flexDirection:"column",
        alignItems:"center",justifyContent:"center",
      }}>
        <span style={{ fontFamily:mono, fontSize:22, fontWeight:700, color, lineHeight:1 }}>
          <Counter to={Math.round(prob*100)} dec={0}/>%
        </span>
        <span style={{ fontFamily:mono, fontSize:8, color:C.textMuted, letterSpacing:1.5, marginTop:2 }}>
          PROB
        </span>
      </div>
    </div>
  );
}

/* ─── SCORE DELTA CHIP ───────────────────────────────────────────── */
function DeltaChip({ delta }) {
  const up=delta>=0, col=up?C.green:C.red;
  return (
    <span style={{
      fontFamily:mono, fontSize:11, fontWeight:700, color:col,
      background:`${col}18`, border:`1px solid ${col}44`,
      borderRadius:4, padding:"2px 7px",
    }}>
      {up?"+":""}{delta.toFixed(3)}
    </span>
  );
}

/* ─── BADGE ──────────────────────────────────────────────────────── */
function Badge({ label, color }) {
  return (
    <span style={{
      fontFamily:mono, fontSize:9, letterSpacing:1.2, fontWeight:700,
      padding:"3px 8px", borderRadius:3,
      background:`${color}20`, border:`1px solid ${color}50`,
      color, textTransform:"uppercase",
    }}>{label}</span>
  );
}

/* ─── K-MER SEQUENCE DISPLAY ─────────────────────────────────────── */
function KmerDisplay({ ref_kmer, alt_kmer, changed_pos }) {
  if (!ref_kmer && !alt_kmer) return null;
  const renderSeq = (seq, isAlt) => {
    if (!seq) return <span style={{ color:C.textMuted, fontFamily:mono, fontSize:11 }}>—</span>;
    return (
      <span style={{ fontFamily:mono, fontSize:11, letterSpacing:1 }}>
        {seq.split("").map((c,i) => {
          const diff = ref_kmer && alt_kmer && ref_kmer[i] !== alt_kmer[i];
          const highlight = diff;
          return (
            <span key={i} style={{
              color: highlight && isAlt ? C.orange : C.textDim,
              background: highlight && isAlt ? `${C.orange}22` : "transparent",
              borderRadius:2, padding:"0 1px",
            }}>{c}</span>
          );
        })}
      </span>
    );
  };
  return (
    <div style={{ background:C.bg, borderRadius:6, padding:"8px 12px", marginTop:8 }}>
      <div style={{ display:"flex", gap:8, alignItems:"center", marginBottom:4 }}>
        <span style={{ fontFamily:mono, fontSize:9, color:C.textMuted, width:28 }}>REF</span>
        {renderSeq(ref_kmer, false)}
      </div>
      <div style={{ display:"flex", gap:8, alignItems:"center" }}>
        <span style={{ fontFamily:mono, fontSize:9, color:C.textMuted, width:28 }}>ALT</span>
        {renderSeq(alt_kmer, true)}
      </div>
    </div>
  );
}

/* ─── SITE ROW ────────────────────────────────────────────────────── */
function SiteRow({ s, idx }) {
  const [open,setOpen]=useState(false);
  const tag = s.disrupted?"DISRUPTED":s.created?"CREATED":s.cryptic?"CRYPTIC":"SHIFTED";
  const col = s.disrupted?C.red:s.created?C.green:s.cryptic?C.orange:C.textMuted;
  const pos = `${s.position>=0?"+":""}${s.position}`;
  const maxScore = Math.max(s.ref_score, s.alt_score, 0.01);

  return (
    <div style={{
      background:C.surface, border:`1px solid ${col}30`,
      borderRadius:8, overflow:"hidden",
      transition:"border-color 0.2s",
      animation:`slideIn 0.3s ${idx*0.04}s both`,
    }}>
      <div onClick={()=>setOpen(v=>!v)} style={{
        padding:"12px 16px", cursor:"pointer",
        display:"grid", gridTemplateColumns:"100px 1fr 120px 24px",
        gap:12, alignItems:"center",
      }}>
        <div>
          <Badge label={tag} color={col}/>
          <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, marginTop:4 }}>
            {s.type.toUpperCase()} · {pos}
          </div>
        </div>

        <div>
          <div style={{ display:"flex", gap:6, alignItems:"center", marginBottom:6 }}>
            <span style={{ fontFamily:mono, fontSize:9, color:C.textMuted, width:24 }}>REF</span>
            <div style={{ flex:1, height:3, background:C.border, borderRadius:2, overflow:"hidden" }}>
              <div style={{ height:"100%", width:`${s.ref_score*100}%`, background:C.textMuted, borderRadius:2 }}/>
            </div>
            <span style={{ fontFamily:mono, fontSize:10, color:C.textDim, width:30, textAlign:"right" }}>
              {s.ref_score.toFixed(2)}
            </span>
          </div>
          <div style={{ display:"flex", gap:6, alignItems:"center" }}>
            <span style={{ fontFamily:mono, fontSize:9, color:C.textMuted, width:24 }}>ALT</span>
            <div style={{ flex:1, height:3, background:C.border, borderRadius:2, overflow:"hidden" }}>
              <div style={{ height:"100%", width:`${s.alt_score*100}%`, background:col, borderRadius:2,
                transition:"width 0.6s ease" }}/>
            </div>
            <span style={{ fontFamily:mono, fontSize:10, color:col, fontWeight:700, width:30, textAlign:"right" }}>
              {s.alt_score.toFixed(2)}
            </span>
          </div>
        </div>

        <div style={{ textAlign:"right" }}>
          <DeltaChip delta={s.delta}/>
        </div>

        <div style={{ fontFamily:mono, fontSize:10, color:C.textMuted }}>
          {open?"▲":"▼"}
        </div>
      </div>

      {open && (
        <div style={{
          padding:"0 16px 14px",
          borderTop:`1px solid ${C.border}`,
          animation:"fadeIn 0.2s ease",
        }}>
          <KmerDisplay ref_kmer={s.ref_kmer} alt_kmer={s.alt_kmer}/>
          <div style={{
            marginTop:10, padding:"10px 12px",
            background:C.bg, borderRadius:6,
            borderLeft:`3px solid ${col}`,
          }}>
            <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:1.5, marginBottom:6 }}>
              MOLECULAR REASONING
            </div>
            <div style={{ fontFamily:mono, fontSize:11, color:C.text, lineHeight:1.85 }}>
              {s.reasoning}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── FEATURE CONTRIBUTIONS (biology-grounded, no fake numbers) ──── */
function FeatureSection({ r }) {
  // These are derived from actual biological rules, not hardcoded scores.
  // XGBoost SHAP would give per-variant values; here we show which rules fired.
  const ref = r.ref?.toUpperCase() || "";
  const alt = r.alt?.toUpperCase() || "";
  const mut = r.mutation_type || "";

  const rules = [
    {
      label: "Canonical +1G (donor)",
      fired: ref === "G",
      description: ref === "G"
        ? `REF is G at position — invariant donor nucleotide. ${alt !== "G" ? "ALT disrupts it." : "ALT preserves it."}`
        : "REF is not G — canonical donor +1 position not affected.",
      severity: ref === "G" && alt !== "G" ? "high" : "low",
    },
    {
      label: "Canonical +2T (donor)",
      fired: ref === "T",
      description: ref === "T"
        ? `REF is T at position — invariant donor GT dinucleotide. ${alt !== "T" ? "ALT disrupts +2T." : "ALT preserves it."}`
        : "REF is not T — donor +2T not directly affected.",
      severity: ref === "T" && alt !== "T" ? "high" : "low",
    },
    {
      label: "Canonical -2A (acceptor)",
      fired: ref === "A",
      description: ref === "A"
        ? `REF is A — acceptor AG dinucleotide context. ${alt !== "A" ? "ALT disrupts -2A." : "ALT preserves it."}`
        : "REF is not A — acceptor -2A not directly affected.",
      severity: ref === "A" && alt !== "A" ? "high" : "low",
    },
    {
      label: "GT dinucleotide in REF",
      fired: ref.includes("GT"),
      description: ref.includes("GT")
        ? `GT motif present in REF allele. ${!alt.includes("GT") ? "Lost in ALT — potential donor disruption." : "Preserved in ALT."}`
        : "No GT motif in REF allele.",
      severity: ref.includes("GT") && !alt.includes("GT") ? "high" : "low",
    },
    {
      label: "AG dinucleotide in REF",
      fired: ref.includes("AG"),
      description: ref.includes("AG")
        ? `AG motif present in REF allele. ${!alt.includes("AG") ? "Lost in ALT — potential acceptor disruption." : "Preserved in ALT."}`
        : "No AG motif in REF allele.",
      severity: ref.includes("AG") && !alt.includes("AG") ? "high" : "low",
    },
    {
      label: "Frameshift risk",
      fired: mut === "Insertion" || mut === "Deletion",
      description: mut === "Insertion" || mut === "Deletion"
        ? `${mut} variant. ${(Math.abs(ref.length - alt.length) % 3 !== 0) ? "Frameshift — disrupts reading frame and splice signals." : "In-frame indel — lower frameshift risk but may still disrupt splice geometry."}`
        : "SNV or MNV — no frameshift from this variant alone.",
      severity: mut === "Insertion" || mut === "Deletion" ? "medium" : "low",
    },
    {
      label: "Multi-nucleotide substitution",
      fired: mut === "MNV",
      description: mut === "MNV"
        ? "Complex multi-nucleotide variant — multiple consecutive positions affected. High risk of disrupting conserved splice site motifs."
        : "Single allele change — MNV effect not applicable.",
      severity: mut === "MNV" ? "medium" : "low",
    },
  ];

  const sevColor = (s) => s==="high" ? C.red : s==="medium" ? C.orange : C.border2;

  return (
    <div style={{ marginTop:16 }}>
      <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:2, marginBottom:4 }}>
        BIOLOGICAL RULE EVALUATION
      </div>
      <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, marginBottom:12, lineHeight:1.6 }}>
        Rules derived from known splice site biology. Probability scores come from XGBoost/DNABERT models.
      </div>
      {rules.map((f,i) => (
        <div key={i} style={{
          marginBottom:8, padding:"10px 12px",
          background: f.fired ? `${sevColor(f.severity)}10` : C.bg,
          border:`1px solid ${f.fired ? sevColor(f.severity)+"44" : C.border}`,
          borderRadius:7,
        }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:4 }}>
            <span style={{ fontFamily:mono, fontSize:10, color: f.fired ? sevColor(f.severity) : C.textMuted }}>
              {f.fired ? "● " : "○ "}{f.label}
            </span>
            {f.fired && <Badge label={f.severity} color={sevColor(f.severity)}/>}
          </div>
          <div style={{ fontFamily:mono, fontSize:10, color:C.textMuted, lineHeight:1.7 }}>
            {f.description}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ─── CONFIDENCE BREAKDOWN ───────────────────────────────────────── */
function ConfidenceBreakdown({ prob, thresh }) {
  const margin = Math.abs(prob - thresh);
  const isPath = prob >= thresh;

  // All values derived from actual model outputs — no fake hardcoded numbers
  const marginPct  = Math.min(margin * 100, 50);      // distance from threshold, capped at 50
  const certLabel  = margin > 0.35 ? "High" : margin > 0.15 ? "Medium" : "Low";
  const certColor  = margin > 0.35 ? C.green : margin > 0.15 ? C.yellow : C.red;

  return (
    <div style={{ marginTop:16, padding:"14px 16px", background:C.bg, borderRadius:8, border:`1px solid ${C.border}` }}>
      <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:2, marginBottom:12 }}>
        PREDICTION CONFIDENCE
      </div>

      {/* Margin from threshold */}
      <div style={{ marginBottom:12 }}>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
          <span style={{ fontFamily:mono, fontSize:10, color:C.textDim }}>Margin from threshold</span>
          <span style={{ fontFamily:mono, fontSize:10, color:C.cyan }}>{margin.toFixed(3)}</span>
        </div>
        <TopoBar val={marginPct} max={50} color={C.cyan} height={4}/>
        <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, marginTop:4, lineHeight:1.6 }}>
          Model output is {(margin*100).toFixed(1)} percentage points from the decision boundary ({(thresh*100).toFixed(0)}%).
          {margin < 0.1 ? " Near-boundary prediction — treat with caution." :
           margin < 0.25 ? " Moderate separation from threshold." :
           " Strong separation from threshold."}
        </div>
      </div>

      {/* Overall confidence */}
      <div style={{
        padding:"10px 12px", borderRadius:7,
        background:`${certColor}12`,
        border:`1px solid ${certColor}40`,
        display:"flex", justifyContent:"space-between", alignItems:"center",
      }}>
        <div>
          <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:1, marginBottom:3 }}>
            OVERALL CONFIDENCE
          </div>
          <div style={{ fontFamily:mono, fontSize:16, fontWeight:700, color:certColor }}>
            {certLabel}
          </div>
        </div>
        <div style={{ textAlign:"right" }}>
          <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, marginBottom:3 }}>
            {isPath ? "PATHOGENIC" : "BENIGN"} call
          </div>
          <div style={{ fontFamily:mono, fontSize:22, fontWeight:700, color:isPath?C.red:C.green }}>
            {(prob*100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, marginTop:12, lineHeight:1.8,
        padding:"8px 10px", background:C.surface, borderRadius:6, border:`1px solid ${C.border}` }}>
        ℹ Confidence is based on margin from the tuned decision threshold, not model calibration.
        For clinical use, validate against ClinVar/ACMG criteria and run splice site analysis.
      </div>
    </div>
  );
}

/* ─── VARIANT INFO GRID ──────────────────────────────────────────── */
function VarGrid({ r }) {
  const rows = [
    ["CHROMOSOME", r.chrom, C.cyan],
    ["POSITION", r.position?.toLocaleString(), C.textDim],
    ["REF ALLELE", r.ref, C.green],
    ["ALT ALLELE", r.alt, C.orange],
    ["MUT TYPE", r.mutation_type, mutColor(r.mutation_type)],
    ["LATENCY", `${r.latency_ms}ms`, C.textMuted],
  ];
  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:6, marginTop:12 }}>
      {rows.map(([k,v,c])=>(
        <div key={k} style={{
          background:C.bg, borderRadius:6, padding:"8px 12px",
          border:`1px solid ${C.border}`,
        }}>
          <div style={{ fontFamily:mono, fontSize:8, color:C.textMuted, letterSpacing:1.2, marginBottom:3 }}>{k}</div>
          <div style={{ fontFamily:mono, fontSize:12, fontWeight:700, color:c||C.text }}>{v}</div>
        </div>
      ))}
    </div>
  );
}

/* ─── PREDICTION CARD ────────────────────────────────────────────── */
function PredCard({ r, onClose }) {
  const isPath = r.prediction === "Pathogenic";
  const color  = pathColor(isPath);
  const [tab,setTab] = useState("overview");

  const tabs = ["overview","features","confidence"];

  return (
    <div style={{
      background:`linear-gradient(160deg, ${C.surface2} 0%, ${C.surface} 100%)`,
      border:`1px solid ${color}40`, borderRadius:14, overflow:"hidden",
      animation:"slideUp 0.5s cubic-bezier(0.16,1,0.3,1)",
    }}>
      {/* header bar */}
      <div style={{
        background:`linear-gradient(90deg, ${color}22, transparent)`,
        borderBottom:`1px solid ${color}30`,
        padding:"16px 20px",
        display:"flex", alignItems:"center", gap:12,
      }}>
        <div style={{
          width:12, height:12, borderRadius:"50%",
          background:color, boxShadow:`0 0 16px ${color}`,
          animation:"pulse 2s infinite",
        }}/>
        <div>
          <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:2 }}>
            {r.model?.toUpperCase()}
          </div>
          <div style={{ fontFamily:sans, fontSize:24, fontWeight:700, color, letterSpacing:-0.5 }}>
            {r.prediction}
          </div>
        </div>
        <div style={{ marginLeft:"auto", display:"flex", gap:8, alignItems:"center" }}>
          <Badge label={r.confidence+" confidence"} color={color}/>
          <button onClick={onClose} style={{
            background:"none", border:`1px solid ${C.border}`, borderRadius:6,
            color:C.textMuted, cursor:"pointer", width:28, height:28,
            fontFamily:mono, fontSize:14, lineHeight:1,
          }}>✕</button>
        </div>
      </div>

      <div style={{ padding:"20px" }}>
        {/* gauge + summary row */}
        <div style={{ display:"flex", gap:20, alignItems:"flex-start" }}>
          <GaugeRing prob={r.probability} isPath={isPath}/>
          <div style={{ flex:1 }}>
            {/* threshold bar */}
            <div style={{ marginBottom:14 }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                <span style={{ fontFamily:mono, fontSize:9, color:C.green, letterSpacing:1 }}>BENIGN</span>
                <span style={{ fontFamily:mono, fontSize:9, color:C.red, letterSpacing:1 }}>PATHOGENIC</span>
              </div>
              <div style={{ position:"relative", height:8, background:C.border, borderRadius:4 }}>
                <div style={{
                  position:"absolute", left:0, top:0, height:"100%",
                  width:`${r.probability*100}%`, borderRadius:4,
                  background:`linear-gradient(90deg,${color}60,${color})`,
                  transition:"width 0.9s cubic-bezier(0.16,1,0.3,1)",
                }}/>
                <div style={{
                  position:"absolute",
                  left:`${r.threshold_used*100}%`,
                  top:-4, transform:"translateX(-50%)",
                  width:2, height:16, background:C.yellow, borderRadius:1,
                }}/>
              </div>
              <div style={{ display:"flex", justifyContent:"space-between", marginTop:5 }}>
                <span style={{ fontFamily:mono, fontSize:8, color:C.yellow }}>
                  ▲ threshold {(r.threshold_used*100).toFixed(0)}%
                </span>
                <span style={{ fontFamily:mono, fontSize:8, color:C.textMuted }}>
                  prob {(r.probability*100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* mutation */}
            <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
              <Badge label={r.mutation_type} color={mutColor(r.mutation_type)}/>
              <Badge label={`chr${r.chrom}:${r.position} ${r.ref}→${r.alt}`} color={C.textMuted}/>
            </div>
          </div>
        </div>

        {/* tabs */}
        <div style={{ display:"flex", gap:0, marginTop:18, borderBottom:`1px solid ${C.border}` }}>
          {tabs.map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={{
              background:"none", border:"none", cursor:"pointer",
              fontFamily:mono, fontSize:10, letterSpacing:1, textTransform:"uppercase",
              color: tab===t ? color : C.textMuted,
              borderBottom: tab===t ? `2px solid ${color}` : "2px solid transparent",
              padding:"8px 16px", marginBottom:-1, transition:"all 0.2s",
            }}>{t}</button>
          ))}
        </div>

        {tab==="overview" && (
          <div style={{ marginTop:14 }}>
            {/* mechanism */}
            <div style={{
              padding:"12px 14px", background:C.bg, borderRadius:8,
              borderLeft:`3px solid ${color}`, marginBottom:12,
            }}>
              <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:1.5, marginBottom:6 }}>
                MOLECULAR MECHANISM
              </div>
              <div style={{ fontFamily:mono, fontSize:11, color:C.text, lineHeight:1.85 }}>
                {r.mechanism}
              </div>
            </div>
            <VarGrid r={r}/>
          </div>
        )}

        {tab==="features" && <FeatureSection r={r}/>}
        {tab==="confidence" && <ConfidenceBreakdown prob={r.probability} thresh={r.threshold_used}/>}
      </div>
    </div>
  );
}

/* ─── STAT PILLS ─────────────────────────────────────────────────── */
function StatPill({ label, val, color, icon }) {
  const active = val > 0;
  return (
    <div style={{
      background: active ? `${color}15` : C.bg,
      border:`1px solid ${active ? color+"50" : C.border}`,
      borderRadius:8, padding:"10px 14px", textAlign:"center", flex:1, minWidth:90,
    }}>
      <div style={{ fontFamily:mono, fontSize:22, fontWeight:700, color: active ? color : C.textMuted }}>
        {val}
      </div>
      <div style={{ fontFamily:mono, fontSize:8, color:C.textMuted, marginTop:3, lineHeight:1.5 }}>
        {label}
      </div>
    </div>
  );
}

/* ─── CRYPTIC SITE HEATMAP ───────────────────────────────────────── */
function CrypticHeatmap({ sites }) {
  if (!sites?.length) return null;
  const donors    = sites.filter(s=>s.type==="donor");
  const acceptors = sites.filter(s=>s.type==="acceptor");

  const renderTrack = (items, label, color) => {
    if (!items.length) return null;
    const minPos = Math.min(...items.map(s=>s.position));
    const maxPos = Math.max(...items.map(s=>s.position));
    const range  = Math.max(maxPos - minPos, 1);
    return (
      <div style={{ marginBottom:10 }}>
        <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:1, marginBottom:4 }}>
          {label.toUpperCase()} SITES ({items.length})
        </div>
        <div style={{ position:"relative", height:28, background:C.border, borderRadius:4 }}>
          {items.map((s,i)=>{
            const xPct = range>0 ? ((s.position-minPos)/range)*90+5 : 50;
            const col  = s.disrupted?C.red:s.created?C.green:s.cryptic?C.orange:C.textMuted;
            const h    = Math.max(s.alt_score*28, 4);
            return (
              <div key={i} title={`pos ${s.position>=0?"+":""}${s.position} Δ${s.delta.toFixed(3)}`} style={{
                position:"absolute", bottom:0, left:`${xPct}%`,
                width:4, height:h, background:col,
                borderRadius:"2px 2px 0 0", transform:"translateX(-50%)",
                opacity:0.85,
              }}/>
            );
          })}
        </div>
        <div style={{ display:"flex", justifyContent:"space-between", marginTop:3 }}>
          <span style={{ fontFamily:mono, fontSize:8, color:C.textMuted }}>{minPos>=0?"+":""}{minPos}bp</span>
          <span style={{ fontFamily:mono, fontSize:8, color:C.textMuted }}>{maxPos>=0?"+":""}{maxPos}bp</span>
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding:"12px 14px", background:C.bg, borderRadius:8, border:`1px solid ${C.border}`, marginBottom:14 }}>
      <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:2, marginBottom:10 }}>
        SPLICE SITE LANDSCAPE
      </div>
      {renderTrack(donors, "donor", C.accent)}
      {renderTrack(acceptors, "acceptor", C.violet)}
      <div style={{ display:"flex", gap:10, marginTop:8 }}>
        {[["DISRUPTED",C.red],["CREATED",C.green],["CRYPTIC",C.orange]].map(([l,c])=>(
          <span key={l} style={{ display:"flex", alignItems:"center", gap:4, fontFamily:mono, fontSize:8, color:C.textMuted }}>
            <span style={{ width:8, height:8, background:c, borderRadius:1 }}/>
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ─── SPLICE CARD ────────────────────────────────────────────────── */
function SpliceCard({ r, onClose }) {
  const sc = sigColor(r.pathogenicity_signal);
  const [showAll, setShowAll] = useState(false);
  const [tab, setTab] = useState("sites");
  const visible = showAll ? r.sites : r.sites.slice(0, 6);

  const totalImpact = r.disrupted_donors+r.disrupted_acceptors+r.created_donors+r.created_acceptors+r.cryptic_donors+r.cryptic_acceptors;

  return (
    <div style={{
      background:`linear-gradient(160deg, ${C.surface2} 0%, ${C.surface} 100%)`,
      border:`1px solid ${sc}40`, borderRadius:14, overflow:"hidden",
      animation:"slideUp 0.5s cubic-bezier(0.16,1,0.3,1)",
    }}>
      {/* header */}
      <div style={{
        background:`linear-gradient(90deg, ${sc}22, transparent)`,
        borderBottom:`1px solid ${sc}30`,
        padding:"16px 20px",
        display:"flex", alignItems:"flex-start", gap:12,
      }}>
        <div style={{ flex:1 }}>
          <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:2, marginBottom:4 }}>
            SPLICE SITE ANALYSIS · {r.latency_ms}ms · {r.sites_found} sites
          </div>
          <div style={{ fontFamily:sans, fontSize:20, fontWeight:700, color:sc, lineHeight:1.2 }}>
            {r.pathogenicity_signal}
          </div>
          <div style={{ fontFamily:mono, fontSize:10, color:C.textMuted, marginTop:4 }}>
            {r.summary}
          </div>
        </div>
        <button onClick={onClose} style={{
          background:"none", border:`1px solid ${C.border}`, borderRadius:6,
          color:C.textMuted, cursor:"pointer", width:28, height:28,
          fontFamily:mono, fontSize:14, lineHeight:1,
        }}>✕</button>
      </div>

      <div style={{ padding:"20px" }}>
        {/* mutation info */}
        <div style={{
          padding:"12px 14px", background:C.bg, borderRadius:8,
          border:`1px solid ${C.border}`,
          borderLeft:`3px solid ${mutColor(r.mutation_type)}`,
          marginBottom:16,
        }}>
          <div style={{ display:"flex", gap:8, alignItems:"center", marginBottom:6 }}>
            <Badge label={r.mutation_type} color={mutColor(r.mutation_type)}/>
            <span style={{ fontFamily:mono, fontSize:10, color:C.textMuted }}>{r.mutation_detail}</span>
          </div>
          <div style={{ fontFamily:mono, fontSize:11, color:C.text, lineHeight:1.85 }}>
            {r.mutation_mechanism}
          </div>
        </div>

        {/* stat pills */}
        <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginBottom:16 }}>
          <StatPill label="Disrupted Donors"    val={r.disrupted_donors}    color={C.red}/>
          <StatPill label="Disrupted Acceptors" val={r.disrupted_acceptors} color={C.red}/>
          <StatPill label="New Donors"          val={r.created_donors}      color={C.orange}/>
          <StatPill label="New Acceptors"       val={r.created_acceptors}   color={C.orange}/>
          <StatPill label="Cryptic Donors"      val={r.cryptic_donors}      color={C.yellow}/>
          <StatPill label="Cryptic Acceptors"   val={r.cryptic_acceptors}   color={C.yellow}/>
        </div>

        {/* max disruption */}
        {r.max_disruption > 0 && (
          <div style={{ marginBottom:16 }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
              <span style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:1 }}>
                MAX SCORE CHANGE
              </span>
              <span style={{ fontFamily:mono, fontSize:10, fontWeight:700, color:sc }}>
                {(r.max_disruption*100).toFixed(1)}%
              </span>
            </div>
            <TopoBar val={r.max_disruption} max={1} color={sc} height={6}/>
          </div>
        )}

        {/* landscape heatmap */}
        {r.sites?.length > 0 && <CrypticHeatmap sites={r.sites}/>}

        {/* tabs */}
        <div style={{ display:"flex", gap:0, marginBottom:14, borderBottom:`1px solid ${C.border}` }}>
          {["sites","variant"].map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={{
              background:"none", border:"none", cursor:"pointer",
              fontFamily:mono, fontSize:10, letterSpacing:1, textTransform:"uppercase",
              color: tab===t ? sc : C.textMuted,
              borderBottom: tab===t ? `2px solid ${sc}` : "2px solid transparent",
              padding:"8px 16px", marginBottom:-1, transition:"all 0.2s",
            }}>{t}</button>
          ))}
        </div>

        {tab==="sites" && (
          <div>
            {visible.length > 0 ? (
              <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                {visible.map((s,i)=><SiteRow key={i} s={s} idx={i}/>)}
              </div>
            ) : (
              <div style={{
                fontFamily:mono, fontSize:11, color:C.textMuted,
                textAlign:"center", padding:"32px",
                border:`1px dashed ${C.border}`, borderRadius:8,
              }}>
                No significant splice site changes detected
              </div>
            )}
            {r.sites?.length > 6 && (
              <button onClick={()=>setShowAll(v=>!v)} style={{
                marginTop:10, width:"100%",
                background:"none", border:`1px solid ${C.border}`,
                borderRadius:8, color:C.textMuted,
                fontFamily:mono, fontSize:10, padding:"10px",
                cursor:"pointer", transition:"all 0.2s",
              }}>
                {showAll ? "Show less ▲" : `Show all ${r.sites.length} sites ▼`}
              </button>
            )}
          </div>
        )}

        {tab==="variant" && <VarGrid r={r}/>}
      </div>
    </div>
  );
}

/* ─── INPUT COMPONENT ────────────────────────────────────────────── */
function Input({ label, hint, required, ...props }) {
  const [focused, setFocused] = useState(false);
  return (
    <div>
      <label style={{
        fontFamily:mono, fontSize:9, letterSpacing:1.5, textTransform:"uppercase",
        color:C.textMuted, marginBottom:5, display:"flex", gap:6, alignItems:"center",
      }}>
        {label}
        {hint && <span style={{ color:required?C.orange:C.textMuted, fontStyle:"normal" }}>({hint})</span>}
      </label>
      <input {...props}
        onFocus={e=>{ setFocused(true); props.onFocus?.(e); }}
        onBlur={e=>{ setFocused(false); props.onBlur?.(e); }}
        style={{
          width:"100%", background:C.surface3,
          border:`1px solid ${focused ? C.accent : C.border2}`,
          borderRadius:7, padding:"11px 14px", color:C.text,
          fontFamily:mono, fontSize:12, outline:"none",
          boxSizing:"border-box", transition:"border-color 0.2s, box-shadow 0.2s",
          boxShadow: focused ? `0 0 0 3px ${C.accentGlow}` : "none",
          ...props.style,
        }}
      />
    </div>
  );
}

/* ─── HEALTH INDICATOR ───────────────────────────────────────────── */
function HealthDot({ ok, label }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:5 }}>
      <div style={{
        width:6, height:6, borderRadius:"50%",
        background: ok ? C.green : C.red,
        boxShadow: ok ? `0 0 8px ${C.green}` : "none",
        animation: ok ? "pulse 2s infinite" : "none",
      }}/>
      <span style={{ fontFamily:mono, fontSize:9, color:ok?C.textDim:C.textMuted }}>
        {label} {ok?"✓":"✗"}
      </span>
    </div>
  );
}

/* ─── MAIN APP ───────────────────────────────────────────────────── */
export default function App() {
  const [form, setForm]     = useState({ chrom:"",position:"",ref:"",alt:"",ref_seq:"",alt_seq:"" });
  const [mode, setMode]     = useState("xgb");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError]   = useState(null);
  const [health, setHealth] = useState(null);
  const resultRef = useRef(null);

  const set = k => e => setForm(f=>({...f,[k]:e.target.value}));

  useEffect(() => {
    fetch(`${API}/health`).then(r=>r.json()).then(setHealth).catch(()=>{});
  }, []);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior:"smooth", block:"nearest" });
    }
  }, [result]);

  const run = async () => {
    const {chrom,position,ref,alt} = form;
    if (!chrom||!position||!ref||!alt) { setError("chrom, position, ref, alt are required."); return; }
    if (mode==="splice_sites" && !form.ref_seq) { setError("ref_seq required for splice site analysis."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const body = {
        chrom, position:parseInt(position),
        ref:ref.toUpperCase(), alt:alt.toUpperCase(),
        ref_seq:form.ref_seq||null, alt_seq:form.alt_seq||null,
      };
      const res = await fetch(`${API}/predict/${mode}`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify(body),
      });
      if (!res.ok) { const e=await res.json(); throw new Error(e.detail||`HTTP ${res.status}`); }
      const data = await res.json();
      setResult({...data, _mode:mode});
    } catch(e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        html,body{background:${C.bg};min-height:100vh;scrollbar-width:thin;scrollbar-color:${C.border2} ${C.bg}}
        ::selection{background:${C.accentGlow};color:${C.text}}
        input::placeholder{color:${C.textMuted};font-size:11px}
        button:hover{opacity:0.9}
        @keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
        @keyframes slideIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes scanline{0%{top:-10%}100%{top:110%}}
        ::-webkit-scrollbar{width:5px}
        ::-webkit-scrollbar-track{background:${C.bg}}
        ::-webkit-scrollbar-thumb{background:${C.border2};border-radius:2px}
      `}</style>

      <div style={{
        minHeight:"100vh", background:C.bg,
        backgroundImage:`
          radial-gradient(ellipse 60% 40% at 20% 0%, ${C.accentGlow} 0%, transparent 60%),
          radial-gradient(ellipse 40% 30% at 80% 10%, ${C.violet}18 0%, transparent 55%)
        `,
        display:"flex", justifyContent:"center",
        padding:"48px 20px 100px",
      }}>
        <div style={{ width:"100%", maxWidth:640 }}>

          {/* ── HEADER ── */}
          <div style={{ marginBottom:44, animation:"fadeIn 0.8s ease" }}>
            {/* status bar */}
            <div style={{
              display:"flex", alignItems:"center", gap:12, marginBottom:20,
              padding:"8px 14px", background:C.surface2,
              border:`1px solid ${C.border}`, borderRadius:8,
              flexWrap:"wrap",
            }}>
              <div style={{
                width:6, height:6, borderRadius:"50%", background:C.green,
                boxShadow:`0 0 10px ${C.green}`, animation:"pulse 2s infinite",
              }}/>
              <span style={{ fontFamily:mono, fontSize:9, letterSpacing:2, color:C.textMuted }}>
                VARIANT CLASSIFIER · v4.0
              </span>
              {health && (
                <div style={{ marginLeft:"auto", display:"flex", gap:14 }}>
                  <HealthDot ok={health.xgb} label="XGB"/>
                  <HealthDot ok={health.dnabert} label="DNABERT"/>
                  <span style={{ fontFamily:mono, fontSize:9, color:C.textMuted }}>
                    thresh {health.xgb_threshold?.toFixed(3)}
                  </span>
                  <span style={{ fontFamily:mono, fontSize:9, color:C.textMuted }}>
                    {health.device?.toUpperCase()}
                  </span>
                </div>
              )}
            </div>

            <h1 style={{
              fontFamily:sans, fontSize:52, fontWeight:700,
              color:C.text, letterSpacing:-2, lineHeight:1,
              marginBottom:12,
            }}>
              Splice<br/>
              <span style={{
                background:`linear-gradient(135deg, ${C.accent}, ${C.cyan})`,
                WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent",
              }}>Variant Lab</span>
            </h1>
            <p style={{ fontFamily:mono, fontSize:10, color:C.textMuted, lineHeight:2, letterSpacing:0.3 }}>
              XGBoost k-mer features · DNABERT-2-117M sequence encoding<br/>
              PWM donor/acceptor scoring · Cryptic site detection · Ensemble fusion
            </p>
          </div>

          {/* ── MODE SELECTOR ── */}
          <div style={{ marginBottom:26 }}>
            <div style={{ fontFamily:mono, fontSize:9, color:C.textMuted, letterSpacing:2, marginBottom:10 }}>
              ANALYSIS MODE
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:8 }}>
              {MODES.map(m=>(
                <button key={m.id} onClick={()=>setMode(m.id)} style={{
                  background: mode===m.id ? `${C.accent}18` : C.surface2,
                  border:`1px solid ${mode===m.id ? C.accent : C.border}`,
                  borderRadius:10, padding:"12px 8px", cursor:"pointer",
                  transition:"all 0.2s",
                  boxShadow: mode===m.id ? `0 0 24px ${C.accent}18` : "none",
                }}>
                  <div style={{
                    fontFamily:sans, fontWeight:600, fontSize:13,
                    color: mode===m.id ? C.accent : C.textDim,
                    marginBottom:3,
                  }}>{m.label}</div>
                  <div style={{ fontFamily:mono, fontSize:8, color:C.textMuted, lineHeight:1.6 }}>
                    {m.sub}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* ── FORM ── */}
          <div style={{
            background:C.surface2, border:`1px solid ${C.border}`,
            borderRadius:12, padding:"20px", marginBottom:16,
          }}>
            <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 2fr", gap:10 }}>
                <Input label="Chromosome" value={form.chrom} onChange={set("chrom")} placeholder="1"/>
                <Input label="Position" type="number" value={form.position} onChange={set("position")} placeholder="925952"/>
              </div>

              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
                <Input label="REF Allele" value={form.ref} onChange={set("ref")} placeholder="G" maxLength={50}/>
                <Input label="ALT Allele" value={form.alt} onChange={set("alt")} placeholder="A" maxLength={50}/>
              </div>

              <Input
                label="REF Sequence ±200bp"
                hint={mode==="splice_sites"?"required":"optional — improves accuracy"}
                required={mode==="splice_sites"}
                value={form.ref_seq}
                onChange={set("ref_seq")}
                placeholder="Genomic window around variant…"
                style={{ fontSize:10, letterSpacing:0.3 }}
              />

              <Input
                label="ALT Sequence"
                hint="auto-built from REF if blank"
                value={form.alt_seq}
                onChange={set("alt_seq")}
                placeholder="Leave blank to auto-compute…"
                style={{ fontSize:10, letterSpacing:0.3 }}
              />
            </div>

            {/* buttons */}
            <div style={{ display:"flex", gap:10, marginTop:16 }}>
              <button onClick={run} disabled={loading} style={{
                flex:1, padding:"14px 0",
                background: loading ? C.surface3
                  : `linear-gradient(135deg, #1d4ed8, #2563eb)`,
                border:`1px solid ${loading ? C.border : C.accent+"66"}`,
                borderRadius:9, color: loading ? C.textMuted : C.text,
                fontFamily:sans, fontWeight:700, fontSize:15,
                cursor: loading ? "not-allowed" : "pointer",
                boxShadow: loading ? "none" : `0 4px 24px ${C.accent}30`,
                display:"flex", alignItems:"center", justifyContent:"center", gap:10,
                transition:"all 0.2s",
              }}>
                {loading ? (
                  <>
                    <div style={{
                      width:14, height:14,
                      border:`2px solid ${C.accent}40`,
                      borderTopColor:C.accent,
                      borderRadius:"50%",
                      animation:"spin 0.6s linear infinite",
                    }}/>
                    Analysing…
                  </>
                ) : "Run Analysis →"}
              </button>
              <button onClick={()=>{ setForm({...EX,alt_seq:""}); setResult(null); setError(null); }} style={{
                padding:"14px 20px", background:C.surface3,
                border:`1px solid ${C.border2}`, borderRadius:9,
                color:C.textDim, fontFamily:mono, fontSize:10, cursor:"pointer",
                transition:"all 0.2s",
              }}>
                Example
              </button>
              {result && (
                <button onClick={()=>setResult(null)} style={{
                  padding:"14px 20px", background:C.surface3,
                  border:`1px solid ${C.border2}`, borderRadius:9,
                  color:C.textDim, fontFamily:mono, fontSize:10, cursor:"pointer",
                }}>
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* ── ERROR ── */}
          {error && (
            <div style={{
              marginBottom:16, padding:"13px 16px",
              background:"#1a080d", border:`1px solid ${C.red}50`,
              borderRadius:9, color:C.red,
              fontFamily:mono, fontSize:11, animation:"fadeIn 0.3s ease",
              display:"flex", gap:10, alignItems:"flex-start",
            }}>
              <span style={{ fontSize:13 }}>⚠</span>
              <span>{error}</span>
            </div>
          )}

          {/* ── RESULT ── */}
          {result && (
            <div ref={resultRef} style={{ animation:"slideUp 0.4s ease" }}>
              {result._mode === "splice_sites"
                ? <SpliceCard r={result} onClose={()=>setResult(null)}/>
                : <PredCard   r={result} onClose={()=>setResult(null)}/>}
            </div>
          )}

          {/* ── FOOTER ── */}
          <div style={{
            marginTop:56, textAlign:"center",
            fontFamily:mono, fontSize:8, color:C.border3, letterSpacing:1.5, lineHeight:2,
          }}>
            {API}<br/>
            XGBoost k-mer · DNABERT-2-117M · Position Weight Matrix · Cryptic Splice
          </div>
        </div>
      </div>
    </>
  );
}