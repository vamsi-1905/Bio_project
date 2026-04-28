import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8000";

/* ════════════════════════════════════════════════════════
   DESIGN: Clinical-lab noir. Think electron microscope
   meets Reuters terminal. Monochrome base with a single
   surgical accent. Every pixel earns its place.
   ════════════════════════════════════════════════════════ */

const T = {
  bg:      "#080a0f",
  paper:   "#0c0f17",
  lift:    "#111520",
  ridge:   "#171c28",
  wire:    "#1e2535",
  dim:     "#2a3248",
  muted:   "#4a5568",
  ghost:   "#64748b",
  soft:    "#94a3b8",
  text:    "#e2e8f0",
  bright:  "#f1f5f9",
  lime:    "#a3e635",     // single vivid accent — lab readout green
  limeDim: "#a3e63520",
  limeWire:"#a3e63550",
  red:     "#f87171",
  redDim:  "#f8717120",
  amber:   "#fbbf24",
  blue:    "#38bdf8",
  purple:  "#a78bfa",
};

const MODES = [
  { id:"xgb",          label:"XGBoost",      tag:"k-mer",    glyph:"▦" },
  { id:"dnabert",      label:"DNABERT-2",    tag:"sequence", glyph:"◈" },
  { id:"ensemble",     label:"Ensemble",     tag:"fusion",   glyph:"⊕" },
  { id:"splice_sites", label:"Splice Sites", tag:"PWM",      glyph:"⌁" },
];

const EX = {
  chrom:"1", position:"925952", ref:"G", alt:"A",
  ref_seq:"AGCTGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
};

/* ── fonts ─────────────────────────────────────────────────────── */
const DIS = "'Instrument Serif','Georgia',serif";
const MON = "'TX-02','Berkeley Mono','Fira Code','JetBrains Mono',monospace";
const SAN = "'Mona Sans','Geist','DM Sans',sans-serif";

/* ── helpers ───────────────────────────────────────────────────── */
const sigColor = s =>
  s?.startsWith("Strong") ? T.red :
  s?.startsWith("Moderate") ? T.amber :
  s?.startsWith("Weak")||s?.startsWith("Low") ? T.blue :
  T.lime;

const mutColor = t =>
  t==="SNV" ? T.blue : t==="Deletion" ? T.red : t==="Insertion" ? T.amber : T.purple;

/* ── animated number ───────────────────────────────────────────── */
function Num({ to, dur=800, dec=0 }) {
  const [v,sv] = useState(0);
  useEffect(() => {
    const s = Date.now();
    const tick = () => {
      const p = Math.min((Date.now()-s)/dur, 1);
      const e = 1 - Math.pow(1-p, 3);
      sv(to * e);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [to]);
  return <>{v.toFixed(dec)}</>;
}

/* ── scan-line bar ─────────────────────────────────────────────── */
function Bar({ val, max=1, color=T.lime, h=3, delay=0 }) {
  const [w, sw] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => sw((val/max)*100), 60 + delay);
    return () => clearTimeout(t);
  }, [val, max]);
  return (
    <div style={{ height:h, background:T.dim, borderRadius:1, overflow:"hidden" }}>
      <div style={{
        height:"100%", width:`${w}%`, background:color, borderRadius:1,
        transition:`width 0.9s cubic-bezier(.16,1,.3,1) ${delay}ms`,
        boxShadow:`0 0 6px ${color}66`,
      }}/>
    </div>
  );
}

/* ── chip/badge ────────────────────────────────────────────────── */
function Chip({ label, color=T.ghost }) {
  return (
    <span style={{
      display:"inline-block",
      fontFamily:MON, fontSize:9, letterSpacing:1.5,
      fontWeight:600, textTransform:"uppercase",
      padding:"3px 8px", borderRadius:2,
      background:`${color}18`, border:`1px solid ${color}55`,
      color, lineHeight:1,
    }}>{label}</span>
  );
}

/* ── section label ─────────────────────────────────────────────── */
function SectionLabel({ children }) {
  return (
    <div style={{
      fontFamily:MON, fontSize:8, letterSpacing:3,
      color:T.muted, textTransform:"uppercase", marginBottom:10,
      display:"flex", alignItems:"center", gap:8,
    }}>
      <div style={{ height:1, width:16, background:T.wire }}/>
      {children}
      <div style={{ height:1, flex:1, background:T.wire }}/>
    </div>
  );
}

/* ── big gauge ─────────────────────────────────────────────────── */
function Gauge({ prob, isPath }) {
  const r = 44, c = 2*Math.PI*r;
  const color = isPath ? T.red : T.lime;
  return (
    <div style={{ position:"relative", width:108, height:108, flexShrink:0 }}>
      {/* tick marks */}
      <svg width={108} height={108} viewBox="0 0 108 108"
           style={{ position:"absolute", inset:0, transform:"rotate(-90deg)" }}>
        {/* background arc */}
        <circle cx={54} cy={54} r={r} fill="none" stroke={T.dim} strokeWidth={7}/>
        {/* progress arc */}
        <circle cx={54} cy={54} r={r} fill="none" stroke={color} strokeWidth={7}
          strokeDasharray={c}
          strokeDashoffset={c*(1-prob)}
          strokeLinecap="butt"
          style={{ transition:"stroke-dashoffset 1.1s cubic-bezier(.16,1,.3,1)" }}/>
      </svg>
      {/* centre */}
      <div style={{
        position:"absolute", inset:0,
        display:"flex", flexDirection:"column",
        alignItems:"center", justifyContent:"center",
      }}>
        <span style={{
          fontFamily:MON, fontWeight:700, fontSize:20,
          color, lineHeight:1,
          textShadow:`0 0 16px ${color}`,
        }}>
          <Num to={Math.round(prob*100)} dec={0}/>%
        </span>
        <span style={{ fontFamily:MON, fontSize:7, color:T.muted, letterSpacing:2, marginTop:3 }}>
          PROB
        </span>
      </div>
    </div>
  );
}

/* ── delta chip ────────────────────────────────────────────────── */
function Delta({ v }) {
  const up = v >= 0;
  const col = up ? T.lime : T.red;
  return (
    <span style={{
      fontFamily:MON, fontSize:10, fontWeight:700, color:col,
      background:`${col}15`, border:`1px solid ${col}44`,
      borderRadius:2, padding:"2px 7px",
    }}>{up?"+":""}{v.toFixed(3)}</span>
  );
}

/* ── kmer visual ────────────────────────────────────────────────── */
function KmerView({ ref_kmer, alt_kmer }) {
  if (!ref_kmer && !alt_kmer) return null;
  const render = (seq, isAlt) => {
    if (!seq) return <span style={{ color:T.muted, fontFamily:MON, fontSize:11 }}>—</span>;
    return (
      <span style={{ fontFamily:MON, fontSize:11, letterSpacing:2 }}>
        {seq.split("").map((c,i) => {
          const changed = ref_kmer && alt_kmer && ref_kmer[i] !== alt_kmer[i];
          return (
            <span key={i} style={{
              color: changed && isAlt ? T.amber : T.soft,
              background: changed && isAlt ? `${T.amber}20` : "transparent",
              borderRadius:1,
            }}>{c}</span>
          );
        })}
      </span>
    );
  };
  return (
    <div style={{
      marginTop:10, background:T.bg, borderRadius:4, padding:"10px 14px",
      border:`1px solid ${T.wire}`,
    }}>
      {[["REF", ref_kmer, false],["ALT", alt_kmer, true]].map(([lbl, seq, isAlt]) => (
        <div key={lbl} style={{ display:"flex", gap:12, alignItems:"center", marginBottom: lbl==="REF"?5:0 }}>
          <span style={{ fontFamily:MON, fontSize:8, color:T.muted, width:24, letterSpacing:1 }}>{lbl}</span>
          {render(seq, isAlt)}
        </div>
      ))}
    </div>
  );
}

/* ── site row ───────────────────────────────────────────────────── */
function SiteRow({ s, idx }) {
  const [open, setOpen] = useState(false);
  const tag  = s.disrupted?"DISRUPTED":s.created?"CREATED":s.cryptic?"CRYPTIC":"SHIFTED";
  const col  = s.disrupted?T.red:s.created?T.lime:s.cryptic?T.amber:T.muted;
  const pos  = `${s.position>=0?"+":""}${s.position}`;

  return (
    <div style={{
      background:T.paper,
      border:`1px solid ${open ? col+"44" : T.wire}`,
      borderRadius:4,
      overflow:"hidden",
      animation:`fadeUp 0.25s ${idx*35}ms both`,
      transition:"border-color 0.2s",
    }}>
      {/* collapsed row */}
      <div onClick={() => setOpen(v=>!v)} style={{
        padding:"11px 16px", cursor:"pointer",
        display:"grid", gridTemplateColumns:"80px 1fr 140px 32px",
        gap:12, alignItems:"center",
      }}>
        <div>
          <Chip label={tag} color={col}/>
          <div style={{ fontFamily:MON, fontSize:8, color:T.muted, marginTop:4, letterSpacing:0.5 }}>
            {s.type} · {pos}
          </div>
        </div>

        {/* score bars */}
        <div>
          {[["REF", s.ref_score, T.ghost], ["ALT", s.alt_score, col]].map(([l,v,c]) => (
            <div key={l} style={{ display:"flex", gap:8, alignItems:"center", marginBottom:l==="REF"?4:0 }}>
              <span style={{ fontFamily:MON, fontSize:8, color:T.muted, width:20 }}>{l}</span>
              <div style={{ flex:1 }}>
                <Bar val={v} color={c} h={2}/>
              </div>
              <span style={{ fontFamily:MON, fontSize:9, color:c, width:32, textAlign:"right" }}>
                {v.toFixed(2)}
              </span>
            </div>
          ))}
        </div>

        <div style={{ display:"flex", justifyContent:"flex-end" }}>
          <Delta v={s.delta}/>
        </div>

        <span style={{ fontFamily:MON, fontSize:10, color:T.muted, textAlign:"center" }}>
          {open ? "▲" : "▼"}
        </span>
      </div>

      {/* expanded */}
      {open && (
        <div style={{
          borderTop:`1px solid ${T.wire}`,
          padding:"12px 16px",
          animation:"fadeIn 0.18s ease",
        }}>
          <KmerView ref_kmer={s.ref_kmer} alt_kmer={s.alt_kmer}/>
          <div style={{
            marginTop:10, padding:"12px 14px",
            background:T.bg, borderRadius:4,
            borderLeft:`2px solid ${col}`,
          }}>
            <SectionLabel>Molecular reasoning</SectionLabel>
            <p style={{ fontFamily:MON, fontSize:11, color:T.soft, lineHeight:1.9, margin:0 }}>
              {s.reasoning}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── heatmap landscape ─────────────────────────────────────────── */
function Landscape({ sites }) {
  if (!sites?.length) return null;
  const track = (items, label, color) => {
    if (!items.length) return null;
    const minP = Math.min(...items.map(s=>s.position));
    const maxP = Math.max(...items.map(s=>s.position));
    const span = Math.max(maxP - minP, 1);
    return (
      <div style={{ marginBottom:14 }}>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
          <span style={{ fontFamily:MON, fontSize:8, color:T.muted, letterSpacing:1.5 }}>{label.toUpperCase()} ({items.length})</span>
          <div style={{ display:"flex", gap:10 }}>
            {[["DISRUPTED",T.red],["CREATED",T.lime],["CRYPTIC",T.amber]].map(([l,c])=>(
              <span key={l} style={{ display:"flex", alignItems:"center", gap:4, fontFamily:MON, fontSize:7, color:T.muted }}>
                <span style={{ display:"inline-block", width:6, height:6, background:c, borderRadius:1 }}/>
                {l}
              </span>
            ))}
          </div>
        </div>
        <div style={{
          position:"relative", height:32, background:T.bg,
          borderRadius:3, border:`1px solid ${T.wire}`,
          overflow:"hidden",
        }}>
          {/* center line */}
          <div style={{
            position:"absolute", left:"50%", top:0,
            width:1, height:"100%", background:T.wire,
          }}/>
          {items.map((s,i) => {
            const xPct = span > 0 ? ((s.position-minP)/span)*88+6 : 50;
            const c    = s.disrupted?T.red:s.created?T.lime:s.cryptic?T.amber:T.muted;
            const ht   = Math.max(s.alt_score*30, 4);
            return (
              <div key={i} title={`${s.type} pos ${s.position>=0?"+":""}${s.position}  Δ${s.delta.toFixed(3)}`}
                style={{
                  position:"absolute", bottom:0, left:`${xPct}%`,
                  width:3, height:`${ht}px`,
                  background:c, borderRadius:"1px 1px 0 0",
                  transform:"translateX(-50%)",
                  boxShadow:`0 0 5px ${c}88`,
                }}/>
            );
          })}
        </div>
        <div style={{ display:"flex", justifyContent:"space-between", marginTop:3 }}>
          <span style={{ fontFamily:MON, fontSize:7, color:T.muted }}>{minP>=0?"+":""}{minP}bp</span>
          <span style={{ fontFamily:MON, fontSize:7, color:T.muted }}>0 (variant)</span>
          <span style={{ fontFamily:MON, fontSize:7, color:T.muted }}>{maxP>=0?"+":""}{maxP}bp</span>
        </div>
      </div>
    );
  };

  return (
    <div style={{
      padding:"14px 16px", background:T.bg,
      borderRadius:6, border:`1px solid ${T.wire}`,
      marginBottom:16,
    }}>
      <SectionLabel>Splice Site Landscape</SectionLabel>
      {track(sites.filter(s=>s.type==="donor"), "Donor", T.blue)}
      {track(sites.filter(s=>s.type==="acceptor"), "Acceptor", T.purple)}
    </div>
  );
}

/* ── feature rules ─────────────────────────────────────────────── */
function Rules({ r }) {
  const ref = r.ref?.toUpperCase() || "";
  const alt = r.alt?.toUpperCase() || "";
  const mut = r.mutation_type || "";
  const rules = [
    { label:"+1G invariant donor",  fired:ref==="G", sev: ref==="G"&&alt!=="G"?"high":"low",
      desc: ref==="G" ? (alt!=="G"?"REF=G → ALT disrupts invariant +1G of GT donor dinucleotide. >99% conserved; virtually always pathogenic when substituted.":"REF=G preserved in ALT — no disruption at +1G donor position.") : "REF is not G; +1G canonical donor position not directly affected." },
    { label:"+2T invariant donor",  fired:ref==="T", sev: ref==="T"&&alt!=="T"?"high":"low",
      desc: ref==="T" ? (alt!=="T"?"REF=T → ALT disrupts +2T of GT donor. Highly conserved; most substitutions abolish splicing.":"REF=T preserved in ALT.") : "REF is not T; +2T donor position not directly affected." },
    { label:"-2A invariant acceptor", fired:ref==="A", sev: ref==="A"&&alt!=="A"?"high":"low",
      desc: ref==="A" ? (alt!=="A"?"REF=A → ALT disrupts -2A of AG acceptor dinucleotide. Loss typically causes exon skipping.":"REF=A preserved.") : "REF is not A; -2A acceptor not directly affected." },
    { label:"GT dinucleotide loss", fired:ref.includes("GT")&&!alt.includes("GT"), sev:"high",
      desc: ref.includes("GT") ? (alt.includes("GT")?"GT preserved in ALT; donor dinucleotide intact.":"GT present in REF but lost in ALT — potential donor site disruption.") : "GT motif not present in REF allele." },
    { label:"AG dinucleotide loss", fired:ref.includes("AG")&&!alt.includes("AG"), sev:"high",
      desc: ref.includes("AG") ? (alt.includes("AG")?"AG preserved in ALT; acceptor dinucleotide intact.":"AG present in REF but lost in ALT — potential acceptor disruption.") : "AG motif not present in REF allele." },
    { label:"Frameshift risk", fired:(mut==="Insertion"||mut==="Deletion")&&(Math.abs(ref.length-alt.length)%3!==0), sev:"medium",
      desc: (mut==="Insertion"||mut==="Deletion") ? (Math.abs(ref.length-alt.length)%3!==0 ? `${Math.abs(ref.length-alt.length)}bp frameshift — disrupts reading frame and splice geometry downstream.`:`In-frame indel (${Math.abs(ref.length-alt.length)}bp) — lower frameshift risk but may alter local splice signals.`) : "SNV/MNV — no length change." },
    { label:"Multi-nucleotide substitution", fired:mut==="MNV", sev:"medium",
      desc: mut==="MNV" ? "Complex MNV — multiple consecutive positions affected. High risk of disrupting conserved splice site motifs simultaneously." : "Not a multi-nucleotide variant." },
  ];

  const sevColor = s => s==="high"?T.red:s==="medium"?T.amber:T.wire;

  return (
    <div style={{ marginTop:2 }}>
      <SectionLabel>Biological Rule Evaluation</SectionLabel>
      <p style={{ fontFamily:MON, fontSize:9, color:T.muted, lineHeight:1.7, marginBottom:12 }}>
        Rules derived from conserved splice site biology. Scores come from ML models, not these rules.
      </p>
      {rules.map((f,i) => (
        <div key={i} style={{
          marginBottom:6, padding:"10px 12px",
          background: f.fired ? `${sevColor(f.sev)}10` : T.bg,
          border:`1px solid ${f.fired ? sevColor(f.sev)+"55" : T.wire}`,
          borderRadius:4,
          transition:"all 0.2s",
        }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:5 }}>
            <span style={{ fontFamily:MON, fontSize:10, color:f.fired?sevColor(f.sev):T.muted }}>
              {f.fired ? "● " : "○ "}{f.label}
            </span>
            {f.fired && <Chip label={f.sev} color={sevColor(f.sev)}/>}
          </div>
          <p style={{ fontFamily:MON, fontSize:10, color:T.muted, lineHeight:1.7, margin:0 }}>{f.desc}</p>
        </div>
      ))}
    </div>
  );
}

/* ── confidence breakdown ──────────────────────────────────────── */
function Confidence({ prob, thresh }) {
  const margin = Math.abs(prob - thresh);
  const certLabel = margin>0.35?"High":margin>0.15?"Medium":"Low";
  const certColor = margin>0.35?T.lime:margin>0.15?T.amber:T.red;
  const isPath = prob >= thresh;

  return (
    <div style={{ marginTop:2 }}>
      <SectionLabel>Prediction Confidence</SectionLabel>
      {/* main confidence tile */}
      <div style={{
        display:"flex", justifyContent:"space-between", alignItems:"center",
        padding:"16px 18px",
        background:T.bg, borderRadius:6,
        border:`1px solid ${certColor}40`,
        marginBottom:12,
      }}>
        <div>
          <div style={{ fontFamily:MON, fontSize:8, color:T.muted, letterSpacing:2, marginBottom:4 }}>CONFIDENCE</div>
          <div style={{ fontFamily:DIS, fontSize:28, color:certColor, lineHeight:1 }}>{certLabel}</div>
        </div>
        <div style={{ textAlign:"right" }}>
          <div style={{ fontFamily:MON, fontSize:8, color:T.muted, letterSpacing:2, marginBottom:4 }}>
            {isPath?"PATHOGENIC":"BENIGN"}
          </div>
          <div style={{ fontFamily:MON, fontSize:24, fontWeight:700, color:isPath?T.red:T.lime }}>
            {(prob*100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* margin bar */}
      <div style={{ marginBottom:12 }}>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
          <span style={{ fontFamily:MON, fontSize:9, color:T.muted }}>Distance from decision boundary</span>
          <span style={{ fontFamily:MON, fontSize:9, color:T.blue }}>{(margin*100).toFixed(1)}pp</span>
        </div>
        <Bar val={margin} max={0.5} color={T.blue} h={4}/>
        <p style={{ fontFamily:MON, fontSize:9, color:T.muted, lineHeight:1.7, marginTop:6 }}>
          {margin < 0.1 ? "Near-boundary — model is uncertain. Treat with caution and run splice site analysis."
           : margin < 0.25 ? "Moderate margin. Consider ensemble and splice site analysis for confirmation."
           : "Strong separation from threshold — high model confidence in this call."}
        </p>
      </div>

      {/* disclaimer */}
      <div style={{
        padding:"10px 12px", background:T.paper, borderRadius:4,
        border:`1px solid ${T.wire}`,
        fontFamily:MON, fontSize:9, color:T.muted, lineHeight:1.8,
      }}>
        ℹ Confidence reflects margin from tuned threshold, not model calibration. For clinical decisions
        validate with ClinVar/ACMG criteria and MaxEntScan/SpliceAI.
      </div>
    </div>
  );
}

/* ── var grid ──────────────────────────────────────────────────── */
function VarGrid({ r }) {
  const rows = [
    ["CHR",     r.chrom,                       T.blue],
    ["POS",     r.position?.toLocaleString(),   T.soft],
    ["REF",     r.ref,                          T.lime],
    ["ALT",     r.alt,                          T.amber],
    ["TYPE",    r.mutation_type,                mutColor(r.mutation_type)],
    ["TIME",    `${r.latency_ms}ms`,            T.muted],
  ];
  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:6 }}>
      {rows.map(([k,v,c])=>(
        <div key={k} style={{
          background:T.bg, borderRadius:4, padding:"9px 12px",
          border:`1px solid ${T.wire}`,
        }}>
          <div style={{ fontFamily:MON, fontSize:7, color:T.muted, letterSpacing:1.5, marginBottom:3 }}>{k}</div>
          <div style={{ fontFamily:MON, fontSize:12, fontWeight:700, color:c }}>{v}</div>
        </div>
      ))}
    </div>
  );
}

/* ── PREDICTION CARD ───────────────────────────────────────────── */
function PredCard({ r, onClose }) {
  const isPath = r.prediction === "Pathogenic";
  const color  = isPath ? T.red : T.lime;
  const [tab,  setTab] = useState("overview");
  const TABS   = ["overview","rules","confidence"];

  return (
    <div style={{
      background:T.paper, border:`1px solid ${color}33`,
      borderRadius:8, overflow:"hidden",
      animation:"slideUp 0.45s cubic-bezier(.16,1,.3,1)",
      boxShadow:`0 0 60px ${color}0a, 0 0 120px ${color}05`,
    }}>
      {/* ── top bar ── */}
      <div style={{
        background:`linear-gradient(90deg, ${color}18 0%, transparent 70%)`,
        borderBottom:`1px solid ${color}22`,
        padding:"18px 22px",
        display:"flex", alignItems:"center", gap:16,
      }}>
        {/* verdict */}
        <div style={{ flex:1 }}>
          <div style={{ fontFamily:MON, fontSize:8, color:T.muted, letterSpacing:3, marginBottom:4 }}>
            {r.model?.toUpperCase()} · {r.latency_ms}ms
          </div>
          <div style={{ display:"flex", alignItems:"baseline", gap:12 }}>
            <span style={{ fontFamily:DIS, fontSize:32, color, lineHeight:1 }}>
              {r.prediction}
            </span>
            <Chip label={r.confidence} color={color}/>
          </div>
        </div>
        {/* gauge */}
        <Gauge prob={r.probability} isPath={isPath}/>
        {/* close */}
        <button onClick={onClose} style={{
          background:"none", border:`1px solid ${T.wire}`,
          borderRadius:4, color:T.muted, cursor:"pointer",
          width:28, height:28, fontFamily:MON, fontSize:12,
          alignSelf:"flex-start",
        }}>✕</button>
      </div>

      <div style={{ padding:"20px 22px" }}>
        {/* probability bar */}
        <div style={{ marginBottom:18 }}>
          <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
            <span style={{ fontFamily:MON, fontSize:8, color:T.lime, letterSpacing:1 }}>BENIGN</span>
            <span style={{ fontFamily:MON, fontSize:8, color:T.muted }}>
              threshold {(r.threshold_used*100).toFixed(0)}%
            </span>
            <span style={{ fontFamily:MON, fontSize:8, color:T.red, letterSpacing:1 }}>PATHOGENIC</span>
          </div>
          <div style={{ position:"relative", height:6, background:T.dim, borderRadius:1 }}>
            <div style={{
              position:"absolute", left:0, top:0, height:"100%",
              width:`${r.probability*100}%`, borderRadius:1,
              background:`linear-gradient(90deg, ${color}80, ${color})`,
              transition:"width 0.9s cubic-bezier(.16,1,.3,1)",
              boxShadow:`0 0 10px ${color}66`,
            }}/>
            {/* threshold tick */}
            <div style={{
              position:"absolute", top:-4, transform:"translateX(-50%)",
              left:`${r.threshold_used*100}%`,
              width:1, height:14, background:T.amber,
              boxShadow:`0 0 6px ${T.amber}`,
            }}/>
          </div>
        </div>

        {/* mutation badges */}
        <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:16 }}>
          <Chip label={r.mutation_type} color={mutColor(r.mutation_type)}/>
          <Chip label={`chr${r.chrom}:${r.position} ${r.ref}→${r.alt}`} color={T.ghost}/>
        </div>

        {/* tabs */}
        <div style={{
          display:"flex", gap:0, borderBottom:`1px solid ${T.wire}`,
          marginBottom:16,
        }}>
          {TABS.map(t => (
            <button key={t} onClick={()=>setTab(t)} style={{
              background:"none", border:"none", cursor:"pointer",
              fontFamily:MON, fontSize:9, letterSpacing:1.5, textTransform:"uppercase",
              color: tab===t ? color : T.muted,
              borderBottom: tab===t ? `1px solid ${color}` : "1px solid transparent",
              padding:"7px 16px", marginBottom:-1, transition:"color 0.2s",
            }}>{t}</button>
          ))}
        </div>

        {tab==="overview" && (
          <div>
            {/* mechanism */}
            <div style={{
              padding:"12px 14px", background:T.bg,
              borderRadius:4, borderLeft:`2px solid ${color}`,
              marginBottom:14,
            }}>
              <SectionLabel>Molecular mechanism</SectionLabel>
              <p style={{ fontFamily:MON, fontSize:11, color:T.soft, lineHeight:1.9, margin:0 }}>
                {r.mechanism}
              </p>
            </div>
            <VarGrid r={r}/>
          </div>
        )}
        {tab==="rules"      && <Rules r={r}/>}
        {tab==="confidence" && <Confidence prob={r.probability} thresh={r.threshold_used}/>}
      </div>
    </div>
  );
}

/* ── stat pill ─────────────────────────────────────────────────── */
function Pill({ label, val, color }) {
  const active = val > 0;
  return (
    <div style={{
      flex:1, minWidth:80, textAlign:"center",
      padding:"10px 10px", borderRadius:4,
      background: active ? `${color}12` : T.bg,
      border:`1px solid ${active ? color+"44" : T.wire}`,
    }}>
      <div style={{ fontFamily:MON, fontSize:20, fontWeight:700, color:active?color:T.muted }}>{val}</div>
      <div style={{ fontFamily:MON, fontSize:7, color:T.muted, marginTop:3, lineHeight:1.5, letterSpacing:0.5 }}>{label}</div>
    </div>
  );
}

/* ── SPLICE CARD ───────────────────────────────────────────────── */
function SpliceCard({ r, onClose }) {
  const sc = sigColor(r.pathogenicity_signal);
  const [showAll, setShowAll] = useState(false);
  const [tab, setTab] = useState("sites");
  const visible = showAll ? r.sites : r.sites?.slice(0,6) || [];

  return (
    <div style={{
      background:T.paper, border:`1px solid ${sc}33`,
      borderRadius:8, overflow:"hidden",
      animation:"slideUp 0.45s cubic-bezier(.16,1,.3,1)",
      boxShadow:`0 0 60px ${sc}08`,
    }}>
      {/* header */}
      <div style={{
        background:`linear-gradient(90deg, ${sc}15 0%, transparent 70%)`,
        borderBottom:`1px solid ${sc}22`,
        padding:"18px 22px",
        display:"flex", alignItems:"flex-start", gap:12,
      }}>
        <div style={{ flex:1 }}>
          <div style={{ fontFamily:MON, fontSize:8, color:T.muted, letterSpacing:3, marginBottom:5 }}>
            SPLICE SITE ANALYSIS · {r.latency_ms}ms · {r.sites_found} sites scored
          </div>
          <div style={{ fontFamily:DIS, fontSize:22, color:sc, lineHeight:1.2, marginBottom:5 }}>
            {r.pathogenicity_signal}
          </div>
          <div style={{ fontFamily:MON, fontSize:10, color:T.muted }}>
            {r.summary}
          </div>
        </div>
        <button onClick={onClose} style={{
          background:"none", border:`1px solid ${T.wire}`, borderRadius:4,
          color:T.muted, cursor:"pointer", width:28, height:28,
          fontFamily:MON, fontSize:12, flexShrink:0,
        }}>✕</button>
      </div>

      <div style={{ padding:"20px 22px" }}>
        {/* mutation box */}
        <div style={{
          padding:"12px 14px", background:T.bg, borderRadius:4,
          borderLeft:`2px solid ${mutColor(r.mutation_type)}`,
          marginBottom:16,
        }}>
          <div style={{ display:"flex", gap:8, alignItems:"center", marginBottom:6 }}>
            <Chip label={r.mutation_type} color={mutColor(r.mutation_type)}/>
            <span style={{ fontFamily:MON, fontSize:10, color:T.muted }}>{r.mutation_detail}</span>
          </div>
          <p style={{ fontFamily:MON, fontSize:11, color:T.soft, lineHeight:1.9, margin:0 }}>
            {r.mutation_mechanism}
          </p>
        </div>

        {/* stat pills */}
        <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginBottom:16 }}>
          <Pill label="Disrupted Donors"    val={r.disrupted_donors}    color={T.red}/>
          <Pill label="Disrupted Acceptors" val={r.disrupted_acceptors} color={T.red}/>
          <Pill label="New Donors"          val={r.created_donors}      color={T.lime}/>
          <Pill label="New Acceptors"       val={r.created_acceptors}   color={T.lime}/>
          <Pill label="Cryptic Donors"      val={r.cryptic_donors}      color={T.amber}/>
          <Pill label="Cryptic Acceptors"   val={r.cryptic_acceptors}   color={T.amber}/>
        </div>

        {/* max change bar */}
        {r.max_disruption > 0 && (
          <div style={{ marginBottom:16 }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
              <span style={{ fontFamily:MON, fontSize:8, color:T.muted, letterSpacing:1 }}>MAX SCORE CHANGE</span>
              <span style={{ fontFamily:MON, fontSize:9, fontWeight:700, color:sc }}>
                {(r.max_disruption*100).toFixed(1)}%
              </span>
            </div>
            <Bar val={r.max_disruption} color={sc} h={5}/>
          </div>
        )}

        {/* landscape */}
        {r.sites?.length > 0 && <Landscape sites={r.sites}/>}

        {/* tabs */}
        <div style={{ display:"flex", borderBottom:`1px solid ${T.wire}`, marginBottom:14 }}>
          {["sites","variant"].map(t => (
            <button key={t} onClick={()=>setTab(t)} style={{
              background:"none", border:"none", cursor:"pointer",
              fontFamily:MON, fontSize:9, letterSpacing:1.5, textTransform:"uppercase",
              color: tab===t ? sc : T.muted,
              borderBottom: tab===t ? `1px solid ${sc}` : "1px solid transparent",
              padding:"7px 16px", marginBottom:-1, transition:"color 0.2s",
            }}>{t}</button>
          ))}
        </div>

        {tab==="sites" && (
          <div>
            {visible.length > 0 ? (
              <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                {visible.map((s,i) => <SiteRow key={i} s={s} idx={i}/>)}
              </div>
            ) : (
              <div style={{
                fontFamily:MON, fontSize:11, color:T.muted, textAlign:"center",
                padding:32, border:`1px dashed ${T.wire}`, borderRadius:6,
              }}>
                No significant splice site changes detected
              </div>
            )}
            {r.sites?.length > 6 && (
              <button onClick={()=>setShowAll(v=>!v)} style={{
                marginTop:10, width:"100%",
                background:"none", border:`1px solid ${T.wire}`,
                borderRadius:4, color:T.muted,
                fontFamily:MON, fontSize:9, padding:10,
                cursor:"pointer",
              }}>
                {showAll ? "Show less ▲" : `Show all ${r.sites.length} sites ▼`}
              </button>
            )}
          </div>
        )}

        {tab==="variant" && (
          <div style={{ marginTop:4 }}>
            <VarGrid r={{
              ...r,
              mutation_type: r.mutation_type,
              latency_ms: r.latency_ms,
            }}/>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── input ─────────────────────────────────────────────────────── */
function Input({ label, hint, required, mono: isMono, ...props }) {
  const [focused, setF] = useState(false);
  return (
    <div>
      <label style={{
        fontFamily:MON, fontSize:8, letterSpacing:2, textTransform:"uppercase",
        color:T.muted, marginBottom:5, display:"flex", gap:8, alignItems:"center",
      }}>
        {label}
        {hint && (
          <span style={{ color:required?T.amber:T.muted, textTransform:"none", letterSpacing:0 }}>
            ({hint})
          </span>
        )}
      </label>
      <input {...props}
        onFocus={e=>{ setF(true); props.onFocus?.(e); }}
        onBlur={e=>{ setF(false); props.onBlur?.(e); }}
        style={{
          width:"100%", background:T.lift,
          border:`1px solid ${focused ? T.lime+"66" : T.wire}`,
          borderRadius:4, padding:"10px 13px", color:T.text,
          fontFamily: isMono ? MON : MON, fontSize:12,
          outline:"none", boxSizing:"border-box",
          transition:"border-color 0.15s, box-shadow 0.15s",
          boxShadow: focused ? `0 0 0 3px ${T.limeDim}` : "none",
          ...props.style,
        }}
      />
    </div>
  );
}

/* ── health dot ────────────────────────────────────────────────── */
function HDot({ ok, label }) {
  return (
    <span style={{ display:"flex", alignItems:"center", gap:5 }}>
      <span style={{
        width:5, height:5, borderRadius:"50%",
        display:"inline-block",
        background:ok?T.lime:T.red,
        boxShadow:ok?`0 0 6px ${T.lime}`:"none",
      }}/>
      <span style={{ fontFamily:MON, fontSize:8, color:ok?T.ghost:T.muted }}>
        {label}
      </span>
    </span>
  );
}

/* ════════════════════════════════════════════════════════
   MAIN APP
   ════════════════════════════════════════════════════════ */
export default function App() {
  const [form,    setForm]    = useState({ chrom:"",position:"",ref:"",alt:"",ref_seq:"",alt_seq:"" });
  const [mode,    setMode]    = useState("xgb");
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);
  const [health,  setHealth]  = useState(null);
  const resultRef = useRef(null);

  const set = k => e => setForm(f=>({...f,[k]:e.target.value}));

  useEffect(() => {
    fetch(`${API}/health`).then(r=>r.json()).then(setHealth).catch(()=>{});
  }, []);

  useEffect(() => {
    if (result && resultRef.current)
      resultRef.current.scrollIntoView({ behavior:"smooth", block:"nearest" });
  }, [result]);

  const run = async () => {
    const { chrom, position, ref, alt } = form;
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
      setResult({...await res.json(), _mode:mode});
    } catch(e) { setError(e.message); }
    finally    { setLoading(false); }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Mona+Sans:wght@300;400;600&family=JetBrains+Mono:wght@400;500;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        html,body{background:${T.bg};min-height:100vh;color:${T.text}}
        input::placeholder{color:${T.muted};font-size:11px}
        button{transition:opacity 0.15s} button:hover{opacity:0.82}
        @keyframes slideUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
        @keyframes spin{to{transform:rotate(360deg)}}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:${T.bg}}
        ::-webkit-scrollbar-thumb{background:${T.wire};border-radius:2px}
      `}</style>

      {/* subtle grid background */}
      <div style={{
        position:"fixed", inset:0, pointerEvents:"none", zIndex:0,
        backgroundImage:`
          linear-gradient(${T.wire}40 1px, transparent 1px),
          linear-gradient(90deg, ${T.wire}40 1px, transparent 1px)
        `,
        backgroundSize:"60px 60px",
        maskImage:"radial-gradient(ellipse 80% 60% at 50% 0%, black 30%, transparent 100%)",
        WebkitMaskImage:"radial-gradient(ellipse 80% 60% at 50% 0%, black 30%, transparent 100%)",
      }}/>

      {/* top glow */}
      <div style={{
        position:"fixed", top:0, left:"30%", right:"30%", height:1,
        background:`linear-gradient(90deg, transparent, ${T.lime}60, transparent)`,
        zIndex:1,
      }}/>

      <div style={{
        position:"relative", zIndex:2,
        minHeight:"100vh", display:"flex", justifyContent:"center",
        padding:"56px 20px 100px",
      }}>
        <div style={{ width:"100%", maxWidth:620 }}>

          {/* ── STATUS BAR ── */}
          <div style={{
            display:"flex", alignItems:"center", gap:14, marginBottom:40,
            padding:"7px 14px", background:T.paper,
            border:`1px solid ${T.wire}`, borderRadius:4,
          }}>
            <div style={{
              width:5, height:5, borderRadius:"50%", background:T.lime,
              boxShadow:`0 0 8px ${T.lime}`, animation:"pulse 2.5s infinite",
            }}/>
            <span style={{ fontFamily:MON, fontSize:8, letterSpacing:3, color:T.muted }}>
              SPLICE VARIANT LAB · v4.0
            </span>
            {health && (
              <div style={{ marginLeft:"auto", display:"flex", gap:14, alignItems:"center" }}>
                <HDot ok={health.xgb}     label="XGB"/>
                <HDot ok={health.dnabert} label="DNA"/>
                <span style={{ fontFamily:MON, fontSize:7, color:T.muted }}>
                  τ={health.xgb_threshold?.toFixed(3)}
                </span>
                <span style={{ fontFamily:MON, fontSize:7, color:T.muted, textTransform:"uppercase" }}>
                  {health.device}
                </span>
              </div>
            )}
          </div>

          {/* ── HEADING ── */}
          <div style={{ marginBottom:48 }}>
            <h1 style={{
              fontFamily:DIS, fontSize:62, fontWeight:400, italic:true,
              color:T.bright, letterSpacing:-1, lineHeight:0.95,
              marginBottom:14,
            }}>
              <span style={{ fontStyle:"italic" }}>Splice</span><br/>
              <span style={{
                WebkitBackgroundClip:"text",
                WebkitTextFillColor:"transparent",
                backgroundImage:`linear-gradient(135deg, ${T.lime}, ${T.blue})`,
                backgroundClip:"text",
              }}>Variant Lab</span>
            </h1>
            <p style={{
              fontFamily:MON, fontSize:9, color:T.muted,
              lineHeight:2.1, letterSpacing:0.5,
            }}>
              XGBoost k-mer features · DNABERT-2-117M sequence encoding<br/>
              PWM donor/acceptor scoring · Cryptic splice site detection
            </p>
          </div>

          {/* ── MODE SELECTOR ── */}
          <div style={{ marginBottom:24 }}>
            <div style={{ fontFamily:MON, fontSize:8, color:T.muted, letterSpacing:3, marginBottom:10 }}>
              ANALYSIS MODE
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:7 }}>
              {MODES.map(m => (
                <button key={m.id} onClick={()=>setMode(m.id)} style={{
                  background: mode===m.id ? T.lift : T.paper,
                  border:`1px solid ${mode===m.id ? T.lime+"66" : T.wire}`,
                  borderRadius:5, padding:"12px 8px", cursor:"pointer",
                  boxShadow: mode===m.id ? `0 0 20px ${T.lime}0f, inset 0 0 0 1px ${T.lime}20` : "none",
                  transition:"all 0.2s",
                  textAlign:"left",
                }}>
                  <div style={{
                    fontFamily:MON, fontSize:14, color:mode===m.id?T.lime:T.muted,
                    marginBottom:4, lineHeight:1,
                  }}>{m.glyph}</div>
                  <div style={{
                    fontFamily:SAN, fontWeight:600, fontSize:12,
                    color:mode===m.id?T.bright:T.soft,
                    marginBottom:2,
                  }}>{m.label}</div>
                  <div style={{ fontFamily:MON, fontSize:8, color:T.muted, letterSpacing:0.5 }}>
                    {m.tag}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* ── FORM ── */}
          <div style={{
            background:T.paper, border:`1px solid ${T.wire}`,
            borderRadius:6, padding:"20px 22px", marginBottom:14,
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
                hint={mode==="splice_sites"?"required for splice analysis":"optional — improves accuracy"}
                required={mode==="splice_sites"}
                value={form.ref_seq} onChange={set("ref_seq")}
                placeholder="Genomic window around variant…"
                style={{ fontSize:10, letterSpacing:0.3 }}
              />
              <Input
                label="ALT Sequence"
                hint="auto-built from REF if blank"
                value={form.alt_seq} onChange={set("alt_seq")}
                placeholder="Leave blank to auto-compute…"
                style={{ fontSize:10 }}
              />
            </div>

            <div style={{ display:"flex", gap:8, marginTop:18 }}>
              <button onClick={run} disabled={loading} style={{
                flex:1, padding:"13px 0",
                background: loading
                  ? T.lift
                  : `linear-gradient(135deg, ${T.lime}dd, ${T.blue}cc)`,
                border:`1px solid ${loading ? T.wire : T.lime+"55"}`,
                borderRadius:5,
                color: loading ? T.muted : T.bg,
                fontFamily:SAN, fontWeight:700, fontSize:14,
                cursor:loading?"not-allowed":"pointer",
                display:"flex", alignItems:"center", justifyContent:"center", gap:9,
                boxShadow: loading ? "none" : `0 0 30px ${T.lime}20`,
                transition:"all 0.2s",
              }}>
                {loading ? (
                  <>
                    <div style={{
                      width:13, height:13,
                      border:`2px solid ${T.muted}`,
                      borderTopColor:T.lime,
                      borderRadius:"50%",
                      animation:"spin 0.55s linear infinite",
                    }}/>
                    Analysing…
                  </>
                ) : "Run Analysis →"}
              </button>
              <button onClick={()=>{ setForm({...EX,alt_seq:""}); setResult(null); setError(null); }} style={{
                padding:"13px 18px", background:T.lift,
                border:`1px solid ${T.wire}`, borderRadius:5,
                color:T.ghost, fontFamily:MON, fontSize:9, cursor:"pointer",
              }}>
                Example
              </button>
              {result && (
                <button onClick={()=>setResult(null)} style={{
                  padding:"13px 18px", background:T.lift,
                  border:`1px solid ${T.wire}`, borderRadius:5,
                  color:T.ghost, fontFamily:MON, fontSize:9, cursor:"pointer",
                }}>
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* ── ERROR ── */}
          {error && (
            <div style={{
              marginBottom:14, padding:"12px 16px",
              background:`${T.red}10`, border:`1px solid ${T.red}44`,
              borderRadius:5, color:T.red, fontFamily:MON, fontSize:11,
              animation:"fadeIn 0.2s ease",
            }}>
              ⚠ {error}
            </div>
          )}

          {/* ── RESULT ── */}
          {result && (
            <div ref={resultRef}>
              {result._mode==="splice_sites"
                ? <SpliceCard r={result} onClose={()=>setResult(null)}/>
                : <PredCard   r={result} onClose={()=>setResult(null)}/>}
            </div>
          )}

          {/* ── FOOTER ── */}
          <div style={{
            marginTop:60, textAlign:"center",
            fontFamily:MON, fontSize:7, color:T.wire, letterSpacing:2, lineHeight:2.5,
          }}>
            {API} · XGBoost k-mer · DNABERT-2-117M · PWM splice scoring<br/>
            Not for clinical use · Validate with ClinVar / ACMG / MaxEntScan
          </div>

        </div>
      </div>
    </>
  );
}