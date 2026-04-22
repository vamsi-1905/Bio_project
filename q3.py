

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy.stats import fisher_exact, chi2 as chi2_dist, norm

BASE_DIR = r"C:\Users\vamsi\Downloads\ibs_lab _q1&q2\ibs_lab"
os.chdir(BASE_DIR)


df = pd.read_csv("splice_dataset_full.csv")
print(f"Total: {len(df):,}  |  Pathogenic: {(df['label']==1).sum():,}  |  Benign: {(df['label']==0).sum():,}")

# Wilson CI 
def wilson_ci(count, total, alpha=0.05):
    z = norm.ppf(1 - alpha/2)
    p = count / total
    denom = 1 + z**2/total
    center = (p + z**2/(2*total)) / denom
    margin = z * np.sqrt(p*(1-p)/total + z**2/(4*total**2)) / denom
    return (center - margin)*100, (center + margin)*100

# Disruption function 
def disrupts_splice(ref, alt):
    ref = str(ref).upper()
    alt = str(alt).upper()
    if len(ref) == 1 and len(alt) == 1:
        return (ref in {"G", "T", "A"}) and (alt != ref)
    else:
        return ("GT" in ref and "GT" not in alt) or \
               ("AG" in ref and "AG" not in alt)

df["disrupts"] = df.apply(lambda r: disrupts_splice(r["ref"], r["alt"]), axis=1)

total     = len(df)
observed  = df["disrupts"].sum()
null_rate = 0.125
expected  = total * null_rate

chi2_val   = ((observed - expected)**2 / expected) + \
             ((total - observed - (total - expected))**2 / (total - expected))
p_baseline = chi2_dist.sf(chi2_val, df=1)
ci_low, ci_high = wilson_ci(observed, total)

path   = df[df["label"]==1]
benign = df[df["label"]==0]
path_dis = path["disrupts"].sum()
ben_dis  = benign["disrupts"].sum()

odds_ratio, p_class = fisher_exact(
    [[path_dis, len(path)-path_dis],
     [ben_dis,  len(benign)-ben_dis]]
)
path_rate = path_dis / len(path) * 100
ben_rate  = ben_dis  / len(benign) * 100
path_ci   = wilson_ci(path_dis, len(path))
ben_ci    = wilson_ci(ben_dis,  len(benign))

# OR confidence interval
a, b, c, d = path_dis, len(path)-path_dis, ben_dis, len(benign)-ben_dis
se_log_or  = np.sqrt(1/a + 1/b + 1/c + 1/d)
or_ci_low  = np.exp(np.log(odds_ratio) - 1.96*se_log_or)
or_ci_high = np.exp(np.log(odds_ratio) + 1.96*se_log_or)

print(f"Observed : {observed/total*100:.2f}%  95%CI [{ci_low:.2f}%, {ci_high:.2f}%]")
print(f"Chi2     : {chi2_val:.2f},  p = {p_baseline:.2e}")
print(f"Path rate: {path_rate:.2f}%  Benign rate: {ben_rate:.2f}%")
print(f"OR       : {odds_ratio:.3f}  [{or_ci_low:.3f}, {or_ci_high:.3f}]  p = {p_class:.2e}")


BG      = "#0f1117"
PANEL   = "#1a1d27"
CRIMSON = "#e63946"
STEEL   = "#457b9d"
GOLD    = "#f4a261"
WHITE   = "#f1faee"
MUTED   = "#8d99ae"
GREEN   = "#2ec4b6"

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : PANEL,
    "axes.edgecolor"    : "#2e3250",
    "axes.labelcolor"   : WHITE,
    "axes.titlecolor"   : WHITE,
    "xtick.color"       : MUTED,
    "ytick.color"       : MUTED,
    "text.color"        : WHITE,
    "grid.color"        : "#2e3250",
    "grid.linewidth"    : 0.6,
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.spines.left"  : False,
    "axes.spines.bottom": False,
})

fig = plt.figure(figsize=(18, 9), facecolor=BG)
fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.12,
                    wspace=0.45, hspace=0.5)
gs = GridSpec(2, 3, figure=fig)

fig.text(0.5, 0.96, "Q3 — Dataset Specificity Validation",
         ha="center", va="top", fontsize=17, fontweight="bold", color=WHITE)
fig.text(0.5, 0.925,
         "Splice-Site Disruption Variants  ·  ClinVar  ·  n = 51,619",
         ha="center", va="top", fontsize=10, color=MUTED)

# Plot 1: Lollipop — Observed vs Baseline 
ax1 = fig.add_subplot(gs[0, 0])
labels_p = ["Observed\nDisruption", "Random\nBaseline"]
values_p = [observed/total*100, 12.5]
colors_p = [CRIMSON, MUTED]

for i, (v, c) in enumerate(zip(values_p, colors_p)):
    ax1.plot([0, v], [i, i], color=c, lw=2.5, solid_capstyle="round", zorder=2)
    ax1.scatter([v], [i], color=c, s=180, zorder=3,
                edgecolors="white", linewidth=1.2)
    ax1.text(v + 1.5, i, f"{v:.1f}%", va="center",
             fontsize=10, fontweight="bold", color=c)

ax1.barh(0, ci_high - ci_low, left=ci_low, height=0.18,
         color=CRIMSON, alpha=0.18, zorder=1)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(labels_p, fontsize=9)
ax1.set_xlim(0, 88)
ax1.set_xlabel("Disruption Rate (%)", fontsize=9)
ax1.set_title("V1 · Enrichment vs Baseline", fontsize=11, fontweight="bold", pad=10)
ax1.axvline(12.5, color=MUTED, ls="--", lw=1, alpha=0.5)
ax1.grid(axis="x", alpha=0.3)
ax1.text(0.97, 0.08, f"χ²={chi2_val:.0f}\np < 0.001",
         transform=ax1.transAxes, ha="right", va="bottom", fontsize=8, color=GOLD,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                   edgecolor=GOLD, linewidth=1))

#Plot 2: Per-class rates with CI
ax2 = fig.add_subplot(gs[0, 1])
class_labels2 = ["Benign", "Pathogenic"]
rates2  = [ben_rate, path_rate]
cis2    = [ben_ci,   path_ci]
colors2 = [STEEL,    CRIMSON]

for i, (r, ci, c) in enumerate(zip(rates2, cis2, colors2)):
    ax2.barh(i, r, height=0.45, color=c, alpha=0.85, edgecolor="none", zorder=2)
    ax2.errorbar(r, i, xerr=[[r - ci[0]], [ci[1] - r]],
                 fmt="none", color="white", capsize=5,
                 capthick=1.5, elinewidth=1.5, zorder=3)
    ax2.text(r + 1.2, i, f"{r:.1f}%", va="center",
             fontsize=10, fontweight="bold", color=c)

ax2.axvline(12.5, color=MUTED, ls="--", lw=1, alpha=0.5)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(class_labels2, fontsize=10)
ax2.set_xlim(0, 88)
ax2.set_xlabel("Disruption Rate (%)", fontsize=9)
ax2.set_title("V1 · Rate by Clinical Class", fontsize=11, fontweight="bold", pad=10)
ax2.grid(axis="x", alpha=0.3)
ax2.text(0.97, 0.08, f"OR = {odds_ratio:.2f}\np = {p_class:.1e}",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=8, color=GOLD,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                   edgecolor=GOLD, linewidth=1))

# Plot 3: Stacked proportions 
ax3 = fig.add_subplot(gs[0, 2])
cats3  = ["Pathogenic", "Benign"]
dis3   = [path_rate,       ben_rate]
nodis3 = [100-path_rate,   100-ben_rate]
x3     = np.arange(len(cats3))
w3     = 0.45

for xi, (d, n, c) in enumerate(zip(dis3, nodis3, [CRIMSON, STEEL])):
    ax3.bar(xi, d,    width=w3, color=c,    alpha=0.9, edgecolor="none")
    ax3.bar(xi, n,    width=w3, bottom=d,   color=c,   alpha=0.22, edgecolor="none")
    ax3.text(xi, d/2,    f"{d:.1f}%",   ha="center", va="center",
             fontsize=9, fontweight="bold", color="white")
    ax3.text(xi, d + n/2, f"{n:.1f}%", ha="center", va="center",
             fontsize=9, color=MUTED)

ax3.set_xticks(x3)
ax3.set_xticklabels(cats3, fontsize=10)
ax3.set_ylabel("Proportion (%)", fontsize=9)
ax3.set_ylim(0, 115)
ax3.set_title("V2 · Disruption Composition", fontsize=11, fontweight="bold", pad=10)
ax3.grid(axis="y", alpha=0.3)
legend_els = [mpatches.Patch(color=CRIMSON, alpha=0.9, label="Disrupts GT/AG"),
              mpatches.Patch(color=MUTED,   alpha=0.5, label="No Disruption")]
ax3.legend(handles=legend_els, fontsize=8, loc="upper right",
           facecolor=PANEL, edgecolor="#2e3250", labelcolor=WHITE)

#Plot 4: Contingency heatmap
ax4 = fig.add_subplot(gs[1, 0])
cont     = np.array([[path_dis, len(path)-path_dis],
                     [ben_dis,  len(benign)-ben_dis]])
cont_pct = cont / cont.sum(axis=1, keepdims=True) * 100

im = ax4.imshow(cont_pct, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=100)
for i in range(2):
    for j in range(2):
        ax4.text(j, i, f"{cont[i,j]:,}\n({cont_pct[i,j]:.1f}%)",
                 ha="center", va="center", fontsize=9, fontweight="bold",
                 color="white",
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])

ax4.set_xticks([0, 1])
ax4.set_xticklabels(["Disrupts GT/AG", "No Disruption"], fontsize=9)
ax4.set_yticks([0, 1])
ax4.set_yticklabels(["Pathogenic", "Benign"], fontsize=9)
ax4.set_title("V2 · Contingency Heatmap", fontsize=11, fontweight="bold", pad=10)
cb = plt.colorbar(im, ax=ax4, fraction=0.04, pad=0.04)
cb.set_label("Row %", color=MUTED)
cb.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)

#Plot 5: Odds ratio forest plot 
ax5 = fig.add_subplot(gs[1, 1])
ax5.barh(0, odds_ratio, height=0.3, color=GOLD, alpha=0.85, edgecolor="none")
ax5.errorbar(odds_ratio, 0,
             xerr=[[odds_ratio - or_ci_low], [or_ci_high - odds_ratio]],
             fmt="none", color="white", capsize=6,
             capthick=2, elinewidth=2, zorder=3)
ax5.axvline(1.0, color=MUTED, ls="--", lw=1.5, alpha=0.8)
ax5.set_xlim(0, or_ci_high * 1.5)
ax5.set_yticks([])
ax5.set_xlabel("Odds Ratio", fontsize=9)
ax5.set_title("V2 · Effect Size (OR + 95% CI)", fontsize=11, fontweight="bold", pad=10)
ax5.grid(axis="x", alpha=0.3)
ax5.text(odds_ratio + 0.04, 0,
         f"OR = {odds_ratio:.2f}\n[{or_ci_low:.2f} – {or_ci_high:.2f}]",
         va="center", fontsize=9, fontweight="bold", color=GOLD)
ax5.text(0.97, 0.1, f"p = {p_class:.1e}",
         transform=ax5.transAxes, ha="right", va="bottom", fontsize=8, color=GOLD,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                   edgecolor=GOLD, linewidth=1))
ax5.text(1.02, -0.25, "OR = 1\n(null)", ha="center", va="top",
         fontsize=7, color=MUTED, transform=ax5.get_yaxis_transform())

# Plot 6: Summary panel 
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("#12151f")
ax6.axis("off")

lines = [
    ("VERIFICATION 1",                             GOLD,    13, "bold"),
    ("Baseline Enrichment",                        WHITE,   10, "bold"),
    (f"  Observed:  {observed/total*100:.1f}%",    MUTED,    9, "normal"),
    (f"  Expected:  12.5%  (random DNA)",          MUTED,    9, "normal"),
    (f"  χ² = {chi2_val:.0f},   p < 0.001",        GREEN,    9, "normal"),
    (f"  95% CI:  [{ci_low:.1f}%, {ci_high:.1f}%]",MUTED,   9, "normal"),
    ("",                                           WHITE,    5, "normal"),
    ("VERIFICATION 2",                             GOLD,    13, "bold"),
    ("Clinical Association",                       WHITE,   10, "bold"),
    (f"  Pathogenic:  {path_rate:.1f}%",           MUTED,    9, "normal"),
    (f"  Benign:      {ben_rate:.1f}%",            MUTED,    9, "normal"),
    (f"  OR = {odds_ratio:.2f}  [{or_ci_low:.2f}–{or_ci_high:.2f}]", GREEN, 9, "normal"),
    (f"  p = {p_class:.1e}  (Fisher Exact)",       MUTED,    9, "normal"),
    ("",                                           WHITE,    5, "normal"),
    ("CONCLUSION",                                 CRIMSON, 11, "bold"),
    ("Canonical splice-site disruption is",        WHITE,  8.5, "normal"),
    ("significantly enriched and strongly",        WHITE,  8.5, "normal"),
    ("associated with pathogenicity —",            WHITE,  8.5, "normal"),
    ("confirming biological specificity.",         WHITE,  8.5, "normal"),
]

y = 0.97
for text, color, size, weight in lines:
    ax6.text(0.05, y, text, transform=ax6.transAxes,
             va="top", fontsize=size, fontweight=weight, color=color)
    y -= size * 0.018 + 0.01

ax6.set_title("Summary", fontsize=11, fontweight="bold", pad=10, color=WHITE)

plt.savefig("q3_specificity_analysis.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.show()
print("\nSaved: q3_specificity_analysis.png")