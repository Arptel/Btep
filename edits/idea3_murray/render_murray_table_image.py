"""
Script to render high-resolution publication-quality images for:
1. Murray's Law Anatomical Compliance Audit Table
2. Complete 8-Way Comprehensive Multi-Paradigm Comparison Table
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. RENDER MURRAY ANATOMICAL AUDIT TABLE
# ==========================================
murray_headers = [
    "Model Configuration",
    "Underlying Paradigm",
    "Bifurcation\nRecall (%)",
    "Bifurcation\nPrecision (%)",
    "Murray Ratio\n(R_bar)",
    "Murray\nDeviation",
    "Predicted\nBifurcations",
    "Anatomical Assessment"
]

murray_rows = [
    ["Baseline (ISBI 2026)", "Standard BCE + MCC", "51.42%", "78.68%", "1.507", "0.546", "73.8", "Severely under-recovers branch points (~49% severed)"],
    ["Vanilla clDice", "+ Uniform Centerline (CVPR 2021)", "53.42%", "78.71%", "1.495", "0.532", "77.5", "Marginal +2% bifurcation gain, uniform loss blind to caliber"],
    ["Standalone cw-BCE (1A)", "+ Caliber Pixel BCE", "57.81%", "70.52%", "1.541", "0.576", "92.5", "Recovers faint branches (+6.4%) but introduces noise spurs"],
    ["Ours cw-clDice (Proposed)", "+ Caliber Medial Axis", "55.42%", "77.53%", "1.513", "0.551", "81.5", "Balanced branch recovery without sacrificing precision"],
    ["Ours Unified (1B)", "Dual Caliber Supervision", "60.29%", "72.51%", "1.518", "0.543", "93.5", "Breaks 60% bifurcation recall (+8.87% over baseline)"],
    ["CAD-Topo-CSA (Track 2)", "Directional Skip Conv", "56.79%", "73.27%", "1.517", "0.548", "89.0", "Directional strip pooling physically preserves bifurcations"],
    ["CAD-Topo-CSA + Unified", "Idea 1 + Idea 2 Synthesis", "62.22%", "68.68%", "1.533", "0.560", "103.2", "Highest bifurcation recall (62.22%, +10.80% vs baseline)"],
    ["CAD-Topo-CSA + Murray", "Idea 1 + 2 + 3 (Triple Synthesis)", "60.19%", "70.07%", "1.518", "0.542", "98.2", "Anatomically stabilized: lower deviation and fewer spurs"]
]

fig, ax = plt.subplots(figsize=(25, 9), dpi=300)
ax.axis('off')
ax.axis('tight')

table = ax.table(
    cellText=murray_rows,
    colLabels=murray_headers,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10.5)

col_widths = [0.17, 0.18, 0.09, 0.09, 0.08, 0.08, 0.08, 0.23]
for i, width in enumerate(col_widths):
    for j in range(len(murray_rows) + 1):
        cell = table[(j, i)]
        cell.set_width(width)

header_color = '#0F172A'
alt_color = '#F8FAFC'

for (row_idx, col_idx), cell in table.get_celld().items():
    cell.set_edgecolor('#CBD5E1')
    cell.set_linewidth(0.8)

    if row_idx == 0:
        cell.set_facecolor(header_color)
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
        cell.get_text().set_fontsize(11)
        cell.set_height(0.08)
    else:
        cell.set_height(0.065)
        r = row_idx - 1
        cell.set_facecolor('#FFFFFF' if r % 2 == 0 else alt_color)

        if col_idx in [0, 7]:
            cell.get_text().set_ha('left')
        if col_idx == 0:
            cell.get_text().set_weight('bold')

        # Highlight winner rows
        if r == 6 and col_idx == 2:  # Highest recall (62.22%)
            cell.set_facecolor('#FEF08A')
            cell.get_text().set_weight('bold')
            cell.get_text().set_color('#14532D')
        elif r == 7 and col_idx in [2, 5]:  # Murray stabilized
            cell.set_facecolor('#DCFCE7')
            cell.get_text().set_weight('bold')
            cell.get_text().set_color('#14532D')

plt.title(
    "STARE Benchmark: Murray's Law Anatomical Bifurcation Audit\n"
    "Auditing Physiological Hemodynamic Minimum-Work Compliance (r0^3 = r1^3 + r2^3) Across All Models",
    fontsize=15,
    fontweight='bold',
    pad=20,
    color='#0F172A'
)

out_dir = "results/stare_multiway"
os.makedirs(out_dir, exist_ok=True)
out_path_murray = os.path.join(out_dir, "stare_murray_audit_table.png")
plt.savefig(out_path_murray, bbox_inches='tight', dpi=300)
plt.close()
print(f"[+] Murray Audit Table saved to: {out_path_murray}")
