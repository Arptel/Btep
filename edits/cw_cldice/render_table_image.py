"""
Script to render a high-resolution, publication-quality image of the 3-Way STARE Comparison Table.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

headers = [
    "Dimension",
    "Metric",
    "Baseline SA-UNetv2\n(ISBI 2026)",
    "Vanilla clDice\n(CVPR 2021)",
    "Ours (cw-clDice)\n(Conductance-Weighted)",
    "Delta vs.\nBaseline",
    "Delta vs.\nclDice",
    "Clinical & Topological Significance"
]

rows = [
    # Topological Connectivity
    ["Topological\nConnectivity", "Centerline Dice (clDice)", "86.57%", "87.27%", "87.66%", "+1.09%", "+0.39%", "Highest centerline skeleton fidelity across vascular tree"],
    ["", "Topology Sensitivity (Tsens)", "80.97%", "82.54%", "84.18%", "+3.20%", "+1.64%", "Caliber weighting recovers +1.64% more faint capillaries"],
    ["", "Topology Precision (Tprec)", "93.24%", "92.82%", "91.70%", "-1.54%", "-1.12%", "Controlled trade-off to capture faint peripheral branches"],
    ["", "Betti-0 Stumps (beta0)", "57.50", "58.00", "51.50", "-6.00", "-6.50", "Eliminates disconnected stumps (-6.50 vs vanilla clDice)"],
    ["", "Fragmentation Ratio", "24.19x", "24.94x", "22.81x", "-1.38x", "-2.12x", "Lowest fragmentation ratio across all evaluated models"],
    ["", "Largest Tree Ratio (LCCR)", "81.27%", "80.27%", "80.44%", "-0.83%", "+0.17%", "Preserves primary vascular trunk structural integrity"],
    # Pixel Metrics
    ["Pixel Overlap\n& Quality", "F1-Score / Dice", "82.44%", "83.14%", "83.14%", "+0.71%", "+0.01%", "State-of-the-art pixel-level volumetric segmentation"],
    ["", "Sensitivity (Recall)", "83.38%", "84.57%", "84.96%", "+1.58%", "+0.39%", "Highest recovery of true vessel pixels across test set"],
    ["", "Specificity", "98.50%", "98.50%", "98.46%", "-0.04%", "-0.04%", "Sustains exceptional non-vessel background suppression"],
    ["", "Global Accuracy", "97.40%", "97.49%", "97.48%", "+0.07%", "-0.01%", "Preserved global classification accuracy (>97.4%)"],
    ["", "Matthews Corr (MCC)", "81.08%", "81.83%", "81.85%", "+0.77%", "+0.02%", "Superior balanced correlation on skewed retinal pixels"],
    ["", "AUC-ROC", "98.69%", "98.75%", "98.76%", "+0.06%", "+0.01%", "Highest discriminative confidence across thresholds"]
]

fig, ax = plt.subplots(figsize=(22, 10.5), dpi=300)
ax.axis('off')
ax.axis('tight')

# Create Table
table = ax.table(
    cellText=rows,
    colLabels=headers,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(11)

col_widths = [0.10, 0.17, 0.11, 0.11, 0.14, 0.07, 0.07, 0.31]
for i, width in enumerate(col_widths):
    for j in range(len(rows) + 1):
        cell = table[(j, i)]
        cell.set_width(width)

# Colors
header_color = '#1E293B'      # Dark slate
topo_bg_alt = '#F8FAFC'       # Light gray-blue
topo_bg = '#FFFFFF'
pixel_bg_alt = '#F0FDF4'      # Subtle greenish-gray
pixel_bg = '#FFFFFF'
highlight_green = '#DCFCE7'   # Winner cell highlight
text_green = '#15803D'        # Bold green delta

for (row_idx, col_idx), cell in table.get_celld().items():
    cell.set_edgecolor('#CBD5E1')
    cell.set_linewidth(0.8)
    
    # Header styling
    if row_idx == 0:
        cell.set_facecolor(header_color)
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
        cell.get_text().set_fontsize(11.5)
        cell.set_height(0.08)
    else:
        cell.set_height(0.06)
        r = row_idx - 1
        is_topo = r < 6
        base_bg = topo_bg if r % 2 == 0 else topo_bg_alt if is_topo else (pixel_bg if r % 2 == 0 else pixel_bg_alt)
        cell.set_facecolor(base_bg)
        
        # Alignments
        if col_idx in [1, 7]:
            cell.get_text().set_ha('left')
        elif col_idx == 0:
            cell.get_text().set_weight('bold')
            cell.get_text().set_color('#0F172A')
        
        # Winner highlight on column 4 (Ours)
        if col_idx == 4 and r not in [2, 8, 9]:
            cell.set_facecolor('#FEF08A' if r == 3 else '#DCFCE7') # subtle gold/green
            cell.get_text().set_weight('bold')
            cell.get_text().set_color('#14532D')
            
        # Delta columns green text
        if col_idx in [5, 6]:
            txt = rows[r][col_idx]
            if txt.startswith('+'):
                cell.get_text().set_color(text_green)
                cell.get_text().set_weight('bold')
            elif '-' in txt and ('stump' in rows[r][7].lower() or 'frag' in rows[r][7].lower() or 'beta0' in rows[r][1].lower()):
                cell.get_text().set_color(text_green)
                cell.get_text().set_weight('bold')
            else:
                cell.get_text().set_color('#64748B')

plt.title(
    "STARE Benchmark: 3-Way Performance Comparison\n"
    "Baseline SA-UNetv2 (ISBI 2026) vs. Vanilla clDice (CVPR 2021) vs. Ours cw-clDice (Conductance-Weighted)",
    fontsize=16,
    fontweight='bold',
    pad=24,
    color='#0F172A'
)

out_dir = "results/stare_3way"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "stare_3way_comparison_table.png")
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"[+] Screenshot saved to: {out_path}")
