"""
Clean, publication-quality figures for the paper.
Run: python3 generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = "/Users/spartan/Desktop/Music Recommendation"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8,
    'savefig.dpi': 220,
})

# ── Shared palette ────────────────────────────────────────────────────────────
C = {
    'blue_dark':   '#1A6FA8',
    'blue_light':  '#D6EAF8',
    'red_dark':    '#B03A2E',
    'red_light':   '#FADBD8',
    'purple_dark': '#6C3483',
    'purple_light':'#E8DAEF',
    'green_dark':  '#1E8449',
    'green_light': '#D5F5E3',
    'orange_dark': '#D35400',
    'orange_light':'#FDEBD0',
    'gray_dark':   '#555F6D',
    'gray_light':  '#EAECEE',
    'neutral_dark':'#2C3E50',
    'neutral_light':'#F2F3F4',
}

def rbox(ax, cx, cy, w, h, label, fc, ec, fs=9, fw='normal', tc='#1a1a2e', lw=1.4):
    """Draw a clean rounded rectangle with centred label."""
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle='round,pad=0.06',
                       facecolor=fc, edgecolor=ec,
                       linewidth=lw, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=tc, zorder=4,
            multialignment='center', linespacing=1.35)
    return p

def arr(ax, x0, y0, x1, y1, color='#444', lw=1.4, style='->', rad=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Multimodal GRU fusion architecture
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(8.5, 6.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 8.5); ax.set_ylim(0, 6.2)
ax.axis('off')

# Column x-centres
XL, XR, XM = 2.0, 6.5, 4.25   # left, right, middle

# Row y-centres (top → bottom)
Y = [5.65, 4.75, 3.80, 2.70, 1.72, 0.78]
#    input  embed  proc  gate  gru   out

BW, BH = 2.0, 0.52   # standard box width / height

# ── Column headers ────────────────────────────────────────────────────────────
ax.text(XL, 6.05, 'Collaborative Branch', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C['blue_dark'])
ax.text(XR, 6.05, 'Acoustic Branch', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C['red_dark'])

# ── Input boxes ───────────────────────────────────────────────────────────────
rbox(ax, XL, Y[0], BW, BH, 'Track ID', C['neutral_light'], C['gray_dark'], fs=9)
rbox(ax, XR, Y[0], BW, BH, 'Track ID', C['neutral_light'], C['gray_dark'], fs=9)

# ── Embedding boxes ───────────────────────────────────────────────────────────
rbox(ax, XL, Y[1], BW+0.1, BH+0.08,
     'SVD Embedding\n128-dim', C['blue_light'], C['blue_dark'], fs=9)
rbox(ax, XR, Y[1], BW+0.3, BH+0.08,
     'Audio Embedding\n1024-dim  (frozen)', C['red_light'], C['red_dark'], fs=8.5)

# ── Processing boxes ──────────────────────────────────────────────────────────
rbox(ax, XL, Y[2], BW, BH,
     'L2 Normalise', C['green_light'], C['green_dark'], fs=9)
rbox(ax, XR, Y[2], BW+0.5, BH+0.18,
     'MLP Projection\n1024 → 256 → 128\n+ LayerNorm', C['orange_light'], C['orange_dark'], fs=8)

# ── Fusion gate (wide, centred) ───────────────────────────────────────────────
GW, GH = 4.8, 0.75
rbox(ax, XM, Y[3], GW, GH,
     'Per-Dimension Fusion Gate\n'
     r'$\mathbf{h} = \boldsymbol{\alpha}\odot\mathbf{e}_{SVD}'
     r'+ (1-\boldsymbol{\alpha})\odot\mathbf{e}_{audio}$',
     C['purple_light'], C['purple_dark'], fs=8.5, fw='normal')

# small gate-weight annotations — placed OUTSIDE the gate box
ax.text(XM - 1.5, Y[3] - 0.52,
        r'$\bar{\alpha}\approx0.74$ → SVD dominates',
        ha='center', va='center', fontsize=7.5, color=C['blue_dark'],
        style='italic')
ax.text(XM + 1.5, Y[3] - 0.52,
        r'$1-\bar{\alpha}\approx0.26$ for audio',
        ha='center', va='center', fontsize=7.5, color=C['red_dark'],
        style='italic')

# ── GRU stack ─────────────────────────────────────────────────────────────────
rbox(ax, XM, Y[4], 3.4, BH + 0.08,
     'GRU Stack  (2 layers, hidden = 256)\n+ Linear Attention',
     C['blue_light'], C['blue_dark'], fs=8.5, fw='bold')

# ── Output ────────────────────────────────────────────────────────────────────
rbox(ax, XM, Y[5], 3.4, BH,
     'Linear Projection → Logits\n(50 000 tracks)',
     C['purple_light'], C['purple_dark'], fs=8.5)

# ── Vertical arrows within each branch ───────────────────────────────────────
for xi in [XL, XR]:
    for i in range(2):   # input→embed, embed→proc
        arr(ax, xi, Y[i] - BH/2 - 0.04,
                xi, Y[i+1] + BH/2 + 0.04, color='#444')

# ── Converging arrows: straight diagonals from branch bottom → gate top ───────
# Left branch bottom centre → gate top-left third (clean diagonal)
ax.annotate('', xy=(XM - GW/3, Y[3] + GH/2),
            xytext=(XL, Y[2] - BH/2 - 0.02),
            arrowprops=dict(arrowstyle='->', color=C['blue_dark'], lw=1.5),
            zorder=5)

# Right branch bottom centre → gate top-right third (clean diagonal)
ax.annotate('', xy=(XM + GW/3, Y[3] + GH/2),
            xytext=(XR, Y[2] - (BH + 0.18)/2 - 0.02),
            arrowprops=dict(arrowstyle='->', color=C['red_dark'], lw=1.5),
            zorder=5)

# ── Vertical arrows: gate → GRU → output ─────────────────────────────────────
arr(ax, XM, Y[3] - GH/2 - 0.03, XM, Y[4] + (BH+0.08)/2 + 0.03, color='#444')
arr(ax, XM, Y[4] - (BH+0.08)/2 - 0.03, XM, Y[5] + BH/2 + 0.03, color='#444')

ax.set_title('Multimodal GRU Architecture with Per-Dimension Learned Fusion Gate',
             fontsize=11, fontweight='bold', y=0.985, color='#1a1a2e')

plt.savefig(f'{OUT}/fig_multimodal_arch.png', bbox_inches='tight',
            facecolor='white', dpi=220)
plt.close()
print("✓ fig_multimodal_arch.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — LSH pipeline  (clean horizontal layout)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.6)
ax.axis('off')

# Box centres along a horizontal spine at y=2.3
Y_SPINE = 2.20
BH2 = 0.72

boxes = [
    # (cx,  label,                              fc,               ec,             fs)
    (0.85,  'User\nContext\n(30 sessions)',      C['green_light'], C['green_dark'],  8.0),
    (2.35,  'Mean SVD\nEmbedding\n(128-dim)',    C['blue_light'],  C['blue_dark'],   8.0),
    (3.95,  'Random\nProjection\nb = 8 bits',   C['orange_light'],C['orange_dark'], 8.0),
    (5.55,  'L = 10\nHash Tables\n(b-bit codes)',C['neutral_light'],C['gray_dark'], 8.0),
    (7.15,  'Union of\nBuckets\n~4 700 tracks', C['purple_light'],C['purple_dark'], 8.0),
    (8.75,  'Cosine\nRe-rank\n(dot product)',   C['blue_light'],  C['blue_dark'],   8.0),
]

BW2 = 1.25
for cx, label, fc, ec, fs in boxes:
    rbox(ax, cx, Y_SPINE, BW2, BH2, label, fc, ec, fs=fs)

# Arrows between boxes
for i in range(len(boxes) - 1):
    x0 = boxes[i][0]  + BW2/2 + 0.04
    x1 = boxes[i+1][0] - BW2/2 - 0.04
    arr(ax, x0, Y_SPINE, x1, Y_SPINE, color='#444')

# Final arrow → Top-10 label
arr(ax, 8.75 + BW2/2 + 0.04, Y_SPINE, 9.65, Y_SPINE, color='#444')
ax.text(9.72, Y_SPINE, 'Top\n10', ha='left', va='center',
        fontsize=9, fontweight='bold', color='#1a1a2e')

# ── Speedup annotation row ────────────────────────────────────────────────────
Y_ANN = 0.95
# brute-force span
ax.annotate('', xy=(8.75 + BW2/2, Y_ANN), xytext=(2.35 - BW2/2, Y_ANN),
            arrowprops=dict(arrowstyle='<->', color='#AAA', lw=1.0))
ax.text(5.55, Y_ANN + 0.22,
        'Brute-force: score all 27 536 tracks  (45 ms per query)',
        ha='center', va='bottom', fontsize=8, color='#888', style='italic')

# LSH span
ax.annotate('', xy=(8.75 + BW2/2, Y_ANN - 0.42), xytext=(5.55 - BW2/2, Y_ANN - 0.42),
            arrowprops=dict(arrowstyle='<->', color=C['green_dark'], lw=1.0))
ax.text(7.15, Y_ANN - 0.42 + 0.18,
        'LSH: ~4 700 candidates  (7.8 ms)   →   5.8× faster',
        ha='center', va='bottom', fontsize=8.5,
        color=C['green_dark'], fontweight='bold')

ax.set_title('LSH Candidate Generation Pipeline  '
             '(Random Projection, b = 8 bits per table, L = 10 tables)',
             fontsize=10, fontweight='bold', pad=8, color='#1a1a2e')

plt.tight_layout(pad=0.3)
plt.savefig(f'{OUT}/fig_lsh_pipeline.png', bbox_inches='tight',
            facecolor='white', dpi=220)
plt.close()
print("✓ fig_lsh_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — MinHash banding S-curve
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 4.0))

j = np.linspace(0, 1, 600)

def p_band(j, b, r):
    return 1 - (1 - j**r)**b

p_old = p_band(j, 20, 5)
p_new = p_band(j, 50, 2)

ax.fill_betweenx([0, 1.04], 0, 0.05,
                 color='#FFA040', alpha=0.18, label='Typical music Jaccard  (0 – 0.05)')
ax.plot(j, p_old, color='#C0392B', lw=2.2, label='Original: 20 bands × 5 rows  (threshold ≈ 0.55)')
ax.plot(j, p_new, color='#1A6FA8', lw=2.2, label='Fixed: 50 bands × 2 rows  (threshold ≈ 0.14)')

# threshold dashed verticals
ax.axvline(0.55, color='#C0392B', lw=1.0, ls='--', alpha=0.55)
ax.axvline(0.14, color='#1A6FA8', lw=1.0, ls='--', alpha=0.55)

# threshold labels — placed at safe y position to avoid legend
ax.text(0.57, 0.38, '0.55\n(threshold)', color='#C0392B', fontsize=8,
        ha='left', va='center')
ax.text(0.16, 0.38, '0.14\n(threshold)', color='#1A6FA8', fontsize=8,
        ha='left', va='center')

# Music users label — bottom-left, well away from legend
ax.text(0.025, 0.10,
        'Music\nusers\nhere',
        ha='center', va='bottom', fontsize=8, color='#D35400',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='#FDEBD0', ec='#D35400',
                  lw=0.8, alpha=0.9))

# annotation arrows — place in blank space
ax.annotate('98.6 % of queries\nfound zero candidates\n(original)',
            xy=(0.04, p_band(0.04, 20, 5)),
            xytext=(0.22, 0.15),
            fontsize=7.5, color='#C0392B', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#C0392B',
                      lw=0.7, alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.9))

ax.annotate('94 % find candidates\n(fixed)',
            xy=(0.04, p_band(0.04, 50, 2)),
            xytext=(0.22, 0.60),
            fontsize=7.5, color='#1A6FA8', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1A6FA8',
                      lw=0.7, alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#1A6FA8', lw=0.9))

ax.set_xlabel('Jaccard Similarity', fontsize=10)
ax.set_ylabel('P(candidate pair retrieved)', fontsize=10)
ax.set_title('MinHash Banding S-Curve:\nOriginal vs Fixed Configuration',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=6.5, loc='lower right',
          framealpha=0.95, edgecolor='#ccc',
          handlelength=1.4, handletextpad=0.4,
          borderpad=0.4, labelspacing=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
ax.grid(True, alpha=0.25, lw=0.6)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(f'{OUT}/fig_minhash_scurve.png', bbox_inches='tight',
            facecolor='white', dpi=220)
plt.close()
print("✓ fig_minhash_scurve.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Efficiency–accuracy Pareto
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 4.2))

# (label, latency_ms, metric, marker, color, markersize, label_xytext, ha)
methods = [
    ('Association\nRules',  0.80,  2.14, 's', '#D35400', 80,  (-0.25, 0.8),  'center'),
    ('MinHash CF',          2.10,  1.84, 'D', '#6C3483', 80,  (2.5,  -0.55), 'left'),
    ('LSH  L=3',            3.10,  2.92, 'o', '#1A6FA8', 80,  (4.0,   0.55), 'left'),
    ('LSH  L=10',           7.80,  2.98, 'o', '#2980B9', 95,  (10.0, -0.55), 'left'),
    ('Brute-Force\nCosine', 45.0,  3.02, '^', '#555F6D', 80,  (55.0,  0.65), 'left'),
    ('Neural GRU',         120.0,  4.48, 'P', '#1E8449', 100, (140.0, 0.70), 'left'),
    ('AU2ACTR',            220.0, 20.22, '*', '#B03A2E', 160, (90.0,  1.20), 'right'),
]

for label, lat, metric, marker, color, ms, (dx, dy), ha in methods:
    ax.scatter(lat, metric, marker=marker, color=color, s=ms,
               zorder=5, edgecolors='white', linewidths=0.7)

    tx = lat + dx
    ty = metric + dy
    ax.annotate(label,
                xy=(lat, metric),
                xytext=(tx, ty),
                fontsize=7.5, color=color, fontweight='bold', ha=ha,
                bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=color,
                          lw=0.7, alpha=0.93),
                arrowprops=dict(arrowstyle='-', color=color, lw=0.7,
                                shrinkA=4, shrinkB=4),
                zorder=6)

# Pareto frontier dashed line
px = [0.80, 3.10, 7.80, 45.0, 120.0, 220.0]
py = [2.14, 2.92, 2.98,  3.02,   4.48,  20.22]
ax.plot(px, py, '--', color='#AABBCC', lw=1.1, zorder=2, label='Pareto frontier')

ax.set_xscale('log')
ax.set_xlabel('Query Latency  (ms, log scale, CPU, single-threaded)', fontsize=9.5)
ax.set_ylabel('Accuracy (%)', fontsize=9.5)
ax.set_title('Efficiency–Accuracy Trade-off Across All Methods', fontsize=10,
             fontweight='bold')

# legend for marker shapes
import matplotlib.lines as mlines
leg_handles = [
    mlines.Line2D([], [], marker='s', color='#D35400', ls='', ms=7,
                  label='Association Rules'),
    mlines.Line2D([], [], marker='D', color='#6C3483', ls='', ms=7,
                  label='MinHash CF'),
    mlines.Line2D([], [], marker='o', color='#2980B9', ls='', ms=7,
                  label='LSH + Cosine'),
    mlines.Line2D([], [], marker='^', color='#555F6D', ls='', ms=7,
                  label='Brute-Force Cosine'),
    mlines.Line2D([], [], marker='P', color='#1E8449', ls='', ms=8,
                  label='Neural GRU  (HR@10)'),
    mlines.Line2D([], [], marker='*', color='#B03A2E', ls='', ms=10,
                  label='AU2ACTR  (HR@10)'),
]
ax.legend(handles=leg_handles, fontsize=7.5, loc='upper left',
          framealpha=0.95, edgecolor='#ccc', ncol=1)

# footnote
ax.text(0.99, 0.01,
        '* Neural HR@10 and approximate R@10\n'
        '  use different ground-truth cardinalities',
        transform=ax.transAxes, fontsize=6.8, color='#888',
        ha='right', va='bottom', style='italic')

ax.grid(True, alpha=0.2, lw=0.6)
ax.tick_params(labelsize=9)
ax.set_ylim(0.8, 23)

plt.tight_layout()
plt.savefig(f'{OUT}/fig_pareto.png', bbox_inches='tight',
            facecolor='white', dpi=220)
plt.close()
print("✓ fig_pareto.png")

print("\nAll figures done.")
