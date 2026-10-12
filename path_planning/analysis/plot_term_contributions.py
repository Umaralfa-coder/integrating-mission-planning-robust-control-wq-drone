"""
Term-contribution chart: for each weight vector tested in the Pareto sweep,
show the magnitude each fitness term actually contributes to F(T),
|w_i * term_i|, on a log y-axis. This is the chart that demonstrates the
scale-mismatch finding -- not a scatter of (mostly identical) solutions.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

with open('pareto_sweep_results.json') as f:
    results = json.load(f)

# Validated categorical palette (dataviz skill, slots 1-4)
COL_COVERAGE = '#2a78d6'   # blue
COL_INFOGAIN = '#eb6834'   # orange
COL_LENGTH   = '#1baf7a'   # aqua
COL_SMOOTH   = '#eda100'   # yellow
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECOND  = '#52514e'
GRID = '#d8d7d2'

names = [r['name'] for r in results]
n = len(names)

terms = {
    'Coverage  (w1·C)':    (COL_COVERAGE, [r['w1'] * r['coverage'] for r in results]),
    'Info gain  (w2·I)':   (COL_INFOGAIN, [r['w2'] * r['info_gain'] for r in results]),
    'Path length  (w3·L)': (COL_LENGTH,   [r['w3'] * r['total_distance'] for r in results]),
    'Smoothness  (w4·S)':  (COL_SMOOTH,   [r['w4'] * r['smoothness'] for r in results]),
}

fig, ax = plt.subplots(figsize=(11, 6.2))
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')

n_terms = len(terms)
group_w = 0.8
bar_w = group_w / n_terms
x = np.arange(n)

for i, (label, (color, values)) in enumerate(terms.items()):
    offsets = x - group_w / 2 + bar_w * i + bar_w / 2
    bars = ax.bar(offsets, values, width=bar_w * 0.86, color=color,
                  edgecolor='#fcfcfb', linewidth=1.5, label=label, zorder=3)
    for rect, v in zip(bars, values):
        va = 'bottom'
        label_txt = f'{v:.3g}'
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.15, label_txt,
                ha='center', va=va, fontsize=8, color=TEXT_SECOND)

ax.set_yscale('log')
ax.set_ylim(1e-3, 3000)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10, color=TEXT_PRIMARY)
ax.set_ylabel('Contribution to F(T), |weight × term|  (log scale)', fontsize=10, color=TEXT_PRIMARY)
ax.set_title('Every weight vector is dominated by the same term: path length\n'
             'Raw units are never normalized, so −w₃·L (meters) outweighs the other\n'
             'three terms (all in [0, 1]) by 2–3 orders of magnitude regardless of the weights chosen',
             fontsize=12, fontweight='bold', color=TEXT_PRIMARY, loc='left')
ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10))
ax.grid(True, which='major', axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.grid(True, which='minor', axis='y', color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
for spine in ('left', 'bottom'):
    ax.spines[spine].set_color(GRID)

legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4,
                    frameon=False, fontsize=9.5)
for text in legend.get_texts():
    text.set_color(TEXT_PRIMARY)

plt.tight_layout()
plt.savefig('../../figures/term_contributions.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: term_contributions.png')
