import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

df = pd.read_csv('runtime_results_mine.csv')

models = df['Model'].unique()
dims = df['In'].unique()

# ICML-style typography and sizing
plt.style.use("classic")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.8,
    }
)

# Colorblind-friendly palette
colors = {
    'FGG-LNN': '#009E73',
    'Chen': '#0072B2',
    'Poincaré': '#D55E00',
    'Euclidean': '#CC79A7',
}

linestyles = {
    'FGG-LNN': (0, ()),
    'Chen': (0, (3, 2)),
    'Poincaré': (0, (1, 2)),
    'Euclidean': (0, (5, 2, 1, 2)),
}

markers = {
    'FGG-LNN': 's',
    'Chen': '^',
    'Poincaré': 'D',
    'Euclidean': 'o',
}

fig, ax = plt.subplots(figsize=(6.75, 3.1))

for model in models:
    model_data = df[df['Model'] == model].sort_values('In')
    line, = ax.plot(
        model_data['In'], 
        model_data['Fwd Mean (ms)'],
        label=model,
        color=colors[model],
        linestyle=linestyles[model],
        marker=markers[model],
        markersize=6,
        linewidth=2,
        zorder=3,
    )
    line.set_path_effects(
        [pe.Stroke(linewidth=4.4, foreground="white"), pe.Normal()]
    )

ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Input/Output Dimension', fontsize=11)
ax.set_ylabel('Forward Pass Time (ms)', fontsize=11)
ax.set_title('Linear layer runtimes', fontsize=13, pad=10)
ax.set_xticks(dims)
ax.set_xticklabels(dims)
ax.legend(frameon=False, loc='upper left')
ax.grid(True, which='major', linestyle='--', alpha=0.25)
ax.set_axisbelow(True)
ax.set_xlim(12, 6000)

plt.tight_layout()
plt.savefig('benchmark_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('benchmark_plot.pdf', bbox_inches='tight')