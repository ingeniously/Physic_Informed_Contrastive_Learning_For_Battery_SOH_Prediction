import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# --- Directory and path management ---
ROOT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = ROOT_DIR.parent

# Make sure the plots directory exists
plots_dir = PROJECT_ROOT / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Path to processed data
csv_file = PROJECT_ROOT / "data_processed" / "preprocessed_battery_health_dataset_all_points.csv"
df = pd.read_csv(csv_file, sep=",")
df.columns = df.columns.str.strip()

if 'Time_rel' not in df.columns:
    raise ValueError(f"The column 'Time_rel' was not found. Actual columns: {df.columns.tolist()}")

# Assign cycle numbers: each time Time_rel == 0, increment the cycle
df['cycle'] = (df['Time_rel'] == 0).cumsum()

# --- Features for paper figure ---
paper_features = [
    "Voltage_measured",
    "Temperature_measured",
    "Voltage_load",
    "Current_load"
]

# Check that all features exist
for f in paper_features:
    if f not in df.columns:
        raise ValueError(f"Requested feature '{f}' not found in data columns: {df.columns.tolist()}")

# Use a more sober continuous colormap, e.g., 'cividis' or 'viridis'
cmap = cm.GnBu # You can also try cm.viridis or cm.YlGnBu for other sober options

cycles = df['cycle'].unique()
norm = Normalize(vmin=cycles.min(), vmax=cycles.max())

n_features = len(paper_features)
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

# Wider padding for colorbar
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8.5), sharex=False)
axes = axes.flatten()

for i, feature in enumerate(paper_features):
    ax = axes[i]
    for cycle_num, cycle_df in df.groupby('cycle'):
        color = cmap(norm(cycle_num))
        ax.plot(
            cycle_df['Time_rel'],
            cycle_df[feature],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            zorder=3
        )
    ax.set_title(feature.replace("_", " ").title(), fontsize=18, fontweight='bold', pad=8)
    ax.set_ylabel("")
    ax.grid(True, alpha=0.23, linestyle="--", zorder=0)
    ax.set_xlabel("Time(s)", fontsize=15, fontweight="bold", labelpad=7)
    ax.tick_params(axis='x', which='major', labelsize=13)
    ax.tick_params(axis='y', which='major', labelsize=13)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#444")
    ax.set_facecolor("#fff")

# Hide any unused subplots (if n_features not multiple of n_cols)
for j in range(len(paper_features), len(axes)):
    fig.delaxes(axes[j])

# Add a single colorbar for all plots, placed outside the grid for clarity
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Place colorbar to the right, outside subplots, not overlapping
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.64])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Normalized Cycle Number", fontsize=15, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

plt.tight_layout(rect=(0.01, 0.01, 0.88, 0.99), pad=2.0, h_pad=2.7, w_pad=2.7)
paper_fig_path = plots_dir / "battery_features_paper_Q1_gradient_sober.png"
plt.savefig(paper_fig_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved Q1-paper-ready figure with sober gradient colorbar to {paper_fig_path}")