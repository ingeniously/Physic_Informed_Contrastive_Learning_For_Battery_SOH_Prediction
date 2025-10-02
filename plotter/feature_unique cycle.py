import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Only use the first discharge cycle (between first and second Time_rel == 0)
time_zeros = df.index[df['Time_rel'] == 0.0].tolist()
if len(time_zeros) > 1:
    first_zero = time_zeros[0]
    second_zero = time_zeros[1]
    df_plot = df.iloc[first_zero:second_zero]
else:
    df_plot = df

# --- Features for paper figure ---
paper_features = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "Resistance"
]

# Check that all features exist
for f in paper_features:
    if f not in df_plot.columns:
        raise ValueError(f"Requested feature '{f}' not found in data columns: {df_plot.columns.tolist()}")

# Custom colors picked to match the image as closely as possible
custom_colors = [
    "#afe5f0",  # Voltage_measured (teal-green)
    "#c7e9c0",  # Current_measured (brown)
    "#7fcdbb",  # Temperature_measured (blue-gray)
    "#41b6c4",  # Current_load (pink-mauve)
    "#2c7fb8",  # Voltage_load (olive-green)
    "#253494",  # Resistance (yellow-olive)
]

# Set up "Q1-level" style: clean, clear, publication font, no y-label, 3 per row
sns.set(style="whitegrid", font_scale=1.45, rc={
    "axes.edgecolor": "0.14",
    "axes.linewidth": 1.7,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "axes.labelpad": 10,
    "axes.titleweight": "bold",
    "legend.frameon": False
})

n_features = len(paper_features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

# Large figure for paper (A4 width, good for publication)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 8.6), sharex=False)
axes = axes.flatten()

for i, feature in enumerate(paper_features):
    ax = axes[i]
    sns.lineplot(
        data=df_plot,
        x='Time_rel',
        y=feature,
        ax=ax,
        color=custom_colors[i],
        linewidth=2.3,
        zorder=3
    )
    # For a clean Q1 appearance, title in color and bold,
    # y-label is omitted, x-label only at the bottom row
    ax.set_title(feature.replace("_", " ").title(), fontsize=18, fontweight='bold', pad=7, color=custom_colors[i])
    ax.set_ylabel("")
    ax.grid(True, alpha=0.23, linestyle="--", zorder=0)
    if i // n_cols == n_rows - 1:  # Only bottom row
        ax.set_xlabel("Time(s)", fontsize=18, fontweight="bold", labelpad=6)
    else:
        ax.set_xlabel("")
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#444")
    ax.set_facecolor("#fff")

# Hide any unused subplots (if n_features not multiple of n_cols)
for j in range(len(paper_features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=(0.01, 0.01, 1, 0.99), pad=2.0, h_pad=2.7, w_pad=2.7)
paper_fig_path = plots_dir / "battery_features_first_cycle_6features_grid_paper_Q1.png"
plt.savefig(paper_fig_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved Q1-paper-ready first cycle 6-feature grid figure to {paper_fig_path}")