import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for even better visuals
sns.set(style="ticks", font_scale=1.18, rc={
    "axes.edgecolor": "0.15",
    "axes.linewidth": 1.2,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "axes.labelpad": 8,
})

csv_file = "/home/choi/PICL/data_processed/preprocessed_battery_health_dataset_all_points.csv"
df = pd.read_csv(csv_file, sep=",")
df.columns = df.columns.str.strip()

if 'Time_rel' not in df.columns:
    raise ValueError("The column 'Time_rel' was not found. Actual columns: {}".format(df.columns.tolist()))

time_zeros = df.index[df['Time_rel'] == 0.0].tolist()
if len(time_zeros) > 1:
    first_zero = time_zeros[0]
    second_zero = time_zeros[1]
    df_plot = df.iloc[first_zero:second_zero]
else:
    df_plot = df

# Exclude 'SoC' and 'Capacity' from the features
features = [col for col in df.columns if col not in ('Time_rel', 'SoC', 'Capacity')]

n_features = len(features)
n_cols = 1
n_rows = n_features

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, n_rows * 2.5), sharex=False)
if n_features == 1:
    axes = [axes]  # Make it iterable

palette = sns.color_palette("Set2", n_colors=len(features))

for i, feature in enumerate(features):
    ax = axes[i]
    sns.lineplot(
        data=df_plot,
        x='Time_rel',
        y=feature,
        ax=ax,
        color=palette[i % len(palette)],
        linewidth=2.2,
        zorder=3,
    )
    ax.set_title(feature, fontsize=13, fontweight='semibold', color=palette[i % len(palette)])
    ax.set_ylabel("")
    ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
    ax.set_xlabel("", fontsize=11)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#888")
    ax.set_facecolor("#fafbfc")

fig.supxlabel("Time_rel (s)", fontsize=14, y=0.02, fontweight='bold')
fig.supylabel("Feature Value", fontsize=14, x=0.01, fontweight='bold')
fig.suptitle("Battery Feature Variation Over Time (First Cycle)", fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=(0.025, 0.04, 1, 0.96))
plt.subplots_adjust(hspace=0.35)
plt.savefig("battery_features_over_time_vertical.png", dpi=300, bbox_inches="tight")
plt.show()