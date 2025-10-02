import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(['science','nature'])
plt.rcParams['text.usetex'] = False
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# Get the root directory of the current script
ROOT_DIR = Path(__file__).parent.resolve()

# Get the project root (parent of the script's folder)
PROJECT_ROOT = ROOT_DIR.parent
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Path to original data
csv_path_original = PROJECT_ROOT / 'data_processed/preprocessed_battery_health_dataset_all_points.csv'
# Path to augmented data
csv_path_augmented = PROJECT_ROOT / 'data_augmentation/battery_phys_augmented.csv'

pdf_dir = PROJECT_ROOT / "data_processed"
pdf_path = pdf_dir / "SoH_per_Cycle_with_Augmentation.pdf"
pdf = PdfPages(pdf_path)

# Focus on just one batch - the first one
batch = ['B0005','B0006','B0007','B0018']

# Using your originally specified colors
battery_colors = [
    "#afe5f0",  # Voltage_measured (teal-green)
    "#c7e9c0",  # Current_measured (brown)
    "#7fcdbb",  # Temperature_measured (blue-gray)
    "#41b6c4",  # Current_load (pink-mauve)
    "#2c7fb8",  # Voltage_load (olive-green)
    "#253494",  # Resistance (yellow-olive)
]

# Markers for the batteries
markers = ['o', 'v', 'D', 's']

# Load both datasets
data_original = pd.read_csv(csv_path_original)
data_augmented = pd.read_csv(csv_path_augmented)

# Check if 'battery_id' and 'cycle_number' columns exist in the augmented data
required_columns = ['battery_id', 'cycle_number']
missing_columns = [col for col in required_columns if col not in data_augmented.columns]
if missing_columns:
    print(f"Warning: Augmented data is missing required columns: {missing_columns}")
    print("Make sure both datasets have 'battery_id' and 'cycle_number' columns")
    print("Available columns in augmented data:", data_augmented.columns.tolist())

# Set figure parameters for better quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

fig = plt.figure(figsize=(10, 6))
custom_lines = []
legends = []

for bat_idx, bat in enumerate(batch):
    # Original data
    bat_data_orig = data_original[data_original['battery_id'] == bat]
    if not bat_data_orig.empty:
        # Get only the last row per cycle, which should correspond to end-of-discharge
        last_per_cycle_orig = bat_data_orig.groupby('cycle_number').tail(1)
        cycles_orig = last_per_cycle_orig['cycle_number'].values
        soh_orig = last_per_cycle_orig['SoH'].values
        color = battery_colors[bat_idx % len(battery_colors)]
        marker = markers[bat_idx % len(markers)]
        
        # Plot original data with solid lines
        line = plt.plot(
            cycles_orig, soh_orig,
            color=color,
            alpha=1,
            linewidth=2.0,
            linestyle='-',  # Solid line for original
            marker=marker,
            markersize=5,
            markevery=max(len(cycles_orig)//20, 1),
            label=f"{bat} (Original)"
        )[0]
        
        custom_lines.append(Line2D([0], [0], color=color, linewidth=2.0, linestyle='-', marker=marker, markersize=5))
        legends.append(f"{bat} (Original)")
        
        # Augmented data for the same battery
        if 'battery_id' in data_augmented.columns:
            bat_data_aug = data_augmented[data_augmented['battery_id'] == bat]
            if not bat_data_aug.empty:
                # Get only the last row per cycle for augmented data
                last_per_cycle_aug = bat_data_aug.groupby('cycle_number').tail(1)
                cycles_aug = last_per_cycle_aug['cycle_number'].values
                soh_aug = last_per_cycle_aug['SoH'].values
                
                # Plot augmented data with same color but dotted line
                aug_line = plt.plot(
                    cycles_aug, soh_aug,
                    color=color,  # Same color as original
                    alpha=0.9,
                    linewidth=2.0,
                    linestyle=':',  # Dotted line for augmented
                    marker=marker,
                    markersize=5,
                    markevery=max(len(cycles_aug)//15, 1),
                    label=f"{bat} (Augmented)"
                )[0]
                
                custom_lines.append(Line2D([0], [0], 
                                          color=color, 
                                          linewidth=2.0, 
                                          linestyle=':', 
                                          marker=marker, 
                                          markersize=5))
                legends.append(f"{bat} (Augmented)")

# Add grid with slight transparency for better readability
plt.grid(True, linestyle='--', alpha=0.3)

plt.xlabel('Cycle', fontweight='bold', fontsize=12)
plt.ylabel('State of Health (SoH) [%]', fontweight='bold', fontsize=12)

# Enhanced legend with better layout
legend = plt.legend(custom_lines, legends, loc='center left',
                    bbox_to_anchor=(1.01, 0.5), frameon=True,
                    ncol=1, fontsize=10, facecolor='white', edgecolor='lightgray')
legend.get_frame().set_alpha(0.8)

plt.ylim([40, 110])

# Title
plt.title("Battery SoH Degradation: Original vs Augmented Data", 
          fontweight='bold', fontsize=14, pad=10)

plt.tight_layout()
plt.savefig('battery_soh_comparison.png', bbox_inches='tight')
pdf.savefig(fig)
plt.close(fig)

pdf.close()
print("Plotting complete! PDF saved to:", pdf_path)