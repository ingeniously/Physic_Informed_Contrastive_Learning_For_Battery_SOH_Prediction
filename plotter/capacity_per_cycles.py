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
csv_path = PROJECT_ROOT / 'data_processed/preprocessed_battery_health_dataset_all_points.csv'
pdf_dir = PROJECT_ROOT / "data_processed"
pdf_path = pdf_dir / "SoH_per_Cycle.pdf"
pdf = PdfPages(pdf_path)

# 7 batches: each is a list of battery IDs
batches = [
    ['B0005','B0006','B0007','B0018'],
    ['B0025','B0026','B0027','B0028'],
    ['B0029','B0030','B0031','B0032'],
    ['B0033','B0034','B0036'],
    ['B0038','B0039','B0040'],
    ['B0042','B0043','B0044'],
    ['B0046','B0047','B0048']
]

# Many colors for per-battery coloring (cycle for each batch)
battery_colors = [
    "#80A6E2", "#FBAED2", "#98DDCA", "#403990", "#FFD3B4", "#B5DEFF", "#B5B5B5",
    "#7BDFF2", "#FBDD85", "#F7756D", "#E7D3AD", "#A4E3B9", "#D3B5E5", "#B5EAD7", "#FFDAC1", "#FFB7B2"
]
# Markers for up to 4 batteries per batch
batch_markers = [
    ['o', 'v', 'D', 's'],
    ['^', '<', '>', 'p'],
    ['*', 'h', 'H', 'X'],
    ['d', 'P', '8'],
    ['1', '2', '3'],
    ['4', '|', '_'],
    ['.', ',', '+']
]

data = pd.read_csv(csv_path)

for batch_idx, batch in enumerate(batches):
    fig = plt.figure(figsize=(8, 4), dpi=200)
    custom_lines = []
    legends = []
    colors = battery_colors[:len(batch)]  # take as many as needed for the batch
    markers = batch_markers[batch_idx]
    for bat_idx, bat in enumerate(batch):
        bat_data = data[data['battery_id'] == bat]# go back to the data_analysis/data_processing.py file and uncomment 'battery' for plotting
        if bat_data.empty:
            continue
        # Get only the last row per cycle, which should correspond to end-of-discharge
        last_per_cycle = bat_data.groupby('cycle_number').tail(1)
        cycles = last_per_cycle['cycle_number'].values
        soh = last_per_cycle['SoH'].values
        color = colors[bat_idx % len(colors)]
        marker = markers[bat_idx % len(markers)]
        plt.plot(
            cycles, soh,
            color=color,
            alpha=1,
            linewidth=1.0,
            marker=marker,
            markersize=2,
            markevery=max(len(cycles)//20, 1)
        )
        custom_lines.append(Line2D([0], [0], color=color, linewidth=1.0, marker=marker, markersize=2.5))
        legends.append(bat)

    plt.xlabel('Cycle')
    plt.ylabel('State of Health (SoH) [%]')
    plt.legend(custom_lines, legends, loc='center left',
               bbox_to_anchor=(1.01, 0.5), frameon=False,
               ncol=1, fontsize=6)
    plt.ylim([40, 110])
    plt.title(f"Batch {batch_idx+1}: " + ", ".join(batch))
    plt.tight_layout()
    plt.show()
    pdf.savefig(fig)
   # plt.savefig(f'trajectory_batch_{batch_idx+1}.svg', format='svg')
    plt.close(fig)

pdf.close()