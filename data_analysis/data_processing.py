import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Suppress specific FutureWarning from pandas about swapaxes
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame.swapaxes.*")

# Get the root directory of the current script
ROOT_DIR = Path(__file__).parent.resolve()

# Get the project root (parent of the script's folder)
PROJECT_ROOT = ROOT_DIR.parent

# Set CUDA device (optional, for downstream ML use)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Path to the metadata file
metadata_path = PROJECT_ROOT / 'data/metadata.csv'

# Load and preprocess metadata
metadata = pd.read_csv(metadata_path)
metadata['battery_id'] = metadata['battery_id'].astype(str)

# Exclude problematic batteries
# Mentioned in the extra_infos/README_49_50_51_52.txt
# ("The experiments were carried out until the experiment control software crashed")
excluded_batteries = ['B0049', 'B0050', 'B0051', 'B0052']

# Filter to only discharge cycles, excluding the listed batteries
discharge_metadata = metadata[
    (metadata['type'] == 'discharge') &
    (~metadata['battery_id'].isin(excluded_batteries))
].copy()

# Assign a sequential cycle number per battery
discharge_metadata['cycle_number'] = discharge_metadata.groupby('battery_id').cumcount() + 1

processed_dfs = []

# Main data processing loop for each discharge cycle
for _, row in tqdm(discharge_metadata.iterrows(), total=len(discharge_metadata)):
    file_path = PROJECT_ROOT / f"data/data/{row['filename']}"
    df = pd.read_csv(file_path).copy()

    # Keep only discharge part: Current_measured <= 0
    df = df[df['Current_measured'] <= 0]

    # Truncate at the first occurrence where voltage < 2.7V to avoid deep discharge
    cutoff_idx = df[df['Voltage_measured'] < 2.7].index.min()
    truncated_df = df if pd.isna(cutoff_idx) else df.iloc[:cutoff_idx].copy()

    # --- Professional handling of Time features ---
    # Relative time: starts at zero for each cycle
    truncated_df['Time_rel'] = truncated_df['Time'] - truncated_df['Time'].iloc[0]
    # Normalized time: scaled to [0, 1] for each cycle
    cycle_time = truncated_df['Time_rel'].max()
    truncated_df['Time_norm'] = truncated_df['Time_rel'] / cycle_time if cycle_time > 0 else truncated_df['Time_rel']

    # --- Capacity calculation ---
    # Use existing 'Capacity' column if present and valid, else compute via coulomb counting
    if 'Capacity' in truncated_df.columns and np.isfinite(truncated_df['Capacity']).all():
        capacity = float(truncated_df['Capacity'].max())
    else:
        # Compute capacity using coulomb counting (Ah)
        truncated_df['Time_difference_hr'] = truncated_df['Time'].diff().fillna(0) / 3600
        truncated_df['Delta_Q'] = truncated_df['Current_measured'] * truncated_df['Time_difference_hr']
        capacity = abs(truncated_df['Delta_Q'].sum())
        truncated_df['Capacity'] = capacity

    # Only use cycles with capacity above threshold (filter out bad data)
    # Experiments were stopped at 30% fade in rated capacity (from 2 Ah to 1.4 Ah)
    if capacity > 1.4:
        # Add battery and cycle info columns
        truncated_df['battery_id'] = row['battery_id']
        truncated_df['cycle_number'] = row['cycle_number']

        # State of Health (SoH): (cycle capacity / nominal capacity) x 100
        truncated_df['SoH'] = (capacity / 2.0) * 100  # Nominal capacity = 2.0 Ah

        # Cumulative charge (for SoC calculation)
        if 'Delta_Q' in truncated_df:
            truncated_df['Cumulative_Q'] = truncated_df['Delta_Q'].cumsum()
            # State of Charge (SoC) as percentage
            truncated_df['SoC'] = 100 * (1 + truncated_df['Cumulative_Q'] / capacity) if capacity > 0 else 0
        else:
            truncated_df['Cumulative_Q'] = 0
            truncated_df['SoC'] = 0

        # --- Resistance calculation (Ohm's Law: R = V / I) ---
        # To avoid division by zero, replace zero currents with np.nan temporarily
        truncated_df['Resistance'] = truncated_df['Voltage_measured'] / truncated_df['Current_measured'].replace(0, np.nan)
        # Replace inf and NaNs with 0 (since Resistance is undefined when Current=0)
        truncated_df['Resistance'] = truncated_df['Resistance'].replace([np.inf, -np.inf], np.nan).fillna(0)
  
        # Select columns for output (including all engineered and raw features)
        selected_columns = [
          #   'cycle_number',       # Cycle index
          #   'battery_id',         # Battery ID  
         #    'Time',               # Absolute time
         #    'Time_rel',           # Relative time (cycle-based)        
            'Voltage_measured',
            'Current_measured',
            'Temperature_measured',
            'Current_load',
            'Voltage_load',
            'SoC',                 # State of Charge (%)
            'Resistance',         # Instantaneous resistance (Ohm)
            'Capacity',            # Cycle capacity (Ah)
            'Time_norm',           # Normalized relative time
            'SoH',                # State of Health (%)
        ]
        # Only keep columns that exist in the dataframe to avoid KeyError
        columns_to_keep = [col for col in selected_columns if col in truncated_df.columns]
        processed_dfs.append(truncated_df[columns_to_keep])

# --- Save processed data ---
data_processed_dir = PROJECT_ROOT / "data_processed"
data_processed_dir.mkdir(parents=True, exist_ok=True)
output_csv_path = data_processed_dir / "preprocessed_battery_health_dataset_all_points.csv"

# Concatenate all processed cycles and save as a new CSV file
if processed_dfs:
    full_dataset = pd.concat(processed_dfs)
    print(f"Final preprocessed dataset shape: {full_dataset.shape}")
    full_dataset.to_csv(output_csv_path, index=False)
    print(f"Saved to {output_csv_path}")
else:
    print("No data to save.")