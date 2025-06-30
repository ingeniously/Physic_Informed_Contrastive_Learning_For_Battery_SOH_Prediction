import os
import warnings
# Suppress specific FutureWarning from pandas about swapaxes
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame.swapaxes.*")

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Get the root directory of the current script
ROOT_DIR = Path(__file__).parent.resolve()

# Get the project root (parent of the script's folder)
PROJECT_ROOT = ROOT_DIR.parent
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

# Process each discharge cycle
for _, row in tqdm(discharge_metadata.iterrows(), total=len(discharge_metadata)):
    file_path = PROJECT_ROOT / f"data/data/{row['filename']}"
    df = pd.read_csv(file_path).copy()
    
    # Truncate(remove) data at the first point where voltage < 2.7V
    # Discharges were carried out at different current load levels until the battery voltage fell to preset voltage thresholds. 
    # Some of these thresholds were lower than that recommended by the OEM (2.7 V) in order to induce deep discharge aging effects
    # They said in the official docu that sometime they went bellow 2.7 to induce deep discharge..But we don't reall focus on deep dischage,so we filter it
    df = df[df['Current_measured'] <= 0]
    cutoff_idx = df[df['Voltage_measured'] < 2.7].index.min()
    truncated_df = df if pd.isna(cutoff_idx) else df.iloc[:cutoff_idx].copy()
    
    # Compute capacity using coulomb counting (sum of current over time)
    # For each row, multiplies the current (in amperes) by the time interval (in hours) to get the amount of charge (in ampere-hours, Ah) moved during that interval
    truncated_df['Time_difference_hr'] = truncated_df['Time'].diff().fillna(0) / 3600
    truncated_df['Delta_Q'] = truncated_df['Current_measured'] * truncated_df['Time_difference_hr']
    capacity = abs(truncated_df['Delta_Q'].sum())
    
    # Only use cycles with capacity above threshold (filter out bad data)"Mentionned in officiial https://c3.ndc.nasa.gov/dashlink/resources/133/ "
    #The experiments were stopped when the batteries reached the end-of-life (EOL) criteria of 30% fade in rated capacity (from 2 Ah to 1.4 Ah).
    if capacity > 1.4:
        # Add battery and cycle info columns
        truncated_df['battery_id'] = row['battery_id']
        truncated_df['cycle_number'] = row['cycle_number']
        # Calculate SoC using coulomb counting
        truncated_df['Cumulative_Q'] = truncated_df['Delta_Q'].cumsum()
        truncated_df['SoC'] = 100 * (1 + truncated_df['Cumulative_Q'] / capacity)
        # Calculate SoH as percentage of nominal capacity (2.0 Ah)
        soh_value = (capacity / 2.0) * 100
        truncated_df['SoH'] = soh_value
        
        # Select only the required columns (keep everything, do not downsample)
        selected_columns = [
            'Voltage_measured',
            'Current_measured',
            'Temperature_measured',
            'Current_load',
            'Voltage_load',
            'SoC',
            'cycle_number',
            #'battery_id',
            'SoH'
        ]
        # Only keep columns that exist in the dataframe to avoid KeyError
        columns_to_keep = [col for col in selected_columns if col in truncated_df.columns]
        processed_dfs.append(truncated_df[columns_to_keep])

# --- NEW CODE: create data_processed folder and save output there ---
data_processed_dir = PROJECT_ROOT / "data_processed"
data_processed_dir.mkdir(parents=True, exist_ok=True)
output_csv_path = data_processed_dir / "preprocessed_battery_health_dataset_all_points.csv"

# Concatenate all cycles and save as new CSV
if processed_dfs:
    full_dataset = pd.concat(processed_dfs)
    print(f"Final preprocessed dataset shape: {full_dataset.shape}")
    full_dataset.to_csv(output_csv_path, index=False)
    print(f"Saved to {output_csv_path}")
else:
    print("No data to save.")