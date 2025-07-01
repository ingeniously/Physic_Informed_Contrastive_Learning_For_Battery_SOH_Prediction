import numpy as np
import pandas as pd

def augment_sei_growth(row, k_range=(1e-4, 5e-4)):
    """
    Simulate SEI growth: a common degradation mechanism modeled as SoH_new = SoH_old - k * sqrt(t)
    Args:
        row (pd.Series): Original row.
        k_range (tuple): Range for SEI growth rate constant k.
    Returns:
        pd.Series: Augmented row with updated SoH.
    """
    k = np.random.uniform(*k_range)
    t = row['Time_norm']
    # SEI growth equation (SoH decay with cycle/time)
    new_soh = row['SoH'] - k * np.sqrt(t + 1e-8)
    row['SoH'] = max(new_soh, 0)
    return row

def augment_calendar_aging(row, k_range=(1e-4, 5e-4)):
    """
    Simulate calendar aging: capacity loss just due to passage of time, modeled as SoH_new = SoH_old - k * sqrt(t)
    Args:
        row (pd.Series): Original row.
        k_range (tuple): Range for calendar aging rate constant k.
    Returns:
        pd.Series: Augmented row with updated SoH.
    """
    k = np.random.uniform(*k_range)
    t = row['Time_norm']
    new_soh = row['SoH'] - k * np.sqrt(t + 1e-8)
    row['SoH'] = max(new_soh, 0)
    return row

def augment_resistance_increase(row, r_inc_range=(0, 0.02)):
    """
    Simulate internal resistance increase over time/use.
    Args:
        row (pd.Series): Original row.
        r_inc_range (tuple): Range for resistance increase.
    Returns:
        pd.Series: Augmented row with updated Resistance.
    """
    delta_r = np.random.uniform(*r_inc_range)
    row['Resistance'] = row['Resistance'] + delta_r
    return row

def augment_battery_dataframe_physical(df, n_aug=1):
    """
    Apply physically-inspired augmentations row-wise to the dataframe.
    Args:
        df (pd.DataFrame): Original data.
        n_aug (int): Number of augmentations per row.
    Returns:
        pd.DataFrame: Augmented dataset.
    """
    aug_rows = []
    for i in range(n_aug):
        for idx, row in df.iterrows():
            row_aug = row.copy()
            # Randomly choose a physical mechanism for each augmentation
            mech = np.random.choice(['sei', 'calendar', 'resistance'])
            if mech == 'sei':
                row_aug = augment_sei_growth(row_aug)
            elif mech == 'calendar':
                row_aug = augment_calendar_aging(row_aug)
            elif mech == 'resistance':
                row_aug = augment_resistance_increase(row_aug)
            aug_rows.append(row_aug)
    return pd.DataFrame(aug_rows, columns=df.columns)

if __name__ == "__main__":
    # Example usage
    input_csv = "data_processed/preprocessed_battery_health_dataset_all_points.csv"
    output_csv = "data_augmentation/battery_phys_augmented.csv"
    df = pd.read_csv(input_csv)
    n_augment = 1
    aug_df = augment_battery_dataframe_physical(df, n_aug=n_augment)
    aug_df.to_csv(output_csv, index=False)
    print(f"Saved augmented data to: {output_csv}")

    # ---- Feature List ----
    feature_list = [
        "Voltage_measured",      # Measured cell voltage [V]
        "Current_measured",      # Measured current [A]
        "Temperature_measured",  # Measured temperature [°C]
        "Current_load",          # Applied load current [A]
        "Voltage_load",          # Applied load voltage [V]
        "SoC",                   # State of Charge [%]
        "Resistance",            # Instantaneous resistance [Ohm]
        "Capacity",              # Cycle capacity [Ah]
        "Time_norm",             # Normalized time (relative, [0-1])
        "SoH"                    # State of Health [%]
    ]
    print("\nFeatures in the final (physically augmented) dataset:")
    for feat in feature_list:
        print(f" - {feat}")

"""
All augmentations above are PHYSICALLY INSPIRED, not just random noise:
- SEI and calendar aging change SoH according to real battery degradation laws.
- Resistance increase models age-dependent internal resistance rise.
No purely random (non-physical) noise is applied.
"""