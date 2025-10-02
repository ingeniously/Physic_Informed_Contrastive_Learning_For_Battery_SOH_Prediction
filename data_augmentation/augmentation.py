import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

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
    # SEI growth also increases resistance
    row['Resistance'] = row['Resistance'] * (1 + 0.01 * k * np.sqrt(t + 1e-8))
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
    # Resistance increase affects voltage and current relationship
    if row['Current_measured'] != 0:
        row['Voltage_measured'] = row['Voltage_measured'] - delta_r * abs(row['Current_measured'])
    return row

def augment_temperature_effects(row, temp_range=(-5, 5)):
    """
    Simulate Arrhenius temperature effects on battery parameters.
    Higher temperature accelerates aging and reactions.
    Args:
        row (pd.Series): Original row
        temp_range: Range for temperature adjustment
    Returns:
        pd.Series: Augmented row with temperature effects
    """
    # Activation energy for typical battery processes (in J/mol)
    Ea = 50000
    # Gas constant (in J/(mol·K))
    R = 8.314
    
    # Convert temperature from Celsius to Kelvin
    orig_temp_K = row['Temperature_measured'] + 273.15
    delta_temp = np.random.uniform(*temp_range)
    new_temp_K = orig_temp_K + delta_temp
    row['Temperature_measured'] += delta_temp
    
    # Arrhenius equation for reaction rate change
    rate_ratio = np.exp((Ea/R) * (1/orig_temp_K - 1/new_temp_K))
    
    # Apply temperature effects on SoH, Resistance, and Capacity
    if delta_temp > 0:  # Higher temperature accelerates aging
        row['SoH'] = max(0, row['SoH'] * (1 - 0.005 * rate_ratio))
        row['Resistance'] = row['Resistance'] * (1 - 0.01 * rate_ratio)  # Lower resistance at higher temp
    else:  # Lower temperature slows aging but increases resistance
        row['Resistance'] = row['Resistance'] * (1 + 0.01 * abs(rate_ratio))
    
    # Temperature affects voltage
    row['Voltage_measured'] = row['Voltage_measured'] * (1 + 0.001 * delta_temp)
    return row

def augment_cycle_aging(row, cycle_factor_range=(0.0001, 0.001)):
    """
    Simulate cycle aging based on depth of discharge (approximated from SoC).
    Args:
        row (pd.Series): Original row
        cycle_factor_range: Range for cycle aging factor
    Returns:
        pd.Series: Augmented row with cycle aging effects
    """
    # Use SoC as an indicator of DoD (Depth of Discharge)
    # Assume: Lower SoC = higher DoD = more stress on battery
    DoD = 1.0 - (row['SoC'] / 100.0)
    
    # Cycle aging factor: higher DoD causes more degradation
    cycle_factor = np.random.uniform(*cycle_factor_range) 
    
    # Nonlinear impact of DoD on degradation (higher DoD causes more damage)
    DoD_impact = DoD**1.5  # Nonlinear relationship
    
    # Apply cycle aging to SoH and Capacity
    soh_reduction = cycle_factor * DoD_impact * row['Time_norm'] 
    row['SoH'] = max(0, row['SoH'] - soh_reduction * 100)  # Convert to percentage
    
    # Cycle aging also impacts resistance
    row['Resistance'] = row['Resistance'] * (1 + 0.005 * soh_reduction)
    return row

def augment_dynamic_load(row, pulse_range=(0.05, 0.25)):
    """
    Simulate dynamic load effects (pulses, variable current).
    Args:
        row (pd.Series): Original row
        pulse_range: Range for current pulse magnitude
    Returns:
        pd.Series: Augmented row with dynamic load effects
    """
    # Apply current pulse
    pulse_magnitude = np.random.uniform(*pulse_range) * np.sign(row['Current_measured'])
    row['Current_measured'] = row['Current_measured'] + pulse_magnitude
    
    # Update Current_load accordingly
    row['Current_load'] = row['Current_load'] + pulse_magnitude
    
    # Voltage response to current change (simple IR relationship)
    row['Voltage_measured'] = row['Voltage_measured'] - pulse_magnitude * row['Resistance']
    
    # Dynamic loads can slightly impact SoC
    row['SoC'] = max(0, min(100, row['SoC'] - 0.1 * abs(pulse_magnitude)))
    return row

def correlate_capacity_soh(row):
    """
    Ensure correlation between Capacity and SoH remains physically consistent.
    Args:
        row (pd.Series): Original row
    Returns:
        pd.Series: Row with consistent Capacity-SoH relationship
    """
    # Assuming nominal capacity is when SoH=100%
    # Calculate nominal capacity based on current capacity and SoH
    nominal_capacity = row['Capacity'] / (row['SoH']/100)
    
    # Recalculate capacity based on current SoH to maintain consistency
    row['Capacity'] = nominal_capacity * (row['SoH']/100)
    return row

def apply_physical_constraints(row):
    """
    Apply physical constraints to ensure physically valid values.
    Args:
        row (pd.Series): Row with augmented values
    Returns:
        pd.Series: Row with physically valid values
    """
    # SoH constraints
    row['SoH'] = max(0, min(100, row['SoH']))
    
    # SoC constraints
    row['SoC'] = max(0, min(100, row['SoC']))
    
    # Resistance cannot be negative
    row['Resistance'] = max(1e-6, row['Resistance'])
    
    # Capacity cannot be negative
    row['Capacity'] = max(1e-6, row['Capacity'])
    
    # Voltage-Current-Resistance relationship check
    if abs(row['Current_measured']) > 1e-6:
        # Ensure V = IR relationship is somewhat maintained
        expected_voltage = row['Voltage_load'] - row['Current_measured'] * row['Resistance']
        row['Voltage_measured'] = (row['Voltage_measured'] + expected_voltage) / 2
    
    return row

def multivariate_augmentation(row):
    """
    Apply multiple aging mechanisms simultaneously with physical correlations.
    Args:
        row (pd.Series): Original row
    Returns:
        pd.Series: Augmented row with multiple mechanisms
    """
    # Copy the row to avoid modifying the original
    new_row = row.copy()
    
    # Randomly select 2-4 mechanisms to apply simultaneously
    mechanisms = np.random.choice(
        ['sei', 'calendar', 'resistance', 'temperature', 'cycle', 'dynamic'],
        size=np.random.randint(2, 5),
        replace=False
    )
    
    # Apply selected mechanisms
    for mech in mechanisms:
        if mech == 'sei':
            new_row = augment_sei_growth(new_row)
        elif mech == 'calendar':
            new_row = augment_calendar_aging(new_row)
        elif mech == 'resistance':
            new_row = augment_resistance_increase(new_row)
        elif mech == 'temperature':
            new_row = augment_temperature_effects(new_row)
        elif mech == 'cycle':
            new_row = augment_cycle_aging(new_row)
        elif mech == 'dynamic':
            new_row = augment_dynamic_load(new_row)
    
    # Ensure physical correlations are maintained
    new_row = correlate_capacity_soh(new_row)
    new_row = apply_physical_constraints(new_row)
    
    return new_row

def generate_trajectory_variations(df, n_trajectories=3, smoothness=0.8):
    """
    Generate entire aging trajectories with consistent aging patterns.
    Args:
        df (pd.DataFrame): Original dataframe
        n_trajectories: Number of trajectory variations to create
        smoothness: How smooth the variations should be (0-1)
    Returns:
        pd.DataFrame: Dataframe with trajectory variations
    """
    # Group by time to identify trajectory points
    time_points = df['Time_norm'].unique()
    time_points.sort()
    
    # Generate variations
    all_variations = []
    
    for _ in range(n_trajectories):
        # Create random aging trajectory modifiers
        # More smoothness = more correlated random values
        random_offsets = np.zeros(len(time_points))
        
        # Generate first point randomly
        random_offsets[0] = np.random.normal(0, 0.05)
        
        # Generate subsequent points with correlation to previous point
        for i in range(1, len(time_points)):
            random_offsets[i] = smoothness * random_offsets[i-1] + (1-smoothness) * np.random.normal(0, 0.05)
        
        # Create interpolation function
        offset_func = interp1d(time_points, random_offsets, fill_value="extrapolate")
        
        # Apply to dataframe
        df_copy = df.copy()
        
        # Apply offset to SoH based on time
        df_copy['SoH'] = df_copy.apply(
            lambda row: max(0, min(100, row['SoH'] * (1 + offset_func(row['Time_norm'])))), 
            axis=1
        )
        
        # Apply consistent changes to related parameters
        df_copy['Capacity'] = df_copy.apply(
            lambda row: max(0.001, row['Capacity'] * (1 + offset_func(row['Time_norm']))), 
            axis=1
        )
        
        df_copy['Resistance'] = df_copy.apply(
            lambda row: max(0.001, row['Resistance'] * (1 - 0.5 * offset_func(row['Time_norm']))), 
            axis=1
        )
        
        all_variations.append(df_copy)
    
    # Combine all variations
    result = pd.concat(all_variations, ignore_index=True)
    return result

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
    
    # First add row-wise augmentations
    for i in range(n_aug):
        for idx, row in df.iterrows():
            row_aug = multivariate_augmentation(row)
            aug_rows.append(row_aug)
    
    row_aug_df = pd.DataFrame(aug_rows, columns=df.columns)
    
    # Then add trajectory-based augmentations
    trajectory_aug_df = generate_trajectory_variations(df, n_trajectories=2)
    
    # Combine both types of augmentations
    combined_df = pd.concat([row_aug_df, trajectory_aug_df], ignore_index=True)
    return combined_df

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
        #     'cycle_number',       # Cycle index
           #  'battery_id',         # Battery ID  
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
    print("\nFeatures in the final (physically augmented) dataset:")
    for feat in feature_list:
        print(f" - {feat}")

"""
All augmentations above are PHYSICALLY INSPIRED, not just random noise:
- SEI and calendar aging change SoH according to real battery degradation laws
- Resistance increase models age-dependent internal resistance rise
- Temperature effects follow Arrhenius equation for reaction kinetics
- Cycle aging accounts for depth-of-discharge impacts
- Dynamic load effects simulate real-world usage patterns
- Physical correlations between parameters are maintained
- Trajectory variations capture consistent aging patterns over time
"""