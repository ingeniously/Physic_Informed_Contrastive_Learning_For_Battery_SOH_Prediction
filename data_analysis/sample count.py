import pandas as pd
import os
from collections import defaultdict

NASA = '/home/choi/PI_CL/data/preprocessed_battery_health_dataset_all_points.csv'


total_numbers = 0
battery_numbers = 0
battery_cycle_counts = defaultdict(set)  # battery: set of unique cycles


df = pd.read_csv(NASA)
nums = df.shape[0]
        # Group by Battery, collect unique cycle_number
for battery, group in df.groupby("battery_id"):
        battery_cycle_counts[battery].update(group["cycle_number"].unique())

print('total batteries:', len(battery_cycle_counts))
print("\nUnique cycles per battery:")
for battery in sorted(battery_cycle_counts.keys()):
    print(f"Battery {battery}: {len(battery_cycle_counts[battery])} cycles")