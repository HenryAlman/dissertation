from pathlib import Path
import pandas as pd
import json

main_folder = Path(r"C:/Users/Amis Thysia/Documents/Napier Masters Stuff/Dissertation/dissertation_logs_actual_3")

for json_path in main_folder.rglob("metrics.json"):
    csv_path = json_path.with_suffix(".csv")

    with open(json_path, 'r') as file:
        raw_data = json.load(file)
        
    records = []
    for metric_name, value in raw_data.items():
        for x_val, y_val in zip(value['x'], value['y']):
            
            if isinstance(x_val, list):
                x_val = x_val[0] if x_val else None  # Takes 1st item or None if empty
                
            if isinstance(y_val, list):
                y_val = y_val[0] if y_val else None

            records.append({'x': x_val, 'metric': metric_name, 'y': y_val})
    
    df_raw = pd.DataFrame(records)
        
    df = df_raw.pivot(index='x', columns='metric', values='y').reset_index()
    df.columns.name = None

    df.to_csv(csv_path, index=False)
