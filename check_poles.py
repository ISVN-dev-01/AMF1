import pandas as pd

for split in ['train', 'val', 'test']:
    df = pd.read_parquet(f'data/models/splits/{split}_data.parquet')
    poles = df['is_pole'].sum()
    print(f'{split}: pole positions = {poles}, total records = {len(df)}, unique races = {df["race_id"].nunique()}')
    
    # Check sample data
    print(f'  Sample is_pole values: {df["is_pole"].value_counts().to_dict()}')
    if poles > 0:
        pole_races = df[df['is_pole'] == 1]['race_id'].unique()
        print(f'  Pole races: {pole_races[:5]}...')
    print()