#!/usr/bin/env python3
"""
Create sample F1 data for testing the feature pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

def create_sample_data():
    """Create realistic sample F1 data for testing"""
    
    # Create directories
    data_dir = Path('data')
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample drivers and teams
    drivers = [
        ('verstappen', 'red_bull'), ('perez', 'red_bull'),
        ('hamilton', 'mercedes'), ('russell', 'mercedes'),
        ('leclerc', 'ferrari'), ('sainz', 'ferrari'),
        ('norris', 'mclaren'), ('piastri', 'mclaren'),
        ('alonso', 'aston_martin'), ('stroll', 'aston_martin')
    ]
    
    # Sample circuits
    circuits = [
        'bahrain', 'saudi_arabia', 'australia', 'japan', 'china',
        'miami', 'emilia_romagna', 'monaco', 'canada', 'spain',
        'austria', 'britain', 'hungary', 'belgium', 'netherlands'
    ]
    
    # Generate race calendar (2024 season)
    start_date = datetime(2024, 3, 1)  # Season starts March
    races = []
    
    for i, circuit in enumerate(circuits):
        race_date = start_date + timedelta(weeks=i*2)  # Races every 2 weeks
        race_id = f"2024_{i+1:02d}"
        
        races.append({
            'race_id': race_id,
            'date_utc': race_date,
            'circuit_id': circuit,
            'season': 2024,
            'round': i+1
        })
    
    print(f"📅 Creating sample data for {len(races)} races...")
    
    # Generate master dataset
    master_data = []
    
    for race in races:
        race_id = race['race_id']
        race_date = race['date_utc']
        circuit_id = race['circuit_id']
        
        # Generate results for each driver
        for driver_id, team_id in drivers:
            
            # Qualifying position (1-20)
            quali_rank = random.randint(1, 20)
            
            # Race position (influenced by quali, but with some randomness)
            position_variance = random.randint(-5, 8)
            race_position = max(1, min(20, quali_rank + position_variance))
            
            # Status
            if random.random() < 0.05:  # 5% DNF rate
                status = 'dnf'
            elif random.random() < 0.02:  # 2% DNS rate  
                status = 'dns'
            else:
                status = 'finished'
            
            # Practice times (FP3)
            base_lap_time = 75 + random.uniform(-2, 3)  # 73-78 second lap times
            fp3_time = base_lap_time + random.uniform(-1, 2)
            
            master_data.append({
                'race_id': race_id,
                'driver_id': driver_id,
                'team_id': team_id,
                'date_utc': race_date,
                'circuit_id': circuit_id,
                'season': 2024,
                'round': race['round'],
                'quali_rank': quali_rank,
                'race_position': race_position,
                'status': status,
                'session_type': 'race',
                'LapTimeSeconds': fp3_time,
                'is_pole': 1 if quali_rank == 1 else 0,
                'is_race_winner': 1 if race_position == 1 else 0
            })
    
    # Create master dataset
    df_master = pd.DataFrame(master_data)
    df_master['date_utc'] = pd.to_datetime(df_master['date_utc'])
    
    print(f"📊 Master dataset: {len(df_master)} records")
    
    # Create labels dataset
    labels_data = []
    
    for _, row in df_master.iterrows():
        labels_data.append({
            'race_id': row['race_id'],
            'driver_id': row['driver_id'],
            'is_pole': row['is_pole'],
            'is_race_winner': row['is_race_winner'],
            'quali_best_time': 75 + random.uniform(-2, 3),  # Qualifying time in seconds
            'race_position': row['race_position']
        })
    
    df_labels = pd.DataFrame(labels_data)
    
    print(f"🏷️  Labels dataset: {len(df_labels)} records")
    
    # Save datasets
    master_file = processed_dir / 'master_dataset.parquet'
    labels_file = processed_dir / 'labels.parquet'
    
    df_master.to_parquet(master_file, index=False)
    df_labels.to_parquet(labels_file, index=False)
    
    print(f"💾 Saved master dataset: {master_file}")
    print(f"💾 Saved labels: {labels_file}")
    
    return df_master, df_labels

if __name__ == "__main__":
    df_master, df_labels = create_sample_data()
    print("✅ Sample data created successfully!")