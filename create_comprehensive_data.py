#!/usr/bin/env python3
"""
Create comprehensive multi-year F1 sample data for Phase 5 testing
Includes seasons 2014-2024 for proper time-aware splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

def create_comprehensive_f1_data():
    """Create realistic multi-year F1 data spanning 2014-2024"""
    
    # Create directories
    data_dir = Path('data')
    processed_dir = data_dir / 'processed'
    features_dir = data_dir / 'features'
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Realistic F1 driver evolution over years
    drivers_by_year = {
        2014: [('hamilton', 'mercedes'), ('rosberg', 'mercedes'), ('vettel', 'red_bull'), 
               ('ricciardo', 'red_bull'), ('alonso', 'ferrari'), ('raikkonen', 'ferrari'),
               ('bottas', 'williams'), ('massa', 'williams'), ('button', 'mclaren'), ('magnussen', 'mclaren')],
        2015: [('hamilton', 'mercedes'), ('rosberg', 'mercedes'), ('vettel', 'ferrari'), 
               ('raikkonen', 'ferrari'), ('ricciardo', 'red_bull'), ('kvyat', 'red_bull'),
               ('bottas', 'williams'), ('massa', 'williams'), ('button', 'mclaren'), ('alonso', 'mclaren')],
        2016: [('hamilton', 'mercedes'), ('rosberg', 'mercedes'), ('vettel', 'ferrari'), 
               ('raikkonen', 'ferrari'), ('ricciardo', 'red_bull'), ('verstappen', 'red_bull'),
               ('bottas', 'williams'), ('massa', 'williams'), ('button', 'mclaren'), ('alonso', 'mclaren')],
        2017: [('hamilton', 'mercedes'), ('bottas', 'mercedes'), ('vettel', 'ferrari'), 
               ('raikkonen', 'ferrari'), ('ricciardo', 'red_bull'), ('verstappen', 'red_bull'),
               ('massa', 'williams'), ('stroll', 'williams'), ('alonso', 'mclaren'), ('vandoorne', 'mclaren')],
        2018: [('hamilton', 'mercedes'), ('bottas', 'mercedes'), ('vettel', 'ferrari'), 
               ('raikkonen', 'ferrari'), ('ricciardo', 'red_bull'), ('verstappen', 'red_bull'),
               ('russell', 'williams'), ('stroll', 'williams'), ('alonso', 'mclaren'), ('vandoorne', 'mclaren')],
        2019: [('hamilton', 'mercedes'), ('bottas', 'mercedes'), ('vettel', 'ferrari'), 
               ('leclerc', 'ferrari'), ('ricciardo', 'red_bull'), ('verstappen', 'red_bull'),
               ('russell', 'williams'), ('kubica', 'williams'), ('norris', 'mclaren'), ('sainz', 'mclaren')],
        2020: [('hamilton', 'mercedes'), ('bottas', 'mercedes'), ('vettel', 'ferrari'), 
               ('leclerc', 'ferrari'), ('ricciardo', 'red_bull'), ('verstappen', 'red_bull'),
               ('russell', 'williams'), ('latifi', 'williams'), ('norris', 'mclaren'), ('sainz', 'mclaren')],
        2021: [('hamilton', 'mercedes'), ('bottas', 'mercedes'), ('leclerc', 'ferrari'), 
               ('sainz', 'ferrari'), ('perez', 'red_bull'), ('verstappen', 'red_bull'),
               ('russell', 'williams'), ('latifi', 'williams'), ('norris', 'mclaren'), ('ricciardo', 'mclaren')],
        2022: [('hamilton', 'mercedes'), ('russell', 'mercedes'), ('leclerc', 'ferrari'), 
               ('sainz', 'ferrari'), ('perez', 'red_bull'), ('verstappen', 'red_bull'),
               ('albon', 'williams'), ('latifi', 'williams'), ('norris', 'mclaren'), ('ricciardo', 'mclaren')],
        2023: [('hamilton', 'mercedes'), ('russell', 'mercedes'), ('leclerc', 'ferrari'), 
               ('sainz', 'ferrari'), ('perez', 'red_bull'), ('verstappen', 'red_bull'),
               ('albon', 'williams'), ('sargeant', 'williams'), ('norris', 'mclaren'), ('piastri', 'mclaren')],
        2024: [('hamilton', 'mercedes'), ('russell', 'mercedes'), ('leclerc', 'ferrari'), 
               ('sainz', 'ferrari'), ('perez', 'red_bull'), ('verstappen', 'red_bull'),
               ('albon', 'williams'), ('sargeant', 'williams'), ('norris', 'mclaren'), ('piastri', 'mclaren')]
    }
    
    # Realistic F1 calendar (20 races per year approximately)
    circuits = [
        'australia', 'bahrain', 'china', 'japan', 'spain', 
        'monaco', 'canada', 'britain', 'germany', 'hungary',
        'belgium', 'italy', 'singapore', 'russia', 'usa',
        'mexico', 'brazil', 'azerbaijan', 'france', 'austria'
    ]
    
    print(f"🏗️  Creating comprehensive F1 data (2014-2024)...")
    
    # Generate complete dataset
    all_data = []
    
    for year in range(2014, 2025):  # 2014-2024 inclusive
        drivers = drivers_by_year[year]
        num_races = 20 if year >= 2016 else 19  # Realistic race count evolution
        
        print(f"   📅 Generating {year} season ({num_races} races, {len(drivers)} drivers)")
        
        # Generate race calendar for year
        start_month = 3 if year >= 2020 else 3  # Season typically starts March
        base_date = datetime(year, start_month, 1)
        
        for race_num in range(1, num_races + 1):
            # Race every 2-3 weeks
            race_date = base_date + timedelta(weeks=(race_num-1) * 2.5)
            race_id = f"{year}_{race_num:02d}"
            circuit = circuits[(race_num-1) % len(circuits)]
            
            # Generate realistic results per driver
            for i, (driver_id, team_id) in enumerate(drivers):
                
                # Team competitiveness evolution (realistic)
                team_strength = {
                    'mercedes': 0.9 if year >= 2014 and year <= 2020 else 0.7,
                    'red_bull': 0.8 if year >= 2021 else 0.7,
                    'ferrari': 0.8 if year in [2017, 2018, 2022] else 0.7,
                    'mclaren': 0.6 if year >= 2021 else 0.5,
                    'williams': 0.4 if year <= 2016 else 0.3
                }.get(team_id, 0.5)
                
                # Driver skill (some drivers improve over time)
                driver_skill = {
                    'hamilton': 0.95, 'verstappen': 0.93, 'leclerc': 0.88,
                    'russell': 0.85, 'norris': 0.82, 'sainz': 0.80,
                    'bottas': 0.78, 'perez': 0.76, 'alonso': 0.90,
                    'vettel': 0.88 if year <= 2018 else 0.82
                }.get(driver_id, 0.65)
                
                # Performance = team_strength * driver_skill + randomness
                performance = team_strength * driver_skill + random.uniform(-0.1, 0.1)
                
                # Convert performance to grid position (1-20)
                # Higher performance = better (lower) grid position
                quali_rank = max(1, min(20, round(21 - performance * 20 + random.uniform(-2, 2))))
                
                # Race result influenced by qualifying but with randomness
                race_position = max(1, min(20, quali_rank + random.randint(-5, 8)))
                
                # DNF probability (lower for better teams/drivers)
                dnf_prob = max(0.02, 0.15 - team_strength * 0.1)
                status = 'dnf' if random.random() < dnf_prob else 'finished'
                if status == 'dnf':
                    race_position = random.randint(16, 20)  # DNF typically means poor finishing position
                
                # Practice session time (correlated with performance)
                base_lap_time = 75 + random.uniform(-1, 2)
                fp3_time = base_lap_time + (1 - performance) * 3 + random.uniform(-0.5, 0.5)
                
                # Labels
                is_pole = 1 if quali_rank == 1 else 0
                is_race_winner = 1 if race_position == 1 and status == 'finished' else 0
                
                all_data.append({
                    'race_id': race_id,
                    'driver_id': driver_id,
                    'team_id': team_id,
                    'date_utc': race_date,
                    'circuit_id': circuit,
                    'season': year,
                    'round': race_num,
                    'quali_rank': quali_rank,
                    'race_position': race_position,
                    'status': status,
                    'session_type': 'race',
                    'LapTimeSeconds': fp3_time,
                    'is_pole': is_pole,
                    'is_race_winner': is_race_winner,
                    'quali_best_time': base_lap_time + (1 - performance) * 2,
                    
                    # Additional realistic features
                    'is_wet': 1 if random.random() < 0.15 else 0,  # 15% wet races
                    'tyre_compound': random.choice(['soft', 'medium', 'hard']),
                    'grid_penalty': random.randint(0, 5) if random.random() < 0.05 else 0
                })
    
    # Create comprehensive dataset
    df_complete = pd.DataFrame(all_data)
    df_complete['date_utc'] = pd.to_datetime(df_complete['date_utc'])
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total records: {len(df_complete):,}")
    print(f"   Years: {df_complete['season'].min()}-{df_complete['season'].max()}")
    print(f"   Races: {df_complete['race_id'].nunique():,}")
    print(f"   Drivers: {df_complete['driver_id'].nunique():,}")
    print(f"   Teams: {df_complete['team_id'].nunique():,}")
    
    # Create labels dataset
    df_labels = df_complete[['race_id', 'driver_id', 'is_pole', 'is_race_winner', 
                            'quali_best_time', 'race_position']].copy()
    
    # Save datasets
    master_file = processed_dir / 'master_dataset_comprehensive.parquet'
    labels_file = processed_dir / 'labels_comprehensive.parquet'
    
    df_complete.to_parquet(master_file, index=False)
    df_labels.to_parquet(labels_file, index=False)
    
    print(f"\n💾 Saved datasets:")
    print(f"   Master: {master_file}")
    print(f"   Labels: {labels_file}")
    
    # Create season summary
    season_summary = df_complete.groupby('season').agg({
        'race_id': 'nunique',
        'driver_id': 'nunique', 
        'team_id': 'nunique',
        'is_pole': 'sum',
        'is_race_winner': 'sum'
    }).rename(columns={
        'race_id': 'races',
        'driver_id': 'drivers', 
        'team_id': 'teams',
        'is_pole': 'pole_positions',
        'is_race_winner': 'race_wins'
    })
    
    print(f"\n📈 Season Summary:")
    print(season_summary)
    
    return df_complete, df_labels

if __name__ == "__main__":
    df_master, df_labels = create_comprehensive_f1_data()
    print("\n✅ Comprehensive multi-year F1 data created successfully!")