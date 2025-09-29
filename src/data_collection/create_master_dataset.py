import pandas as pd
from pathlib import Path
import numpy as np

def load_ergast_data():
    """Load Ergast data (races, qualifying, results)"""
    data_dir = Path('data/raw')
    
    datasets = {}
    
    # Load races
    races_file = data_dir / 'races.parquet'
    if races_file.exists():
        datasets['races'] = pd.read_parquet(races_file)
        print(f"Loaded {len(datasets['races'])} race records")
    else:
        print("Warning: races.parquet not found")
        datasets['races'] = pd.DataFrame()
    
    # Load qualifying
    qualifying_file = data_dir / 'qualifying.parquet'
    if qualifying_file.exists():
        datasets['qualifying'] = pd.read_parquet(qualifying_file)
        print(f"Loaded {len(datasets['qualifying'])} qualifying records")
    else:
        print("Warning: qualifying.parquet not found")
        datasets['qualifying'] = pd.DataFrame()
    
    # Load results
    results_file = data_dir / 'results.parquet'
    if results_file.exists():
        datasets['results'] = pd.read_parquet(results_file)
        print(f"Loaded {len(datasets['results'])} result records")
    else:
        print("Warning: results.parquet not found")
        datasets['results'] = pd.DataFrame()
    
    return datasets

def load_fastf1_data():
    """Load FastF1 practice and qualifying data"""
    data_dir = Path('data/raw')
    fastf1_files = list(data_dir.glob('fastf1_*.parquet'))
    
    if not fastf1_files:
        print("Warning: No FastF1 files found")
        return pd.DataFrame()
    
    fastf1_data = []
    for file in fastf1_files:
        df = pd.read_parquet(file)
        fastf1_data.append(df)
        print(f"Loaded {len(df)} records from {file.name}")
    
    if fastf1_data:
        return pd.concat(fastf1_data, ignore_index=True)
    else:
        return pd.DataFrame()

def load_weather_data():
    """Load weather data"""
    weather_dir = Path('data/raw/weather')
    weather_files = list(weather_dir.glob('*.parquet'))
    
    if not weather_files:
        print("Warning: No weather files found")
        return pd.DataFrame()
    
    weather_data = []
    for file in weather_files:
        df = pd.read_parquet(file)
        weather_data.append(df)
        print(f"Loaded {len(df)} weather records from {file.name}")
    
    if weather_data:
        return pd.concat(weather_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_canonical_keys(df, data_type):
    """Create canonical keys for different data types"""
    if data_type == 'races':
        df['race_id'] = df['season'].astype(str) + '_' + df['round'].astype(str)
        df['circuit_id'] = df.get('Circuit.circuitId', '')
        df['date_utc'] = pd.to_datetime(df['date'])
        
    elif data_type == 'qualifying':
        df['race_id'] = df['season'].astype(str) + '_' + df['round'].astype(str) 
        df['driver_id'] = df.get('Driver.driverId', '')
        df['team_id'] = df.get('Constructor.constructorId', '')
        df['circuit_id'] = df.get('circuitId', '')
        df['date_utc'] = pd.to_datetime(df['date'])
        
    elif data_type == 'results':
        df['race_id'] = df['season'].astype(str) + '_' + df['round'].astype(str)
        df['driver_id'] = df.get('Driver.driverId', '')
        df['team_id'] = df.get('Constructor.constructorId', '')
        df['circuit_id'] = df.get('circuitId', '')
        df['date_utc'] = pd.to_datetime(df['date'])
        
    elif data_type == 'fastf1':
        # Extract race info from session identifier
        if 'session' in df.columns:
            session_parts = df['session'].str.split('_', expand=True)
            if len(session_parts.columns) >= 3:
                df['season'] = session_parts[0].astype(int)
                df['event_name'] = session_parts[1]
                df['session_type'] = session_parts[2]
                df['race_id'] = df['season'].astype(str) + '_' + df.get('round', '').astype(str)
        
        df['driver_id'] = df.get('Driver', '').str.upper()
        df['team_id'] = df.get('Team', '')
        
    elif data_type == 'weather':
        df['race_id'] = df['season'].astype(str) + '_' + df['round'].astype(str)
        df['circuit_id'] = df.get('circuitId', '')
        df['date_utc'] = pd.to_datetime(df['date'])
    
    return df

def merge_datasets(ergast_data, fastf1_data, weather_data):
    """Merge all datasets into master dataset"""
    
    # Start with races as the base
    master_df = ergast_data['races'].copy()
    
    if not master_df.empty:
        master_df = create_canonical_keys(master_df, 'races')
        print(f"Base races data: {len(master_df)} records")
        
        # Add qualifying data
        if not ergast_data['qualifying'].empty:
            qualifying_df = create_canonical_keys(ergast_data['qualifying'].copy(), 'qualifying')
            
            # Aggregate qualifying data per race_id (best lap time, grid position stats)
            qualifying_agg = qualifying_df.groupby('race_id').agg({
                'Q1': lambda x: x.dropna().min() if x.dropna().any() else None,
                'Q2': lambda x: x.dropna().min() if x.dropna().any() else None, 
                'Q3': lambda x: x.dropna().min() if x.dropna().any() else None,
                'position': ['mean', 'std']
            }).reset_index()
            
            qualifying_agg.columns = ['race_id', 'q1_fastest', 'q2_fastest', 'q3_fastest', 
                                    'qualifying_pos_mean', 'qualifying_pos_std']
                                    
            master_df = master_df.merge(qualifying_agg, on='race_id', how='left')
            print(f"Added qualifying data for {len(qualifying_agg)} races")
        
        # Add results data  
        if not ergast_data['results'].empty:
            results_df = create_canonical_keys(ergast_data['results'].copy(), 'results')
            
            # Aggregate results per race
            results_agg = results_df.groupby('race_id').agg({
                'position': ['mean', 'std'],
                'points': 'sum',
                'laps': 'mean',
                'milliseconds': lambda x: x.dropna().min() if x.dropna().any() else None
            }).reset_index()
            
            results_agg.columns = ['race_id', 'finish_pos_mean', 'finish_pos_std', 
                                 'total_points', 'avg_laps', 'fastest_time_ms']
                                 
            master_df = master_df.merge(results_agg, on='race_id', how='left')
            print(f"Added results data for {len(results_agg)} races")
        
        # Add FastF1 data
        if not fastf1_data.empty:
            fastf1_df = create_canonical_keys(fastf1_data.copy(), 'fastf1')
            
            # Convert lap times to seconds for analysis
            if 'LapTime' in fastf1_df.columns:
                fastf1_df['LapTimeSeconds'] = pd.to_timedelta(fastf1_df['LapTime']).dt.total_seconds()
            
            # Aggregate FastF1 data per session
            fastf1_agg = fastf1_df.groupby(['race_id', 'session_type']).agg({
                'LapTimeSeconds': ['mean', 'min', 'std'],
                'LapNumber': 'max'
            }).reset_index()
            
            # Pivot to get session types as columns
            fastf1_pivot = fastf1_agg.pivot(index='race_id', columns='session_type', 
                                          values=['LapTimeSeconds']).reset_index()
            fastf1_pivot.columns = ['race_id'] + [f'fastf1_{col[1]}_{col[0]}' for col in fastf1_pivot.columns[1:]]
            
            master_df = master_df.merge(fastf1_pivot, on='race_id', how='left')
            print(f"Added FastF1 data for {len(fastf1_pivot)} race sessions")
        
        # Add weather data
        if not weather_data.empty:
            weather_df = create_canonical_keys(weather_data.copy(), 'weather')
            
            # Select key weather features
            weather_features = ['race_id', 'temperature', 'humidity', 'pressure', 
                              'wind_speed', 'weather_main']
            weather_df = weather_df[weather_features]
            
            master_df = master_df.merge(weather_df, on='race_id', how='left')
            print(f"Added weather data for {len(weather_df)} races")
    
    return master_df

def main():
    """Main function to create master dataset"""
    print("Creating master dataset...")
    
    # Load all data
    ergast_data = load_ergast_data()
    fastf1_data = load_fastf1_data()
    weather_data = load_weather_data()
    
    # Create master dataset
    master_df = merge_datasets(ergast_data, fastf1_data, weather_data)
    
    if not master_df.empty:
        # Save master dataset
        output_file = Path('data/raw/master_dataset.parquet')
        master_df.to_parquet(output_file, index=False)
        
        print(f"\nMaster dataset created with {len(master_df)} records")
        print(f"Saved to {output_file}")
        print(f"Columns: {list(master_df.columns)}")
        
        # Show sample
        print("\nSample data:")
        print(master_df[['season', 'race_id', 'raceName', 'circuit_id', 'date_utc']].head())
        
    else:
        print("No data to create master dataset")

if __name__ == "__main__":
    main()