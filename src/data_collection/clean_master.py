import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def time_to_seconds(ts):
    """
    Convert time strings to seconds
    Handles formats like: '1:23.456', '83.456', NaN, etc.
    """
    if pd.isna(ts):
        return None
    
    # Convert to string if not already
    ts_str = str(ts).strip()
    
    # Handle empty strings
    if not ts_str or ts_str.lower() in ['nan', 'nat', 'none']:
        return None
    
    try:
        # Format: "1:23.456" (minutes:seconds.milliseconds)
        if ':' in ts_str:
            parts = ts_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
        
        # Format: "83.456" (just seconds)
        else:
            return float(ts_str)
            
    except (ValueError, TypeError):
        print(f"Warning: Could not parse time '{ts_str}'")
        return None
    
    return None

def standardize_driver_names():
    """
    Create mapping dict for driver name standardization
    """
    return {
        # Common variations and abbreviations
        'VER': 'verstappen',
        'HAM': 'hamilton', 
        'LEC': 'leclerc',
        'SAI': 'sainz',
        'PER': 'perez',
        'RUS': 'russell',
        'NOR': 'norris',
        'PIA': 'piastri',
        'ALO': 'alonso',
        'STR': 'stroll',
        'TSU': 'tsunoda',
        'RIC': 'ricciardo',
        'GAS': 'gasly',
        'OCO': 'ocon',
        'ALB': 'albon',
        'SAR': 'sargeant',
        'MAG': 'magnussen',
        'HUL': 'hulkenberg',
        'BOT': 'bottas',
        'ZHO': 'zhou',
        # Full name variations
        'max verstappen': 'verstappen',
        'lewis hamilton': 'hamilton',
        'charles leclerc': 'leclerc',
        'carlos sainz': 'sainz',
        'sergio perez': 'perez',
        'george russell': 'russell',
        'lando norris': 'norris',
        'oscar piastri': 'piastri',
        'fernando alonso': 'alonso',
        'lance stroll': 'stroll',
        'yuki tsunoda': 'tsunoda',
        'daniel ricciardo': 'ricciardo',
        'pierre gasly': 'gasly',
        'esteban ocon': 'ocon',
        'alexander albon': 'albon',
        'logan sargeant': 'sargeant',
        'kevin magnussen': 'magnussen',
        'nico hulkenberg': 'hulkenberg',
        'valtteri bottas': 'bottas',
        'guanyu zhou': 'zhou'
    }

def standardize_team_names():
    """
    Create mapping dict for team name standardization
    """
    return {
        'red_bull': 'red_bull',
        'red bull': 'red_bull',
        'redbull': 'red_bull',
        'rb': 'red_bull',
        'mercedes': 'mercedes',
        'merc': 'mercedes',
        'ferrari': 'ferrari',
        'fer': 'ferrari',
        'mclaren': 'mclaren',
        'mcl': 'mclaren',
        'aston_martin': 'aston_martin',
        'aston martin': 'aston_martin',
        'am': 'aston_martin',
        'alpine': 'alpine',
        'alp': 'alpine',
        'alphatauri': 'alphatauri',
        'alpha_tauri': 'alphatauri',
        'at': 'alphatauri',
        'rb_racing': 'alphatauri',  # RB rebranding
        'williams': 'williams',
        'wil': 'williams',
        'haas': 'haas',
        'haa': 'haas',
        'alfa_romeo': 'sauber',
        'alfa romeo': 'sauber',
        'sauber': 'sauber',
        'sau': 'sauber'
    }

def normalize_tyre_compound(compound):
    """
    Normalize tyre compound strings to standard categories
    """
    if pd.isna(compound):
        return None
    
    compound_str = str(compound).lower().strip()
    
    # Mapping to standard compound names
    tyre_mapping = {
        # Dry compounds
        'soft': 'soft',
        's': 'soft',
        'c5': 'soft',
        'c4': 'soft',
        'red': 'soft',
        
        'medium': 'medium', 
        'm': 'medium',
        'c3': 'medium',
        'yellow': 'medium',
        
        'hard': 'hard',
        'h': 'hard', 
        'c2': 'hard',
        'c1': 'hard',
        'white': 'hard',
        
        # Wet compounds
        'intermediate': 'inter',
        'inter': 'inter',
        'i': 'inter',
        'green': 'inter',
        
        'wet': 'wet',
        'w': 'wet',
        'full_wet': 'wet',
        'blue': 'wet'
    }
    
    return tyre_mapping.get(compound_str, compound_str)

def mark_dnf_status(df):
    """
    Mark DNF/DNS/DSQ rows and create status column
    """
    df = df.copy()
    
    # Initialize status column
    df['status'] = 'finished'
    
    # Check various columns for DNF indicators
    dnf_indicators = ['dnf', 'did not finish', 'retired', 'accident', 'collision', 
                     'engine', 'gearbox', 'hydraulics', 'electrical']
    dns_indicators = ['dns', 'did not start', 'excluded']
    dsq_indicators = ['dsq', 'disqualified', 'excluded']
    
    # Check status-related columns
    status_columns = ['status', 'statusId', 'positionText', 'position']
    
    for col in status_columns:
        if col in df.columns:
            col_str = df[col].astype(str).str.lower()
            
            # Mark DNF
            for indicator in dnf_indicators:
                df.loc[col_str.str.contains(indicator, na=False), 'status'] = 'dnf'
            
            # Mark DNS
            for indicator in dns_indicators:
                df.loc[col_str.str.contains(indicator, na=False), 'status'] = 'dns'
                
            # Mark DSQ
            for indicator in dsq_indicators:
                df.loc[col_str.str.contains(indicator, na=False), 'status'] = 'dsq'
    
    # If position is NaN but lap count exists, likely DNF
    if 'position' in df.columns and 'laps' in df.columns:
        df.loc[(pd.isna(df['position']) | (df['position'] == '\\N')) & 
               (df['laps'].notna()) & (df['laps'] > 0), 'status'] = 'dnf'
    
    return df

def ensure_timezone_aware(df):
    """
    Ensure datetime columns are timezone-aware (UTC)
    """
    datetime_columns = ['date_utc', 'date', 'time']
    
    for col in datetime_columns:
        if col in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Make timezone aware if not already
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize('UTC')
            else:
                df[col] = df[col].dt.tz_convert('UTC')
    
    return df

def clean_and_normalize_data(df):
    """
    Main function to clean and normalize the master dataset
    """
    print(f"Starting data cleaning for {len(df)} records...")
    
    df_clean = df.copy()
    
    # 1. Convert time strings to seconds
    time_columns = ['LapTime', 'Q1', 'Q2', 'Q3', 'fastestLapTime', 'milliseconds']
    
    for col in time_columns:
        if col in df_clean.columns:
            print(f"Converting time column: {col}")
            if col == 'milliseconds':
                # Milliseconds are already numeric, convert to seconds
                df_clean[f'{col}_seconds'] = pd.to_numeric(df_clean[col], errors='coerce') / 1000
            else:
                df_clean[f'{col}_seconds'] = df_clean[col].apply(time_to_seconds)
    
    # 2. Standardize driver names
    driver_mapping = standardize_driver_names()
    driver_columns = ['Driver', 'driver_id', 'Driver.driverId']
    
    for col in driver_columns:
        if col in df_clean.columns:
            print(f"Standardizing driver names in: {col}")
            df_clean[f'{col}_clean'] = (df_clean[col]
                                      .astype(str)
                                      .str.lower()
                                      .str.strip()
                                      .map(driver_mapping)
                                      .fillna(df_clean[col].astype(str).str.lower()))
    
    # 3. Standardize team names
    team_mapping = standardize_team_names()
    team_columns = ['Team', 'team_id', 'Constructor.constructorId']
    
    for col in team_columns:
        if col in df_clean.columns:
            print(f"Standardizing team names in: {col}")
            df_clean[f'{col}_clean'] = (df_clean[col]
                                      .astype(str) 
                                      .str.lower()
                                      .str.strip()
                                      .str.replace(' ', '_')
                                      .map(team_mapping)
                                      .fillna(df_clean[col].astype(str).str.lower().str.replace(' ', '_')))
    
    # 4. Normalize tyre compounds
    tyre_columns = ['Compound', 'tyre', 'compound']
    
    for col in tyre_columns:
        if col in df_clean.columns:
            print(f"Normalizing tyre compounds in: {col}")
            df_clean[f'{col}_normalized'] = df_clean[col].apply(normalize_tyre_compound)
    
    # 5. Mark DNF/DNS/DSQ status
    print("Marking DNF/DNS/DSQ status...")
    df_clean = mark_dnf_status(df_clean)
    
    # 6. Ensure timezone-aware datetimes
    print("Converting datetime columns to UTC...")
    df_clean = ensure_timezone_aware(df_clean)
    
    # 7. Clean numeric columns
    numeric_columns = ['position', 'points', 'laps', 'grid', 'LapNumber']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            # Replace '\N' with NaN and convert to numeric
            df_clean[col] = df_clean[col].replace('\\N', np.nan)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 8. Create derived features for reliability analysis
    print("Creating reliability features...")
    
    # Race completion rate per driver
    if 'status' in df_clean.columns and 'Driver_clean' in df_clean.columns:
        completion_stats = df_clean.groupby('Driver_clean')['status'].apply(
            lambda x: (x == 'finished').sum() / len(x)
        ).reset_index()
        completion_stats.columns = ['Driver_clean', 'completion_rate']
        df_clean = df_clean.merge(completion_stats, on='Driver_clean', how='left')
    
    # Average finishing position per driver (excluding DNFs)
    if 'position' in df_clean.columns and 'Driver_clean' in df_clean.columns:
        avg_position = df_clean[df_clean['status'] == 'finished'].groupby('Driver_clean')['position'].mean().reset_index()
        avg_position.columns = ['Driver_clean', 'avg_finish_position']
        df_clean = df_clean.merge(avg_position, on='Driver_clean', how='left')
    
    print(f"Data cleaning completed. Final dataset: {len(df_clean)} records")
    return df_clean

def main():
    """
    Main function to clean master dataset
    """
    print("=" * 60)
    print("PHASE 2 - DATA CLEANING & VALIDATION")
    print("=" * 60)
    
    # Check if master dataset exists
    input_file = Path('data/raw/master_dataset.parquet')
    if not input_file.exists():
        print(f"❌ Master dataset not found: {input_file}")
        print("Please run Phase 1 data collection first.")
        return False
    
    # Load master dataset
    print(f"Loading master dataset from: {input_file}")
    df_raw = pd.read_parquet(input_file)
    print(f"Loaded {len(df_raw)} records with {len(df_raw.columns)} columns")
    
    # Clean and normalize data
    df_clean = clean_and_normalize_data(df_raw)
    
    # Create output directory
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned dataset
    output_file = output_dir / 'master_dataset.parquet'
    df_clean.to_parquet(output_file, index=False)
    
    print(f"\n✅ Cleaned dataset saved to: {output_file}")
    print(f"Final dataset: {len(df_clean)} records, {len(df_clean.columns)} columns")
    
    # Show sample of cleaned data
    print(f"\nSample of cleaned data:")
    sample_columns = ['race_id', 'Driver_clean', 'Team_clean', 'status', 'date_utc']
    available_columns = [col for col in sample_columns if col in df_clean.columns]
    if available_columns:
        print(df_clean[available_columns].head())
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)