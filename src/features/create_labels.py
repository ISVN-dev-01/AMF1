import pandas as pd
import numpy as np
from pathlib import Path

def load_processed_data():
    """Load the processed master dataset"""
    input_file = Path('data/processed/master_dataset.parquet')
    
    if not input_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {input_file}")
    
    print(f"Loading processed dataset from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df

def create_qualifying_labels(df):
    """
    Create qualifying-based labels: is_pole, quali_best_time
    """
    print("Creating qualifying labels...")
    
    # Filter for qualifying sessions
    # Check for different possible column names for session type
    session_columns = ['session_type', 'sessionType', 'Session']
    session_col = None
    
    for col in session_columns:
        if col in df.columns:
            session_col = col
            break
    
    if session_col is None:
        print("Warning: No session type column found. Attempting to identify qualifying data from other columns...")
        # Try to identify qualifying data from other means
        # Look for Q1, Q2, Q3 columns or qualifying-specific columns
        if any(col in df.columns for col in ['Q1', 'Q2', 'Q3']):
            quali = df[df[['Q1', 'Q2', 'Q3']].notna().any(axis=1)].copy()
        else:
            print("Warning: Cannot identify qualifying data. Using all data.")
            quali = df.copy()
    else:
        # Filter for qualifying sessions (Q, Qualifying, etc.)
        qualifying_indicators = ['Q', 'qualifying', 'Qualifying', 'QUALIFYING']
        quali_mask = df[session_col].isin(qualifying_indicators)
        quali = df[quali_mask].copy()
    
    print(f"Found {len(quali)} qualifying records")
    
    if len(quali) == 0:
        print("Warning: No qualifying data found. Creating empty qualifying labels.")
        return pd.DataFrame(columns=['race_id', 'driver_id', 'quali_best_time', 'is_pole', 'quali_rank'])
    
    # Find the best lap time column
    time_columns = ['lap_time_sec', 'LapTime_seconds', 'Q3_seconds', 'Q2_seconds', 'Q1_seconds', 
                    'fastestLapTime_seconds', 'LapTimeSeconds']
    
    time_col = None
    for col in time_columns:
        if col in quali.columns and quali[col].notna().sum() > 0:
            time_col = col
            break
    
    if time_col is None:
        print("Warning: No lap time column found. Attempting to create from Q1/Q2/Q3...")
        # Try to use Q1, Q2, Q3 columns
        q_columns = ['Q3_seconds', 'Q2_seconds', 'Q1_seconds']
        available_q_cols = [col for col in q_columns if col in quali.columns]
        
        if available_q_cols:
            # Use the best available Q time (Q3 > Q2 > Q1)
            quali['best_quali_time'] = quali[available_q_cols].min(axis=1)
            time_col = 'best_quali_time'
        else:
            print("Error: No time data available for qualifying labels")
            return pd.DataFrame(columns=['race_id', 'driver_id', 'quali_best_time', 'is_pole', 'quali_rank'])
    
    # Find driver identifier column
    driver_columns = ['driver_id', 'Driver_clean', 'Driver', 'driverId']
    driver_col = None
    
    for col in driver_columns:
        if col in quali.columns:
            driver_col = col
            break
    
    if driver_col is None:
        raise ValueError("No driver identifier column found")
    
    # Group by race and driver to get best qualifying time
    print(f"Using time column: {time_col}, driver column: {driver_col}")
    
    # Remove any non-numeric values and get minimum time per driver per race
    quali_clean = quali[quali[time_col].notna() & (quali[time_col] > 0)].copy()
    
    if len(quali_clean) == 0:
        print("Warning: No valid qualifying times found")
        return pd.DataFrame(columns=['race_id', 'driver_id', 'quali_best_time', 'is_pole', 'quali_rank'])
    
    # Get best qualifying time per driver per race
    best = quali_clean.groupby(['race_id', driver_col])[time_col].min().reset_index()
    best.columns = ['race_id', 'driver_id', 'quali_best_time']
    
    # Calculate qualifying rank within each race (1 = fastest)
    best['quali_rank'] = best.groupby('race_id')['quali_best_time'].rank(method='min')
    
    # Create pole position flag (rank 1 = pole position)
    best['is_pole'] = (best['quali_rank'] == 1).astype(int)
    
    print(f"Created qualifying labels for {len(best)} driver-race combinations")
    print(f"Pole positions identified: {best['is_pole'].sum()}")
    
    return best

def create_race_labels(df):
    """
    Create race-based labels: race_position, is_race_winner
    """
    print("Creating race labels...")
    
    # Filter for race sessions
    session_columns = ['session_type', 'sessionType', 'Session']
    session_col = None
    
    for col in session_columns:
        if col in df.columns:
            session_col = col
            break
    
    if session_col is None:
        print("Warning: No session type column found. Looking for race position data...")
        # Use all data and look for position columns
        race = df.copy()
    else:
        # Filter for race sessions
        race_indicators = ['R', 'race', 'Race', 'RACE']
        race_mask = df[session_col].isin(race_indicators)
        race = df[race_mask].copy()
    
    print(f"Found {len(race)} race records")
    
    # Find position column
    position_columns = ['race_position', 'position', 'Position', 'finish_position', 'positionOrder']
    position_col = None
    
    for col in position_columns:
        if col in race.columns and race[col].notna().sum() > 0:
            position_col = col
            break
    
    if position_col is None:
        print("Warning: No race position column found. Creating empty race labels.")
        return pd.DataFrame(columns=['race_id', 'driver_id', 'race_position', 'is_race_winner'])
    
    # Find driver identifier column
    driver_columns = ['driver_id', 'Driver_clean', 'Driver', 'driverId']
    driver_col = None
    
    for col in driver_columns:
        if col in race.columns:
            driver_col = col
            break
    
    if driver_col is None:
        raise ValueError("No driver identifier column found")
    
    print(f"Using position column: {position_col}, driver column: {driver_col}")
    
    # Get race positions (remove duplicates and invalid positions)
    race_clean = race[race[position_col].notna()].copy()
    
    # Convert position to numeric
    race_clean[position_col] = pd.to_numeric(race_clean[position_col], errors='coerce')
    race_clean = race_clean[race_clean[position_col].notna()].copy()
    
    # Select unique race-driver combinations
    race_pos = race_clean[['race_id', driver_col, position_col]].drop_duplicates()
    race_pos.columns = ['race_id', 'driver_id', 'race_position']
    
    # Create race winner flag (position 1 = winner)
    race_pos['is_race_winner'] = (race_pos['race_position'] == 1).astype(int)
    
    print(f"Created race labels for {len(race_pos)} driver-race combinations")
    print(f"Race winners identified: {race_pos['is_race_winner'].sum()}")
    
    return race_pos

def merge_labels(quali_labels, race_labels):
    """
    Merge qualifying and race labels into a single labels dataset
    """
    print("Merging qualifying and race labels...")
    
    # Merge on race_id and driver_id
    labels = pd.merge(
        quali_labels[['race_id', 'driver_id', 'quali_best_time', 'is_pole', 'quali_rank']], 
        race_labels[['race_id', 'driver_id', 'race_position', 'is_race_winner']], 
        on=['race_id', 'driver_id'], 
        how='outer'
    )
    
    # Fill missing values
    labels['is_pole'] = labels['is_pole'].fillna(0).astype(int)
    labels['is_race_winner'] = labels['is_race_winner'].fillna(0).astype(int)
    
    print(f"Final labels dataset: {len(labels)} driver-race combinations")
    
    return labels

def validate_labels(labels_df):
    """
    Validate the created labels
    """
    print("Validating labels...")
    
    # Basic validation checks
    validations = []
    
    # Check for required columns
    required_columns = ['race_id', 'driver_id', 'is_pole', 'is_race_winner']
    missing_columns = [col for col in required_columns if col not in labels_df.columns]
    
    if missing_columns:
        validations.append(f"❌ Missing required columns: {missing_columns}")
    else:
        validations.append("✅ All required columns present")
    
    # Check for null race_id or driver_id
    null_race_ids = labels_df['race_id'].isnull().sum()
    null_driver_ids = labels_df['driver_id'].isnull().sum()
    
    if null_race_ids == 0 and null_driver_ids == 0:
        validations.append("✅ No null race_id or driver_id values")
    else:
        validations.append(f"❌ Found {null_race_ids} null race_ids, {null_driver_ids} null driver_ids")
    
    # Check pole positions per race (should be 1 per race)
    if 'is_pole' in labels_df.columns:
        poles_per_race = labels_df.groupby('race_id')['is_pole'].sum()
        races_with_multiple_poles = (poles_per_race > 1).sum()
        races_with_no_poles = (poles_per_race == 0).sum()
        
        if races_with_multiple_poles == 0:
            validations.append("✅ No races with multiple pole positions")
        else:
            validations.append(f"⚠️  {races_with_multiple_poles} races with multiple pole positions")
        
        if races_with_no_poles == 0:
            validations.append("✅ All races have a pole position")
        else:
            validations.append(f"⚠️  {races_with_no_poles} races missing pole position")
    
    # Check race winners per race (should be 1 per race)
    if 'is_race_winner' in labels_df.columns:
        winners_per_race = labels_df.groupby('race_id')['is_race_winner'].sum()
        races_with_multiple_winners = (winners_per_race > 1).sum()
        races_with_no_winners = (winners_per_race == 0).sum()
        
        if races_with_multiple_winners == 0:
            validations.append("✅ No races with multiple winners")
        else:
            validations.append(f"⚠️  {races_with_multiple_winners} races with multiple winners")
        
        if races_with_no_winners == 0:
            validations.append("✅ All races have a winner")
        else:
            validations.append(f"⚠️  {races_with_no_winners} races missing winner")
    
    # Print validation results
    print("\nValidation Results:")
    for validation in validations:
        print(f"  {validation}")
    
    return validations

def create_summary_stats(labels_df):
    """
    Create summary statistics for the labels
    """
    print("\nLabel Summary Statistics:")
    print("=" * 40)
    
    # Basic counts
    print(f"Total driver-race combinations: {len(labels_df):,}")
    print(f"Unique races: {labels_df['race_id'].nunique():,}")
    print(f"Unique drivers: {labels_df['driver_id'].nunique():,}")
    
    # Pole positions
    if 'is_pole' in labels_df.columns:
        total_poles = labels_df['is_pole'].sum()
        print(f"Total pole positions: {total_poles:,}")
        
        # Top pole sitters
        pole_leaders = labels_df[labels_df['is_pole'] == 1]['driver_id'].value_counts().head()
        print(f"Top pole position holders:")
        for driver, count in pole_leaders.items():
            print(f"  {driver}: {count}")
    
    # Race wins
    if 'is_race_winner' in labels_df.columns:
        total_wins = labels_df['is_race_winner'].sum()
        print(f"Total race wins: {total_wins:,}")
        
        # Top winners
        win_leaders = labels_df[labels_df['is_race_winner'] == 1]['driver_id'].value_counts().head()
        print(f"Top race winners:")
        for driver, count in win_leaders.items():
            print(f"  {driver}: {count}")
    
    # Average qualifying times (if available)
    if 'quali_best_time' in labels_df.columns:
        avg_quali_time = labels_df['quali_best_time'].mean()
        if not pd.isna(avg_quali_time):
            print(f"Average qualifying best time: {avg_quali_time:.3f} seconds")

def main():
    """
    Main function to create labels
    """
    print("=" * 60)
    print("PHASE 3 - LABEL GENERATION")
    print("=" * 60)
    print("Goal: Create target columns for ML models")
    
    try:
        # Load processed data
        df = load_processed_data()
        
        # Create qualifying labels
        quali_labels = create_qualifying_labels(df)
        
        # Create race labels
        race_labels = create_race_labels(df)
        
        # Merge labels
        labels = merge_labels(quali_labels, race_labels)
        
        # Validate labels
        validate_labels(labels)
        
        # Create summary statistics
        create_summary_stats(labels)
        
        # Save labels
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'labels.parquet'
        
        labels.to_parquet(output_file, index=False)
        
        print(f"\n✅ Labels saved to: {output_file}")
        print(f"Labels dataset: {len(labels)} records, {len(labels.columns)} columns")
        
        # Show sample
        print(f"\nSample labels:")
        sample_columns = ['race_id', 'driver_id', 'is_pole', 'is_race_winner']
        available_columns = [col for col in sample_columns if col in labels.columns]
        if available_columns:
            print(labels[available_columns].head(10))
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating labels: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)