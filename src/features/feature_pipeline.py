import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def load_processed_data():
    """Load processed master dataset and labels"""
    master_file = Path('data/processed/master_dataset.parquet')
    labels_file = Path('data/processed/labels.parquet')
    
    if not master_file.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_file}")
    
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels not found: {labels_file}")
    
    print(f"Loading master dataset: {master_file}")
    df_master = pd.read_parquet(master_file)
    
    print(f"Loading labels: {labels_file}")
    df_labels = pd.read_parquet(labels_file)
    
    print(f"Master dataset: {len(df_master)} records")
    print(f"Labels: {len(df_labels)} records")
    
    return df_master, df_labels

def compute_driver_history(df, cutoff_date):
    """
    Compute driver-level historical aggregates up to cutoff_date
    CRITICAL: Only uses data with date_utc < cutoff_date to prevent leakage
    """
    print(f"Computing driver history features with cutoff: {cutoff_date}")
    
    # Filter to only historical data before cutoff
    historical_df = df[df['date_utc'] < cutoff_date].copy()
    
    if len(historical_df) == 0:
        print("Warning: No historical data available for cutoff date")
        return pd.DataFrame()
    
    # Find appropriate driver column
    driver_columns = ['driver_id', 'Driver_clean', 'Driver']
    driver_col = None
    for col in driver_columns:
        if col in historical_df.columns:
            driver_col = col
            break
    
    if driver_col is None:
        raise ValueError("No driver column found")
    
    # Sort by date for proper rolling calculations
    historical_df = historical_df.sort_values(['date_utc', driver_col])
    
    driver_features = []
    
    # Group by driver for historical aggregates
    for driver, driver_data in historical_df.groupby(driver_col):
        driver_data = driver_data.sort_values('date_utc').copy()
        
        # Recent qualifying performance (last 3 races)
        if 'quali_rank' in driver_data.columns:
            driver_data['driver_recent_quali_mean_3'] = (
                driver_data['quali_rank'].rolling(3, min_periods=1).mean().shift(1)
            )
            driver_data['driver_recent_quali_std_3'] = (
                driver_data['quali_rank'].rolling(3, min_periods=1).std().shift(1)
            )
        
        # DNF rate over last 10 races
        if 'status' in driver_data.columns:
            driver_data['is_dnf'] = (driver_data['status'] == 'dnf').astype(int)
            driver_data['driver_dnf_rate_10'] = (
                driver_data['is_dnf'].rolling(10, min_periods=1).mean().shift(1)
            )
        
        # Career averages (all historical data)
        if 'quali_rank' in driver_data.columns:
            driver_data['driver_career_avg_quali'] = (
                driver_data['quali_rank'].expanding().mean().shift(1)
            )
        
        if 'race_position' in driver_data.columns:
            driver_data['driver_career_avg_position'] = (
                driver_data['race_position'].expanding().mean().shift(1)
            )
        
        # Recent form (last 5 races average position)
        if 'race_position' in driver_data.columns:
            driver_data['driver_recent_position_mean_5'] = (
                driver_data['race_position'].rolling(5, min_periods=1).mean().shift(1)
            )
        
        # Add driver identifier
        driver_data['driver_id'] = driver
        driver_features.append(driver_data)
    
    if driver_features:
        result = pd.concat(driver_features, ignore_index=True)
        print(f"Generated driver history features for {result[driver_col].nunique()} drivers")
        return result
    else:
        return pd.DataFrame()

def compute_team_history(df, cutoff_date):
    """
    Compute team-level historical aggregates up to cutoff_date
    """
    print(f"Computing team history features with cutoff: {cutoff_date}")
    
    # Filter to only historical data before cutoff
    historical_df = df[df['date_utc'] < cutoff_date].copy()
    
    if len(historical_df) == 0:
        return pd.DataFrame()
    
    # Find appropriate team column
    team_columns = ['team_id', 'Team_clean', 'Team']
    team_col = None
    for col in team_columns:
        if col in historical_df.columns:
            team_col = col
            break
    
    if team_col is None:
        print("Warning: No team column found")
        return historical_df
    
    # Sort by date for proper calculations
    historical_df = historical_df.sort_values(['date_utc', team_col])
    
    team_features = []
    
    # Get current season for season-specific features
    historical_df['season'] = historical_df['date_utc'].dt.year
    
    for team, team_data in historical_df.groupby(team_col):
        team_data = team_data.sort_values('date_utc').copy()
        
        # Team season average qualifying
        if 'quali_rank' in team_data.columns:
            # Season-to-date average (within each season)
            for season in team_data['season'].unique():
                season_mask = team_data['season'] == season
                season_data = team_data[season_mask].copy()
                
                if len(season_data) > 0:
                    team_season_avg = season_data['quali_rank'].expanding().mean().shift(1)
                    team_data.loc[season_mask, 'team_season_avg_quali'] = team_season_avg
        
        # Team recent performance (last 5 races)
        if 'race_position' in team_data.columns:
            team_data['team_recent_position_mean_5'] = (
                team_data['race_position'].rolling(5, min_periods=1).mean().shift(1)
            )
        
        # Team reliability (DNF rate)
        if 'status' in team_data.columns:
            team_data['is_dnf'] = (team_data['status'] == 'dnf').astype(int)
            team_data['team_dnf_rate_10'] = (
                team_data['is_dnf'].rolling(10, min_periods=1).mean().shift(1)
            )
        
        team_data['team_id'] = team
        team_features.append(team_data)
    
    if team_features:
        result = pd.concat(team_features, ignore_index=True)
        print(f"Generated team history features for {result[team_col].nunique()} teams")
        return result
    else:
        return historical_df

def compute_track_history(df, cutoff_date):
    """
    Compute track-level historical aggregates up to cutoff_date
    """
    print(f"Computing track history features with cutoff: {cutoff_date}")
    
    # Filter to only historical data before cutoff
    historical_df = df[df['date_utc'] < cutoff_date].copy()
    
    if len(historical_df) == 0:
        return pd.DataFrame()
    
    # Track affinity features (driver performance at specific tracks)
    driver_columns = ['driver_id', 'Driver_clean', 'Driver']
    driver_col = None
    for col in driver_columns:
        if col in historical_df.columns:
            driver_col = col
            break
    
    if driver_col is None or 'circuit_id' not in historical_df.columns:
        print("Warning: Missing driver or circuit columns for track history")
        return historical_df
    
    track_features = []
    
    # Driver-track combinations
    for (driver, circuit), group in historical_df.groupby([driver_col, 'circuit_id']):
        group = group.sort_values('date_utc').copy()
        
        # Driver average qualifying at this track
        if 'quali_rank' in group.columns:
            group['driver_track_avg_quali'] = (
                group['quali_rank'].expanding().mean().shift(1)
            )
        
        # Driver average race position at this track
        if 'race_position' in group.columns:
            group['driver_track_avg_position'] = (
                group['race_position'].expanding().mean().shift(1)
            )
        
        # Driver wins at this track
        if 'is_race_winner' in group.columns:
            group['driver_track_win_rate'] = (
                group['is_race_winner'].expanding().mean().shift(1)
            )
        
        track_features.append(group)
    
    # General track statistics
    track_stats = []
    for circuit, circuit_data in historical_df.groupby('circuit_id'):
        circuit_data = circuit_data.sort_values('date_utc').copy()
        
        # Track pole-to-win conversion rate
        if 'is_pole' in circuit_data.columns and 'is_race_winner' in circuit_data.columns:
            # For each race, calculate historical pole-to-win rate at this track
            pole_winners = circuit_data[circuit_data['is_pole'] == 1]['is_race_winner']
            if len(pole_winners) > 0:
                track_pole_win_rate = pole_winners.expanding().mean().shift(1)
                circuit_data.loc[circuit_data['is_pole'] == 1, 'track_pole_win_rate'] = track_pole_win_rate
        
        track_stats.append(circuit_data)
    
    if track_features:
        track_result = pd.concat(track_features, ignore_index=True)
    else:
        track_result = historical_df
    
    if track_stats:
        track_stats_result = pd.concat(track_stats, ignore_index=True)
        # Merge track statistics
        merge_cols = ['race_id', driver_col, 'circuit_id']
        available_merge_cols = [col for col in merge_cols if col in track_result.columns and col in track_stats_result.columns]
        
        if available_merge_cols:
            track_result = track_result.merge(
                track_stats_result[available_merge_cols + ['track_pole_win_rate']], 
                on=available_merge_cols, 
                how='left'
            )
    
    print(f"Generated track history features")
    return track_result

def compute_practice_features(df_session):
    """
    Compute practice session features (FP1/FP2/FP3)
    These are within-session features so no cutoff needed
    """
    print("Computing practice features...")
    
    if df_session.empty:
        return df_session
    
    # Find session type and lap time columns
    session_col = None
    session_columns = ['session_type', 'sessionType', 'Session']
    for col in session_columns:
        if col in df_session.columns:
            session_col = col
            break
    
    lap_time_cols = ['LapTimeSeconds', 'lap_time_sec', 'LapTime_seconds']
    lap_time_col = None
    for col in lap_time_cols:
        if col in df_session.columns and df_session[col].notna().sum() > 0:
            lap_time_col = col
            break
    
    if session_col is None or lap_time_col is None:
        print("Warning: Missing session or lap time columns for practice features")
        return df_session
    
    # Practice session features
    practice_sessions = ['FP1', 'FP2', 'FP3']
    
    for session_type in practice_sessions:
        session_data = df_session[df_session[session_col] == session_type].copy()
        
        if len(session_data) == 0:
            continue
        
        # Best lap time in session per driver
        best_times = session_data.groupby(['race_id', 'driver_id'])[lap_time_col].min().reset_index()
        best_times.columns = ['race_id', 'driver_id', f'{session_type.lower()}_best']
        
        # Gap to P1 (fastest driver in session)
        for race_id in best_times['race_id'].unique():
            race_times = best_times[best_times['race_id'] == race_id].copy()
            if len(race_times) > 0:
                p1_time = race_times[f'{session_type.lower()}_best'].min()
                gap_col = f'{session_type.lower()}_gap_to_p1'
                best_times.loc[best_times['race_id'] == race_id, gap_col] = (
                    race_times[f'{session_type.lower()}_best'] - p1_time
                )
        
        # Merge back to main dataframe
        merge_cols = ['race_id', 'driver_id']
        df_session = df_session.merge(best_times, on=merge_cols, how='left')
    
    print(f"Generated practice features for {df_session['race_id'].nunique()} races")
    return df_session

def compute_weather_features(df):
    """
    Compute weather-based features
    """
    print("Computing weather features...")
    
    # Weather columns that might exist
    weather_cols = {
        'temperature': 'ambient_temp',
        'humidity': 'humidity',
        'wind_speed': 'wind_speed',
        'weather_main': 'weather_condition',
        'weather_description': 'weather_desc'
    }
    
    for original, new_name in weather_cols.items():
        if original in df.columns:
            df[new_name] = df[original]
    
    # Create wet weather indicator
    if 'weather_main' in df.columns:
        wet_conditions = ['Rain', 'Drizzle', 'Thunderstorm']
        df['is_wet'] = df['weather_main'].isin(wet_conditions).astype(int)
    elif 'weather_description' in df.columns:
        wet_keywords = ['rain', 'drizzle', 'shower', 'thunderstorm']
        df['is_wet'] = df['weather_description'].str.lower().str.contains('|'.join(wet_keywords), na=False).astype(int)
    else:
        df['is_wet'] = 0  # Default to dry conditions
    
    print("Generated weather features")
    return df

def compute_tyre_features(df):
    """
    Compute tyre-related features
    """
    print("Computing tyre features...")
    
    # Look for normalized compound column
    compound_cols = ['Compound_normalized', 'compound_normalized', 'Compound']
    compound_col = None
    for col in compound_cols:
        if col in df.columns:
            compound_col = col
            break
    
    if compound_col is not None:
        # Most common compound used (fastest lap compound proxy)
        df['fastest_compound'] = df.groupby(['race_id', 'driver_id'])[compound_col].transform('first')
        
        # Tyre age on fastest lap (if available)
        if 'LapNumber' in df.columns:
            # Simplified: assume tyres changed every 20 laps on average
            df['tyre_age_on_fastest_lap'] = df['LapNumber'] % 20
    
    print("Generated tyre features")
    return df

def compute_interaction_features(df):
    """
    Compute interaction features between different aspects
    """
    print("Computing interaction features...")
    
    # Driver-team FP3 rank percentile
    if 'fp3_best' in df.columns:
        # Rank FP3 times within each race
        df['fp3_rank'] = df.groupby('race_id')['fp3_best'].rank(method='min')
        df['total_drivers'] = df.groupby('race_id')['driver_id'].transform('count')
        df['driver_team_fp3_rank_pct'] = df['fp3_rank'] / df['total_drivers']
    
    # Driver track affinity * weather interaction
    if 'driver_track_avg_quali' in df.columns and 'is_wet' in df.columns:
        df['driver_track_affinity_wet'] = df['driver_track_avg_quali'] * df['is_wet']
    
    # Team performance * driver experience
    if 'team_season_avg_quali' in df.columns and 'driver_career_avg_quali' in df.columns:
        df['team_driver_synergy'] = df['team_season_avg_quali'] * df['driver_career_avg_quali']
    
    print("Generated interaction features")
    return df

def assemble_features_for_session(df_master, df_labels, race_id):
    """
    Assemble all features for a given race, ensuring no data leakage
    """
    print(f"Assembling features for race: {race_id}")
    
    # Get race date for cutoff
    race_data = df_master[df_master['race_id'] == race_id]
    if len(race_data) == 0:
        print(f"Warning: No data found for race {race_id}")
        return pd.DataFrame()
    
    race_date = race_data['date_utc'].iloc[0]
    cutoff_date = race_date  # Use race date as cutoff
    
    print(f"Race date: {race_date}, using cutoff: {cutoff_date}")
    
    # Compute historical features (before race date)
    df_driver_hist = compute_driver_history(df_master, cutoff_date)
    df_team_hist = compute_team_history(df_master, cutoff_date)
    df_track_hist = compute_track_history(df_master, cutoff_date)
    
    # Get practice data for this race (no cutoff needed - within session)
    race_practice = df_master[df_master['race_id'] == race_id].copy()
    df_practice = compute_practice_features(race_practice)
    
    # Compute other features
    df_weather = compute_weather_features(race_data.copy())
    df_tyres = compute_tyre_features(race_data.copy())
    
    # Start with race data as base
    features_df = race_data.copy()
    
    # Find driver column
    driver_columns = ['driver_id', 'Driver_clean', 'Driver']
    driver_col = None
    for col in driver_columns:
        if col in features_df.columns:
            driver_col = col
            break
    
    if driver_col is None:
        print("Warning: No driver column found")
        return pd.DataFrame()
    
    # Merge historical features
    merge_cols = ['race_id', driver_col]
    
    if not df_driver_hist.empty:
        historical_features = [col for col in df_driver_hist.columns 
                             if col.startswith(('driver_recent_', 'driver_career_', 'driver_dnf_'))]
        if historical_features:
            driver_hist_subset = df_driver_hist[merge_cols + historical_features].drop_duplicates()
            features_df = features_df.merge(driver_hist_subset, on=merge_cols, how='left')
    
    if not df_team_hist.empty:
        team_features = [col for col in df_team_hist.columns 
                        if col.startswith(('team_season_', 'team_recent_', 'team_dnf_'))]
        if team_features:
            team_hist_subset = df_team_hist[merge_cols + team_features].drop_duplicates()
            features_df = features_df.merge(team_hist_subset, on=merge_cols, how='left')
    
    if not df_track_hist.empty:
        track_features = [col for col in df_track_hist.columns 
                         if col.startswith(('driver_track_', 'track_pole_'))]
        if track_features:
            track_hist_subset = df_track_hist[merge_cols + track_features].drop_duplicates()
            features_df = features_df.merge(track_hist_subset, on=merge_cols, how='left')
    
    # Merge practice features
    practice_features = [col for col in df_practice.columns 
                        if any(col.startswith(fp) for fp in ['fp1_', 'fp2_', 'fp3_'])]
    if practice_features:
        practice_subset = df_practice[merge_cols + practice_features].drop_duplicates()
        features_df = features_df.merge(practice_subset, on=merge_cols, how='left')
    
    # Add weather and tyre features
    weather_features = ['is_wet', 'ambient_temp', 'wind_speed']
    available_weather = [col for col in weather_features if col in df_weather.columns]
    if available_weather:
        for col in available_weather:
            features_df[col] = df_weather[col].iloc[0]  # Same for all drivers in race
    
    tyre_features = ['fastest_compound', 'tyre_age_on_fastest_lap']
    available_tyre = [col for col in tyre_features if col in df_tyres.columns]
    for col in available_tyre:
        if col in df_tyres.columns:
            features_df[col] = df_tyres[col]
    
    # Compute interaction features
    features_df = compute_interaction_features(features_df)
    
    # Merge labels
    if not df_labels.empty:
        label_cols = ['race_id', 'driver_id', 'is_pole', 'is_race_winner', 'quali_best_time', 'race_position']
        available_labels = [col for col in label_cols if col in df_labels.columns]
        if len(available_labels) >= 3:  # At least race_id, driver_id, and one label
            features_df = features_df.merge(df_labels[available_labels], on=['race_id', 'driver_id'], how='left')
    
    print(f"Assembled {len(features_df)} records with {len(features_df.columns)} features for race {race_id}")
    return features_df

def main():
    """
    Main function to create complete feature matrix
    """
    print("=" * 60)
    print("PHASE 4 - FEATURE ENGINEERING")
    print("=" * 60)
    print("Goal: Create leakage-safe features for ML models")
    
    try:
        # Load data
        df_master, df_labels = load_processed_data()
        
        # Create output directory
        output_dir = Path('data/features')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all unique races
        unique_races = sorted(df_master['race_id'].unique())
        print(f"Processing {len(unique_races)} races...")
        
        all_features = []
        
        # Process each race separately to ensure no leakage
        for i, race_id in enumerate(unique_races):
            if i % 10 == 0:
                print(f"Processing race {i+1}/{len(unique_races)}: {race_id}")
            
            race_features = assemble_features_for_session(df_master, df_labels, race_id)
            if not race_features.empty:
                all_features.append(race_features)
        
        if all_features:
            # Combine all race features
            complete_features = pd.concat(all_features, ignore_index=True)
            
            # Ensure required columns exist
            required_cols = ['race_id', 'driver_id', 'date_utc']
            missing_cols = [col for col in required_cols if col not in complete_features.columns]
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
            
            # Save complete feature matrix
            output_file = output_dir / 'complete_features.parquet'
            complete_features.to_parquet(output_file, index=False)
            
            print(f"\n✅ Complete feature matrix saved to: {output_file}")
            print(f"Features dataset: {len(complete_features)} records, {len(complete_features.columns)} columns")
            
            # Show feature summary
            feature_cols = [col for col in complete_features.columns 
                          if col not in ['race_id', 'driver_id', 'date_utc', 'is_pole', 'is_race_winner']]
            print(f"Generated {len(feature_cols)} feature columns")
            
            # Show sample
            print(f"\nSample features:")
            sample_cols = ['race_id', 'driver_id', 'fp3_best', 'driver_recent_quali_mean_3', 'is_pole']
            available_sample_cols = [col for col in sample_cols if col in complete_features.columns]
            if available_sample_cols:
                print(complete_features[available_sample_cols].head())
            
            # Feature completeness report
            print(f"\nFeature Completeness:")
            feature_completeness = complete_features[feature_cols].notna().mean().sort_values(ascending=False)
            print("Top 10 most complete features:")
            for feature, completeness in feature_completeness.head(10).items():
                print(f"  {feature}: {completeness:.1%}")
            
            return True
            
        else:
            print("❌ No features generated")
            return False
            
    except Exception as e:
        print(f"❌ Error creating features: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)