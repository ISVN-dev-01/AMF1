#!/usr/bin/env python3
"""
Feature Engineering Utilities for Stage-2 Marina Bay Model
Helper functions for computing cutoff-aware features with recency weighting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

def compute_recency_weighted_track_affinity(
    df: pd.DataFrame, 
    driver_id: int, 
    circuit_id: str, 
    cutoff_date: pd.Timestamp, 
    decay: float = 0.5
) -> Dict[str, float]:
    """
    Compute recency-weighted track affinity features for a driver at a specific circuit
    
    Args:
        df: Historical race data
        driver_id: Driver ID to compute affinity for
        circuit_id: Circuit ID (e.g., 'marina_bay')
        cutoff_date: Only use data before this date
        decay: Exponential decay factor (higher = more decay)
    
    Returns:
        Dictionary with track affinity features
    """
    # Filter to driver's historical races at this circuit before cutoff
    driver_circuit_data = df[
        (df['driver_id'] == driver_id) & 
        (df['circuit_id'] == circuit_id) & 
        (df['race_date'] < cutoff_date) &
        (df['session_type'] == 'R')  # Race sessions only
    ].sort_values('race_date', ascending=False)  # Most recent first
    
    if len(driver_circuit_data) == 0:
        return {
            'track_races_count': 0,
            'track_wins': 0,
            'track_podiums': 0,
            'track_avg_finish': np.nan,
            'track_recency_weighted_finish': np.nan,
            'track_best_finish_last3': np.nan,
            'track_consistency_score': np.nan
        }
    
    # Basic counts
    races_count = len(driver_circuit_data)
    wins = (driver_circuit_data['race_position'] == 1).sum()
    podiums = (driver_circuit_data['race_position'] <= 3).sum()
    
    # Average finish
    valid_finishes = driver_circuit_data['race_position'].dropna()
    avg_finish = valid_finishes.mean() if len(valid_finishes) > 0 else np.nan
    
    # Recency-weighted average finish
    if len(valid_finishes) > 0:
        # Create exponential decay weights (most recent = highest weight)
        weights = np.exp(-decay * np.arange(len(valid_finishes)))
        recency_weighted_finish = np.average(valid_finishes, weights=weights)
    else:
        recency_weighted_finish = np.nan
    
    # Best finish in last 3 appearances
    best_last3 = driver_circuit_data['race_position'].head(3).min()
    
    # Consistency score (inverse of standard deviation, scaled)
    if len(valid_finishes) >= 3:
        consistency = 1.0 / (1.0 + valid_finishes.std())
    else:
        consistency = np.nan
    
    return {
        'track_races_count': int(races_count),
        'track_wins': int(wins),
        'track_podiums': int(podiums),
        'track_avg_finish': float(avg_finish) if not pd.isna(avg_finish) else np.nan,
        'track_recency_weighted_finish': float(recency_weighted_finish) if not pd.isna(recency_weighted_finish) else np.nan,
        'track_best_finish_last3': float(best_last3) if not pd.isna(best_last3) else np.nan,
        'track_consistency_score': float(consistency) if not pd.isna(consistency) else np.nan
    }

def compute_season_form_features(
    df: pd.DataFrame,
    driver_id: int,
    season: int,
    cutoff_date: pd.Timestamp,
    rolling_window: int = 3
) -> Dict[str, float]:
    """
    Compute current season form features for a driver
    
    Args:
        df: Historical race data
        driver_id: Driver ID
        season: Season year
        cutoff_date: Only use data before this date
        rolling_window: Number of recent races for rolling averages
    
    Returns:
        Dictionary with season form features
    """
    # Filter to driver's races in the specified season before cutoff
    season_data = df[
        (df['driver_id'] == driver_id) &
        (df['season'] == season) &
        (df['race_date'] < cutoff_date) &
        (df['session_type'] == 'R')
    ].sort_values('race_date', ascending=False)  # Most recent first
    
    if len(season_data) == 0:
        return {
            'season_races_count': 0,
            'season_points_total': 0,
            'season_podiums_count': 0,
            'season_wins_count': 0,
            'season_avg_finish': np.nan,
            'season_avg_finish_last_n': np.nan,
            'season_avg_quali_last_n': np.nan,
            'season_dnf_rate': 0.0,
            'season_points_per_race': 0.0
        }
    
    # Basic season stats
    races_count = len(season_data)
    points_total = season_data['points'].sum()
    podiums_count = (season_data['race_position'] <= 3).sum()
    wins_count = (season_data['race_position'] == 1).sum()
    dnf_count = season_data['dnf'].sum()
    
    # Average finish position (all season)
    valid_finishes = season_data['race_position'].dropna()
    avg_finish = valid_finishes.mean() if len(valid_finishes) > 0 else np.nan
    
    # Rolling averages for recent form
    recent_races = season_data.head(rolling_window)
    
    avg_finish_recent = recent_races['race_position'].mean() if len(recent_races) > 0 else np.nan
    avg_quali_recent = recent_races['quali_rank'].mean() if len(recent_races) > 0 and 'quali_rank' in recent_races.columns else np.nan
    
    # Rates and derived metrics
    dnf_rate = dnf_count / races_count if races_count > 0 else 0.0
    points_per_race = points_total / races_count if races_count > 0 else 0.0
    
    return {
        'season_races_count': int(races_count),
        'season_points_total': int(points_total),
        'season_podiums_count': int(podiums_count),
        'season_wins_count': int(wins_count),
        'season_avg_finish': float(avg_finish) if not pd.isna(avg_finish) else np.nan,
        'season_avg_finish_last_n': float(avg_finish_recent) if not pd.isna(avg_finish_recent) else np.nan,
        'season_avg_quali_last_n': float(avg_quali_recent) if not pd.isna(avg_quali_recent) else np.nan,
        'season_dnf_rate': float(dnf_rate),
        'season_points_per_race': float(points_per_race)
    }

def compute_championship_position_features(
    df: pd.DataFrame,
    season: int,
    cutoff_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Compute championship standings and position features for all drivers
    
    Args:
        df: Historical race data
        season: Season year
        cutoff_date: Only use data before this date
    
    Returns:
        DataFrame with championship features per driver
    """
    # Get season data before cutoff
    season_data = df[
        (df['season'] == season) &
        (df['race_date'] < cutoff_date) &
        (df['session_type'] == 'R')
    ]
    
    # Calculate championship standings
    standings = season_data.groupby('driver_id').agg({
        'points': 'sum',
        'race_position': ['count', 'mean'],
        'podium': 'sum',
        'dnf': 'sum'
    }).round(2)
    
    # Flatten column names
    standings.columns = ['total_points', 'races_completed', 'avg_finish', 'podiums', 'dnfs']
    standings = standings.reset_index()
    
    # Calculate championship position
    standings['championship_position'] = standings['total_points'].rank(method='dense', ascending=False)
    
    # Points gap to leader
    leader_points = standings['total_points'].max()
    standings['points_gap_to_leader'] = leader_points - standings['total_points']
    
    # Points gap to next position
    standings = standings.sort_values('total_points', ascending=False)
    standings['points_gap_to_next'] = standings['total_points'].diff(-1).fillna(0).abs()
    
    # Normalize championship position (0-1 scale)
    max_position = standings['championship_position'].max()
    standings['championship_position_normalized'] = 1 - (standings['championship_position'] - 1) / (max_position - 1)
    
    return standings

def build_stage2_feature_matrix(
    df: pd.DataFrame,
    race_id: str,
    race_date: pd.Timestamp,
    race_season: int,
    circuit_id: str,
    driver_list: List[int]
) -> pd.DataFrame:
    """
    Build complete feature matrix for a specific race
    
    Args:
        df: Historical race data
        race_id: Race identifier
        race_date: Race date
        race_season: Race season
        circuit_id: Circuit identifier
        driver_list: List of driver IDs in the race
    
    Returns:
        DataFrame with features for each driver
    """
    features_list = []
    
    for driver_id in driver_list:
        # Basic race info
        base_features = {
            'race_id': race_id,
            'race_date': race_date,
            'season': race_season,
            'circuit_id': circuit_id,
            'driver_id': driver_id
        }
        
        # Track affinity features (Marina Bay specific)
        if circuit_id == 'marina_bay':
            track_features = compute_recency_weighted_track_affinity(
                df, driver_id, circuit_id, race_date
            )
            # Rename features to be Marina Bay specific
            marina_features = {f"marina_{k.replace('track_', '')}": v for k, v in track_features.items()}
        else:
            # Empty Marina Bay features for other circuits
            marina_features = {
                'marina_races_count': 0,
                'marina_wins': 0,
                'marina_podiums': 0,
                'marina_avg_finish': np.nan,
                'marina_recency_weighted_finish': np.nan,
                'marina_best_finish_last3': np.nan,
                'marina_consistency_score': np.nan
            }
        
        # Season form features
        form_features = compute_season_form_features(
            df, driver_id, race_season, race_date
        )
        
        # Combine all features
        driver_features = {**base_features, **marina_features, **form_features}
        features_list.append(driver_features)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features_list)
    
    # Add championship features
    championship_df = compute_championship_position_features(df, race_season, race_date)
    feature_df = feature_df.merge(championship_df, on='driver_id', how='left')
    
    # Add circuit-specific features
    feature_df['is_marina_bay'] = int(circuit_id == 'marina_bay')
    
    # Safety car probability by circuit
    safety_car_probs = {
        'marina_bay': 0.75,
        'monaco': 0.70,
        'baku': 0.65,
        'jeddah': 0.60,
        'default': 0.30
    }
    feature_df['safety_car_prob'] = safety_car_probs.get(circuit_id, safety_car_probs['default'])
    
    return feature_df

def add_qualifying_features(
    feature_df: pd.DataFrame,
    qualifying_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Add qualifying-based features to the feature matrix
    
    Args:
        feature_df: Base feature DataFrame
        qualifying_data: Qualifying results with driver_id and quali_rank
    
    Returns:
        Enhanced feature DataFrame with qualifying features
    """
    # Merge qualifying data
    enhanced_df = feature_df.merge(
        qualifying_data[['driver_id', 'quali_rank']], 
        on='driver_id', 
        how='left'
    )
    
    # Grid position (same as quali rank typically)
    enhanced_df['grid_position'] = enhanced_df['quali_rank']
    
    # Qualifying performance features
    enhanced_df['quali_top3'] = (enhanced_df['quali_rank'] <= 3).astype(int)
    enhanced_df['quali_top10'] = (enhanced_df['quali_rank'] <= 10).astype(int)
    enhanced_df['quali_pole'] = (enhanced_df['quali_rank'] == 1).astype(int)
    
    # Grid advantage (inverse of position)
    max_grid = enhanced_df['grid_position'].max()
    enhanced_df['grid_advantage'] = (max_grid + 1 - enhanced_df['grid_position']) / max_grid
    
    return enhanced_df

def validate_feature_matrix(feature_df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate the feature matrix and return diagnostic information
    
    Args:
        feature_df: Feature DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    diagnostics = {
        'total_rows': len(feature_df),
        'total_features': len(feature_df.columns),
        'missing_values': feature_df.isnull().sum().to_dict(),
        'feature_types': feature_df.dtypes.to_dict(),
        'numeric_features': feature_df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': feature_df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Check for potential issues
    issues = []
    
    # Check for high missing value rates
    missing_rates = feature_df.isnull().mean()
    high_missing = missing_rates[missing_rates > 0.5].index.tolist()
    if high_missing:
        issues.append(f"High missing value rates (>50%) in: {high_missing}")
    
    # Check for constant features
    numeric_df = feature_df.select_dtypes(include=[np.number])
    constant_features = []
    for col in numeric_df.columns:
        if numeric_df[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        issues.append(f"Constant features detected: {constant_features}")
    
    diagnostics['issues'] = issues
    
    return diagnostics

# Example usage and testing functions
def test_feature_engineering():
    """Test the feature engineering functions with mock data"""
    # Create mock data
    np.random.seed(42)
    
    mock_data = []
    drivers = [1, 11, 16, 44]
    circuits = ['marina_bay', 'monza', 'silverstone']
    
    for year in [2022, 2023, 2024]:
        for month, circuit in enumerate(circuits, 1):
            race_date = pd.Timestamp(f"{year}-{month:02d}-15", tz='UTC')
            
            for pos, driver in enumerate(drivers):
                mock_data.append({
                    'season': year,
                    'race_id': f"{year}_{circuit}",
                    'race_date': race_date,
                    'session_type': 'R',
                    'driver_id': driver,
                    'circuit_id': circuit,
                    'race_position': pos + 1,
                    'quali_rank': pos + 1,
                    'points': max(25 - pos * 5, 0),
                    'podium': int(pos < 3),
                    'dnf': 0
                })
    
    df = pd.DataFrame(mock_data)
    cutoff = pd.Timestamp("2024-12-31", tz='UTC')
    
    # Test track affinity
    print("Testing track affinity computation...")
    affinity = compute_recency_weighted_track_affinity(df, 44, 'marina_bay', cutoff)
    print(f"Marina Bay affinity for driver 44: {affinity}")
    
    # Test season form
    print("\nTesting season form computation...")
    form = compute_season_form_features(df, 44, 2024, cutoff)
    print(f"2024 season form for driver 44: {form}")
    
    # Test feature matrix building
    print("\nTesting feature matrix building...")
    feature_matrix = build_stage2_feature_matrix(
        df, "2024_marina_bay", cutoff, 2024, 'marina_bay', [1, 11, 16, 44]
    )
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Feature columns: {list(feature_matrix.columns)}")
    
    # Validate
    diagnostics = validate_feature_matrix(feature_matrix)
    print(f"\nValidation diagnostics: {diagnostics}")

if __name__ == "__main__":
    test_feature_engineering()