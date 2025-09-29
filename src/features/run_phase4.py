#!/usr/bin/env python3
"""
PHASE 4 Orchestrator: Feature Engineering
Cutoff-aware, leakage-safe feature computation
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'features'))

from feature_pipeline import (
    compute_driver_history,
    compute_team_history, 
    compute_track_history,
    compute_practice_features,
    assemble_features_for_session
)

def run_phase4():
    """Run Phase 4: Feature Engineering"""
    print("=" * 60)
    print("PHASE 4: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Check inputs
    data_dir = Path('data')
    processed_dir = data_dir / 'processed'
    features_dir = data_dir / 'features'
    
    master_file = processed_dir / 'master_dataset.parquet'
    labels_file = processed_dir / 'labels.parquet'
    
    if not master_file.exists():
        print(f"❌ Master dataset not found: {master_file}")
        print("Run Phase 2 first: python src/data_collection/run_phase2.py")
        return False
    
    if not labels_file.exists():
        print(f"❌ Labels not found: {labels_file}")  
        print("Run Phase 3 first: python src/features/run_phase3.py")
        return False
    
    # Create features directory
    features_dir.mkdir(exist_ok=True)
    
    print(f"📁 Loading master dataset from {master_file}")
    df_master = pd.read_parquet(master_file)
    
    print(f"📁 Loading labels from {labels_file}")
    df_labels = pd.read_parquet(labels_file)
    
    print(f"📊 Dataset overview:")
    print(f"   Master records: {len(df_master):,}")
    print(f"   Label records: {len(df_labels):,}")
    print(f"   Date range: {df_master['date_utc'].min()} to {df_master['date_utc'].max()}")
    print(f"   Unique races: {df_master['race_id'].nunique():,}")
    print(f"   Unique drivers: {df_master['driver_id'].nunique():,}")
    
    # Get all unique race_ids for feature computation
    race_ids = sorted(df_master['race_id'].unique())
    
    print(f"\n🔧 Computing features for {len(race_ids)} races...")
    
    all_features = []
    processed_races = 0
    
    for i, race_id in enumerate(race_ids):
        try:
            if i % 10 == 0:
                print(f"   Processing race {i+1}/{len(race_ids)}: {race_id}")
            
            # Assemble features for this race (cutoff-aware)
            race_features = assemble_features_for_session(df_master, df_labels, race_id)
            
            if not race_features.empty:
                all_features.append(race_features)
                processed_races += 1
                
        except Exception as e:
            print(f"   ⚠️  Error processing {race_id}: {e}")
            continue
    
    if not all_features:
        print("❌ No features generated")
        return False
    
    # Combine all features
    print(f"\n📈 Combining features from {processed_races} races...")
    df_features = pd.concat(all_features, ignore_index=True)
    
    # Output summary
    print(f"\n📊 Feature generation summary:")
    print(f"   Total feature records: {len(df_features):,}")
    print(f"   Features per race: {len(df_features) / processed_races:.1f}")
    print(f"   Feature columns: {len(df_features.columns)}")
    
    # Feature completeness
    feature_cols = [col for col in df_features.columns 
                   if col not in ['race_id', 'driver_id', 'date_utc', 'is_pole', 'is_race_winner']]
    
    print(f"\n🎯 Feature completeness (top 10):")
    completeness = df_features[feature_cols].notna().mean().sort_values(ascending=False)
    for i, (feature, pct) in enumerate(completeness.head(10).items()):
        print(f"   {i+1:2d}. {feature:25s}: {pct:6.1%}")
    
    # Sample feature values
    print(f"\n📋 Sample feature values:")
    sample_features = ['driver_recent_quali_mean_3', 'team_season_avg_quali', 'driver_track_avg_quali']
    for feature in sample_features:
        if feature in df_features.columns:
            values = df_features[feature].dropna()
            if len(values) > 0:
                print(f"   {feature:25s}: {values.min():.2f} to {values.max():.2f} (mean: {values.mean():.2f})")
    
    # Save features  
    output_file = features_dir / 'complete_features.parquet'
    print(f"\n💾 Saving features to {output_file}")
    df_features.to_parquet(output_file, index=False)
    
    print(f"✅ Phase 4 complete!")
    print(f"   Features saved: {output_file}")
    print(f"   Records: {len(df_features):,}")
    print(f"   Columns: {len(df_features.columns)}")
    
    return True

if __name__ == "__main__":
    success = run_phase4()
    if not success:
        sys.exit(1)