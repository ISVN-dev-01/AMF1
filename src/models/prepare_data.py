#!/usr/bin/env python3
"""
PHASE 5: Data Preparation for ML Training
Time-aware splits and group-aware cross-validation for F1 prediction models
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class F1DataPreparer:
    """
    Handles time-aware train/validation/test splits and group-aware cross-validation
    for F1 machine learning models
    """
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.features_dir = self.data_dir / 'features'
        self.models_dir = self.data_dir / 'models'
        
        # Create models directory
        self.models_dir.mkdir(exist_ok=True)
        
        # Target columns for different prediction tasks
        self.target_columns = {
            'pole_prediction': 'is_pole',
            'race_winner_prediction': 'is_race_winner',
            'qualifying_time': 'quali_best_time',
            'race_position': 'race_position'
        }
        
        # Columns to exclude from features
        self.exclude_columns = [
            'race_id', 'driver_id', 'date_utc', 'season', 'round',
            # Target leakage columns
            'is_pole', 'is_race_winner', 'quali_best_time', 'race_position',
            'is_pole_x', 'is_pole_y', 'is_race_winner_x', 'is_race_winner_y',
            'race_position_x', 'race_position_y'
        ]
    
    def load_comprehensive_data(self) -> pd.DataFrame:
        """Load the comprehensive multi-year F1 dataset"""
        
        master_file = self.processed_dir / 'master_dataset_comprehensive.parquet'
        
        if not master_file.exists():
            raise FileNotFoundError(
                f"Comprehensive dataset not found: {master_file}\n"
                "Run create_comprehensive_data.py first"
            )
        
        print(f"📁 Loading comprehensive dataset from {master_file}")
        df = pd.read_parquet(master_file)
        
        print(f"📊 Dataset overview:")
        print(f"   Records: {len(df):,}")
        print(f"   Years: {df['season'].min()}-{df['season'].max()}")
        print(f"   Races: {df['race_id'].nunique():,}")
        print(f"   Drivers: {df['driver_id'].nunique():,}")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features for ML"""
        
        print("🔧 Creating additional engineered features...")
        
        df_features = df.copy()
        
        # Time-based features
        df_features['race_month'] = df_features['date_utc'].dt.month
        df_features['race_quarter'] = df_features['date_utc'].dt.quarter
        df_features['is_season_start'] = (df_features['round'] <= 3).astype(int)
        df_features['is_season_end'] = (df_features['round'] >= 18).astype(int)
        
        # Driver experience (races completed by this race)
        df_features['driver_experience'] = df_features.groupby('driver_id').cumcount()
        
        # Team consistency (within season)
        team_consistency = df_features.groupby(['season', 'team_id'])['quali_rank'].transform('std')
        df_features['team_consistency'] = 1 / (1 + team_consistency.fillna(1))
        
        # Circuit familiarity (times driver raced at this circuit)
        df_features['circuit_familiarity'] = df_features.groupby(['driver_id', 'circuit_id']).cumcount()
        
        # Recent form indicators (last 3 races performance trend)
        df_features = df_features.sort_values(['driver_id', 'date_utc'])
        df_features['recent_quali_trend'] = (
            df_features.groupby('driver_id')['quali_rank']
            .rolling(3, min_periods=1).mean()
            .shift(1)  # Prevent leakage
            .reset_index(0, drop=True)
        )
        
        # Championship position (running sum of points - simplified)
        # Points system: 1st=25, 2nd=18, 3rd=15, 4th=12, 5th=10, 6th=8, 7th=6, 8th=4, 9th=2, 10th=1
        points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        df_features['race_points'] = df_features['race_position'].map(lambda x: points_map.get(x, 0))
        df_features['season_points'] = (
            df_features.groupby(['driver_id', 'season'])['race_points']
            .cumsum()
            .shift(1)  # Prevent leakage - use points before this race
            .fillna(0)
        )
        
        # Encode categorical variables
        le_driver = LabelEncoder()
        le_team = LabelEncoder()
        le_circuit = LabelEncoder()
        le_tyre = LabelEncoder()
        
        df_features['driver_encoded'] = le_driver.fit_transform(df_features['driver_id'])
        df_features['team_encoded'] = le_team.fit_transform(df_features['team_id'])
        df_features['circuit_encoded'] = le_circuit.fit_transform(df_features['circuit_id'])
        
        # Handle tyre compound if exists
        if 'tyre_compound' in df_features.columns:
            df_features['tyre_encoded'] = le_tyre.fit_transform(df_features['tyre_compound'])
        
        # Save encoders for later use
        encoders = {
            'driver': le_driver,
            'team': le_team,
            'circuit': le_circuit,
            'tyre': le_tyre if 'tyre_compound' in df_features.columns else None
        }
        
        encoders_file = self.models_dir / 'label_encoders.pkl'
        with open(encoders_file, 'wb') as f:
            pickle.dump(encoders, f)
        
        print(f"   💾 Saved label encoders to {encoders_file}")
        
        return df_features
    
    def time_aware_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-aware train/validation/test splits
        
        Train: 2014-2022 (9 seasons)
        Validation: 2023 (1 season) 
        Test: 2024 (1 season)
        """
        
        print("\n🕐 Creating time-aware train/validation/test splits...")
        
        # Ensure data is sorted by date
        df_sorted = df.sort_values(['date_utc', 'race_id']).reset_index(drop=True)
        
        # Define time-based splits
        train_data = df_sorted[df_sorted['season'] <= 2022].copy()
        val_data = df_sorted[df_sorted['season'] == 2023].copy()
        test_data = df_sorted[df_sorted['season'] == 2024].copy()
        
        print(f"📊 Split summary:")
        print(f"   Train (2014-2022): {len(train_data):,} records ({len(train_data)/len(df_sorted)*100:.1f}%)")
        print(f"   Validation (2023): {len(val_data):,} records ({len(val_data)/len(df_sorted)*100:.1f}%)")
        print(f"   Test (2024):       {len(test_data):,} records ({len(test_data)/len(df_sorted)*100:.1f}%)")
        
        print(f"\n📅 Date ranges:")
        print(f"   Train: {train_data['date_utc'].min().date()} to {train_data['date_utc'].max().date()}")
        print(f"   Val:   {val_data['date_utc'].min().date()} to {val_data['date_utc'].max().date()}")
        print(f"   Test:  {test_data['date_utc'].min().date()} to {test_data['date_utc'].max().date()}")
        
        # Validate no temporal leakage
        assert train_data['date_utc'].max() < val_data['date_utc'].min(), "Temporal leakage: train overlaps with validation"
        assert val_data['date_utc'].max() < test_data['date_utc'].min(), "Temporal leakage: validation overlaps with test"
        
        print("   ✅ No temporal leakage detected")
        
        return train_data, val_data, test_data
    
    def prepare_ml_data(self, df: pd.DataFrame, target_task: str = 'pole_prediction') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features (X), targets (y), and groups for ML training
        
        Args:
            df: DataFrame with features and targets
            target_task: One of 'pole_prediction', 'race_winner_prediction', 'qualifying_time', 'race_position'
        
        Returns:
            X: Feature matrix
            y: Target vector  
            groups: Group identifiers (race_id) for GroupKFold
        """
        
        if target_task not in self.target_columns:
            raise ValueError(f"Unknown target task: {target_task}. Choose from {list(self.target_columns.keys())}")
        
        target_col = self.target_columns[target_task]
        
        print(f"\n🎯 Preparing ML data for task: {target_task}")
        print(f"   Target column: {target_col}")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in self.exclude_columns]
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prepare targets
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        y = df[target_col].values
        
        # Prepare groups (race_id for GroupKFold)
        groups = df['race_id'].values
        
        print(f"📈 ML data summary:")
        print(f"   Features shape: {X.shape}")
        print(f"   Targets shape: {y.shape}")
        print(f"   Feature columns: {len(feature_cols)}")
        print(f"   Unique groups (races): {len(np.unique(groups))}")
        
        if target_task in ['pole_prediction', 'race_winner_prediction']:
            print(f"   Target distribution: {np.bincount(y.astype(int))}")
            print(f"   Positive class rate: {np.mean(y):.3f}")
        else:
            print(f"   Target range: {y.min():.2f} to {y.max():.2f}")
            print(f"   Target mean: {y.mean():.2f} ± {y.std():.2f}")
        
        return X, y, groups
    
    def create_group_cv_splits(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                              n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create group-aware cross-validation splits using race_id grouping
        
        This ensures that all drivers from the same race are in the same fold,
        preventing data leakage between train/validation folds.
        """
        
        print(f"\n🔄 Creating GroupKFold cross-validation splits (n_splits={n_splits})...")
        
        gkf = GroupKFold(n_splits=n_splits)
        cv_splits = list(gkf.split(X, y, groups))
        
        print(f"📊 Cross-validation summary:")
        for i, (train_idx, val_idx) in enumerate(cv_splits):
            train_groups = len(np.unique(groups[train_idx]))
            val_groups = len(np.unique(groups[val_idx]))
            
            print(f"   Fold {i+1}: Train {len(train_idx):,} samples ({train_groups} races), "
                  f"Val {len(val_idx):,} samples ({val_groups} races)")
            
            # Verify no group overlap
            train_race_ids = set(groups[train_idx])
            val_race_ids = set(groups[val_idx])
            overlap = train_race_ids.intersection(val_race_ids)
            
            if overlap:
                raise ValueError(f"Group leakage detected in fold {i+1}: {overlap}")
        
        print("   ✅ No group leakage detected across folds")
        
        return cv_splits
    
    def save_splits_and_indices(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                               test_data: pd.DataFrame, cv_splits: List[Tuple[np.ndarray, np.ndarray]]):
        """Save train/val/test splits and CV indices for reproducibility"""
        
        print(f"\n💾 Saving splits and indices for reproducibility...")
        
        # Save data splits
        splits_dir = self.models_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        train_data.to_parquet(splits_dir / 'train_data.parquet', index=False)
        val_data.to_parquet(splits_dir / 'val_data.parquet', index=False)
        test_data.to_parquet(splits_dir / 'test_data.parquet', index=False)
        
        # Save CV indices
        cv_indices_file = self.processed_dir / 'cv_indices.pkl'
        with open(cv_indices_file, 'wb') as f:
            pickle.dump(cv_splits, f)
        
        # Save metadata
        metadata = {
            'split_strategy': 'time_aware',
            'train_years': '2014-2022',
            'val_years': '2023',
            'test_years': '2024',
            'cv_method': 'GroupKFold',
            'cv_splits': len(cv_splits),
            'group_column': 'race_id',
            'created_at': datetime.now().isoformat(),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }
        
        metadata_file = self.models_dir / 'split_metadata.json'
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Train/val/test splits: {splits_dir}/")
        print(f"   💾 CV indices: {cv_indices_file}")
        print(f"   💾 Metadata: {metadata_file}")
        
        return splits_dir, cv_indices_file, metadata_file
    
    def run_complete_preparation(self, target_task: str = 'pole_prediction', 
                               cv_splits: int = 5) -> Dict[str, Any]:
        """
        Run the complete Phase 5 data preparation pipeline
        """
        
        print("=" * 80)
        print("PHASE 5: DATA PREPARATION FOR ML TRAINING")
        print("=" * 80)
        
        # 1. Load comprehensive data
        df = self.load_comprehensive_data()
        
        # 2. Create additional features
        df_features = self.create_features(df)
        
        # 3. Time-aware splits
        train_data, val_data, test_data = self.time_aware_split(df_features)
        
        # 4. Prepare ML data for training set (for CV)
        X_train, y_train, groups_train = self.prepare_ml_data(train_data, target_task)
        
        # 5. Create group-aware CV splits
        cv_indices = self.create_group_cv_splits(X_train, y_train, groups_train, cv_splits)
        
        # 6. Save everything for reproducibility
        splits_dir, cv_file, metadata_file = self.save_splits_and_indices(
            train_data, val_data, test_data, cv_indices
        )
        
        # 7. Prepare validation and test data
        X_val, y_val, groups_val = self.prepare_ml_data(val_data, target_task)
        X_test, y_test, groups_test = self.prepare_ml_data(test_data, target_task)
        
        # Return comprehensive results
        results = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'X_train': X_train,
            'y_train': y_train,
            'groups_train': groups_train,
            'X_val': X_val,
            'y_val': y_val,
            'groups_val': groups_val,
            'X_test': X_test,
            'y_test': y_test,
            'groups_test': groups_test,
            'cv_indices': cv_indices,
            'feature_columns': [col for col in df_features.columns if col not in self.exclude_columns],
            'target_task': target_task,
            'files': {
                'splits_dir': splits_dir,
                'cv_indices': cv_file,
                'metadata': metadata_file
            }
        }
        
        print(f"\n✅ Phase 5 data preparation complete!")
        print(f"   Target task: {target_task}")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   CV folds: {len(cv_indices)}")
        
        return results

def demonstrate_group_cv_usage(results: Dict[str, Any], max_folds: int = 3):
    """Demonstrate how to use GroupKFold for model training"""
    
    print(f"\n" + "="*60)
    print("DEMONSTRATION: GROUP-AWARE CROSS-VALIDATION")
    print("="*60)
    
    X_train = results['X_train']
    y_train = results['y_train']
    groups_train = results['groups_train']
    cv_indices = results['cv_indices']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target task: {results['target_task']}")
    
    # Demonstrate CV usage (limit to max_folds for demo)
    for fold, (train_idx, val_idx) in enumerate(cv_indices[:max_folds]):
        print(f"\n🔄 Fold {fold + 1}:")
        
        # Get fold data
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        groups_fold_train = groups_train[train_idx]
        groups_fold_val = groups_train[val_idx]
        
        print(f"   Train: {len(X_fold_train):,} samples from {len(np.unique(groups_fold_train))} races")
        print(f"   Val:   {len(X_fold_val):,} samples from {len(np.unique(groups_fold_val))} races")
        
        # Example: This is where you would train your model
        print(f"   📝 Example model training code:")
        print(f"      model = RandomForestClassifier()")
        print(f"      model.fit(X_fold_train, y_fold_train)")
        print(f"      val_predictions = model.predict(X_fold_val)")
        print(f"      fold_score = accuracy_score(y_fold_val, val_predictions)")
    
    print(f"\n💡 Complete GroupKFold usage example:")
    print(f"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Load saved results
    results = load_preparation_results()
    X_train, y_train, groups_train = results['X_train'], results['y_train'], results['groups_train']
    cv_indices = results['cv_indices']
    
    # Cross-validation loop
    cv_scores = []
    for train_idx, val_idx in cv_indices:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train[train_idx], y_train[train_idx])
        val_pred = model.predict(X_train[val_idx])
        score = accuracy_score(y_train[val_idx], val_pred)
        cv_scores.append(score)
    
    print(f"CV Mean: {{np.mean(cv_scores):.3f}} ± {{np.std(cv_scores):.3f}}")
    """)

if __name__ == "__main__":
    # Run Phase 5 data preparation
    preparer = F1DataPreparer()
    
    # Prepare data for pole prediction
    results = preparer.run_complete_preparation(target_task='pole_prediction', cv_splits=5)
    
    # Demonstrate GroupKFold usage
    demonstrate_group_cv_usage(results, max_folds=3)