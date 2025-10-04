#!/usr/bin/env python3
"""
Test Phase 5: Validate time-aware splits and group-aware cross-validation
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from src.models.prepare_data import F1DataPreparer
except ImportError:
    # Try alternative path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from models.prepare_data import F1DataPreparer

def test_phase5_implementation():
    """Test Phase 5 implementation with different prediction tasks"""
    
    print("=" * 80)
    print("TESTING PHASE 5: DATA PREPARATION VALIDATION")
    print("=" * 80)
    
    preparer = F1DataPreparer()
    
    # Test different prediction tasks
    prediction_tasks = ['pole_prediction', 'race_winner_prediction']
    
    for task in prediction_tasks:
        print(f"\n🎯 Testing task: {task}")
        print("-" * 50)
        
        try:
            # Run preparation
            results = preparer.run_complete_preparation(
                target_task=task, 
                cv_splits=3  # Smaller for testing
            )
            
            # Validate results structure
            assert 'X_train' in results
            assert 'y_train' in results
            assert 'groups_train' in results
            assert 'cv_indices' in results
            
            print(f"   ✅ Data preparation successful")
            print(f"   ✅ All required components present")
            
            # Test actual model training with GroupKFold
            test_model_training(results, task)
            
        except Exception as e:
            print(f"   ❌ Error in {task}: {e}")
            continue
    
    # Test saved files
    test_saved_files(preparer)
    
    print(f"\n🎉 Phase 5 testing complete!")

def test_model_training(results, task):
    """Test actual model training with GroupKFold"""
    
    print(f"\n🤖 Testing model training for {task}...")
    
    X_train = results['X_train']
    y_train = results['y_train']
    groups_train = results['groups_train']
    cv_indices = results['cv_indices']
    
    cv_scores = []
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv_indices[:2]):  # Limit to 2 folds for testing
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)  # Small for testing
        model.fit(X_fold_train, y_fold_train)
        
        # Predict
        val_pred = model.predict(X_fold_val)
        score = accuracy_score(y_fold_val, val_pred)
        cv_scores.append(score)
        
        print(f"   Fold {fold+1}: Accuracy = {score:.3f}")
        
        # Verify no group leakage
        train_groups = set(groups_train[train_idx])
        val_groups = set(groups_train[val_idx])
        overlap = train_groups.intersection(val_groups)
        
        assert len(overlap) == 0, f"Group leakage detected: {overlap}"
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"   CV Mean: {mean_score:.3f} ± {std_score:.3f}")
    print(f"   ✅ Model training successful")
    print(f"   ✅ No group leakage detected")

def test_saved_files(preparer):
    """Test that all required files were saved correctly"""
    
    print(f"\n📁 Testing saved files...")
    
    # Check directories
    models_dir = preparer.models_dir
    splits_dir = models_dir / 'splits'
    processed_dir = preparer.processed_dir
    
    # Required files
    required_files = [
        splits_dir / 'train_data.parquet',
        splits_dir / 'val_data.parquet', 
        splits_dir / 'test_data.parquet',
        processed_dir / 'cv_indices.pkl',
        models_dir / 'split_metadata.json',
        models_dir / 'label_encoders.pkl'
    ]
    
    for file_path in required_files:
        if file_path.exists():
            print(f"   ✅ {file_path.name}")
        else:
            print(f"   ❌ Missing: {file_path}")
    
    # Test loading saved files
    try:
        # Load CV indices
        with open(processed_dir / 'cv_indices.pkl', 'rb') as f:
            cv_indices = pickle.load(f)
        print(f"   ✅ CV indices loadable: {len(cv_indices)} folds")
        
        # Load split data
        train_data = pd.read_parquet(splits_dir / 'train_data.parquet')
        val_data = pd.read_parquet(splits_dir / 'val_data.parquet')
        test_data = pd.read_parquet(splits_dir / 'test_data.parquet')
        
        print(f"   ✅ Split data loadable: {len(train_data)}/{len(val_data)}/{len(test_data)} records")
        
        # Validate temporal ordering
        assert train_data['date_utc'].max() < val_data['date_utc'].min()
        assert val_data['date_utc'].max() < test_data['date_utc'].min()
        print(f"   ✅ Temporal ordering preserved")
        
    except Exception as e:
        print(f"   ❌ Error loading saved files: {e}")

def demonstrate_usage():
    """Demonstrate how to use the prepared data"""
    
    print(f"\n" + "="*60)
    print("USAGE DEMONSTRATION")
    print("="*60)
    
    try:
        # Load prepared data
        models_dir = Path('data/models')
        splits_dir = models_dir / 'splits'
        
        train_data = pd.read_parquet(splits_dir / 'train_data.parquet')
        
        # Load CV indices
        with open('data/processed/cv_indices.pkl', 'rb') as f:
            cv_indices = pickle.load(f)
        
        print(f"📊 Loaded data summary:")
        print(f"   Training data: {len(train_data):,} records")
        print(f"   Date range: {train_data['date_utc'].min().date()} to {train_data['date_utc'].max().date()}")
        print(f"   CV folds: {len(cv_indices)}")
        
        # Prepare features for demonstration
        preparer = F1DataPreparer()
        X_train, y_train, groups_train = preparer.prepare_ml_data(train_data, 'pole_prediction')
        
        print(f"\n🔄 Cross-validation demonstration:")
        for i, (train_idx, val_idx) in enumerate(cv_indices[:2]):  # Show first 2 folds
            train_races = len(np.unique(groups_train[train_idx]))
            val_races = len(np.unique(groups_train[val_idx]))
            
            print(f"   Fold {i+1}: {len(train_idx):,} train samples ({train_races} races), "
                  f"{len(val_idx):,} val samples ({val_races} races)")
        
        print(f"\n✅ Usage demonstration complete")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")

if __name__ == "__main__":
    # Run comprehensive testing
    test_phase5_implementation()
    
    # Demonstrate usage
    demonstrate_usage()