#!/usr/bin/env python3
"""
Validate Phase 5: Test time-aware splits and group-aware cross-validation
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def validate_phase5_results():
    """Validate Phase 5 implementation results"""
    
    print("=" * 80)
    print("VALIDATING PHASE 5: TIME-AWARE SPLITS & GROUP-AWARE CV")
    print("=" * 80)
    
    # Check required directories and files
    models_dir = Path('data/models')
    splits_dir = models_dir / 'splits'
    processed_dir = Path('data/processed')
    
    print("📁 Checking required files...")
    
    required_files = {
        'train_data': splits_dir / 'train_data.parquet',
        'val_data': splits_dir / 'val_data.parquet',
        'test_data': splits_dir / 'test_data.parquet',
        'cv_indices': processed_dir / 'cv_indices.pkl',
        'metadata': models_dir / 'split_metadata.json',
        'encoders': models_dir / 'label_encoders.pkl'
    }
    
    missing_files = []
    for name, file_path in required_files.items():
        if file_path.exists():
            print(f"   ✅ {name}: {file_path}")
        else:
            print(f"   ❌ Missing {name}: {file_path}")
            missing_files.append(name)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Run 'python src/models/prepare_data.py' first")
        return False
    
    # Load and validate data splits
    print(f"\n📊 Validating data splits...")
    
    train_data = pd.read_parquet(required_files['train_data'])
    val_data = pd.read_parquet(required_files['val_data'])
    test_data = pd.read_parquet(required_files['test_data'])
    
    print(f"   Train: {len(train_data):,} records ({train_data['season'].min()}-{train_data['season'].max()})")
    print(f"   Val:   {len(val_data):,} records ({val_data['season'].min()}-{val_data['season'].max()})")
    print(f"   Test:  {len(test_data):,} records ({test_data['season'].min()}-{test_data['season'].max()})")
    
    # Validate temporal ordering
    train_max_date = train_data['date_utc'].max()
    val_min_date = val_data['date_utc'].min()
    val_max_date = val_data['date_utc'].max()
    test_min_date = test_data['date_utc'].min()
    
    print(f"\n🕐 Validating temporal ordering...")
    print(f"   Train ends:    {train_max_date.date()}")
    print(f"   Val starts:    {val_min_date.date()}")
    print(f"   Val ends:      {val_max_date.date()}")
    print(f"   Test starts:   {test_min_date.date()}")
    
    # Check for temporal leakage
    temporal_leakage = False
    if train_max_date >= val_min_date:
        print(f"   ❌ Temporal leakage: Train overlaps with validation")
        temporal_leakage = True
    
    if val_max_date >= test_min_date:
        print(f"   ❌ Temporal leakage: Validation overlaps with test")
        temporal_leakage = True
    
    if not temporal_leakage:
        print(f"   ✅ No temporal leakage detected")
    
    # Validate expected split ratios
    total_records = len(train_data) + len(val_data) + len(test_data)
    train_pct = len(train_data) / total_records * 100
    val_pct = len(val_data) / total_records * 100
    test_pct = len(test_data) / total_records * 100
    
    print(f"\n📈 Split proportions:")
    print(f"   Train: {train_pct:.1f}% (expected ~82%)")
    print(f"   Val:   {val_pct:.1f}% (expected ~9%)")
    print(f"   Test:  {test_pct:.1f}% (expected ~9%)")
    
    # Load and validate CV indices
    print(f"\n🔄 Validating cross-validation indices...")
    
    with open(required_files['cv_indices'], 'rb') as f:
        cv_indices = pickle.load(f)
    
    print(f"   CV folds: {len(cv_indices)}")
    
    # Test group-aware splitting
    test_group_awareness(train_data, cv_indices)
    
    # Load metadata
    with open(required_files['metadata'], 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📋 Metadata validation:")
    print(f"   Split strategy: {metadata['split_strategy']}")
    print(f"   CV method: {metadata['cv_method']}")
    print(f"   Group column: {metadata['group_column']}")
    print(f"   Created: {metadata['created_at'][:19]}")
    
    # Test actual model training
    test_model_with_cv(train_data, cv_indices)
    
    print(f"\n✅ Phase 5 validation complete!")
    return True

def test_group_awareness(train_data, cv_indices):
    """Test that GroupKFold properly separates races"""
    
    print(f"\n🏁 Testing group-aware splitting...")
    
    # Prepare simple features for testing
    feature_cols = ['quali_rank', 'race_position', 'season', 'round']
    available_cols = [col for col in feature_cols if col in train_data.columns]
    
    if len(available_cols) < 2:
        print(f"   ⚠️  Insufficient feature columns for testing")
        return
    
    X = train_data[available_cols].fillna(0).values
    y = train_data['is_pole'].values
    groups = train_data['race_id'].values
    
    print(f"   Test data: {X.shape[0]:,} samples, {len(np.unique(groups))} unique races")
    
    # Test each fold
    group_leakage_detected = False
    
    for fold, (train_idx, val_idx) in enumerate(cv_indices):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        
        if overlap:
            print(f"   ❌ Fold {fold+1}: Group leakage detected - {len(overlap)} overlapping races")
            group_leakage_detected = True
        else:
            print(f"   ✅ Fold {fold+1}: No group leakage ({len(train_groups)} train races, {len(val_groups)} val races)")
    
    if not group_leakage_detected:
        print(f"   ✅ All folds pass group-awareness test")

def test_model_with_cv(train_data, cv_indices):
    """Test actual model training with cross-validation"""
    
    print(f"\n🤖 Testing model training with GroupKFold...")
    
    # Prepare features
    feature_cols = ['quali_rank', 'race_position', 'season', 'round', 'driver_experience']
    available_cols = [col for col in feature_cols if col in train_data.columns]
    
    if len(available_cols) < 3:
        print(f"   ⚠️  Insufficient features for model testing")
        return
    
    X = train_data[available_cols].fillna(0).values
    y = train_data['is_pole'].values
    
    print(f"   Features: {available_cols}")
    print(f"   Training samples: {X.shape[0]:,}")
    print(f"   Positive class rate: {np.mean(y):.3f}")
    
    # Cross-validation
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_indices[:3]):  # Test first 3 folds
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
        model.fit(X_fold_train, y_fold_train)
        
        # Predict and score
        y_pred = model.predict(X_fold_val)
        score = accuracy_score(y_fold_val, y_pred)
        cv_scores.append(score)
        
        print(f"   Fold {fold+1}: Accuracy = {score:.3f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"   CV Mean: {mean_score:.3f} ± {std_score:.3f}")
    print(f"   ✅ Model training successful")

def demonstrate_production_usage():
    """Demonstrate how to use the prepared data in production"""
    
    print(f"\n" + "="*60)
    print("PRODUCTION USAGE DEMONSTRATION")
    print("="*60)
    
    # Load prepared data
    models_dir = Path('data/models')
    splits_dir = models_dir / 'splits'
    
    print(f"🔧 Loading prepared datasets...")
    
    train_data = pd.read_parquet(splits_dir / 'train_data.parquet')
    val_data = pd.read_parquet(splits_dir / 'val_data.parquet')
    test_data = pd.read_parquet(splits_dir / 'test_data.parquet')
    
    with open('data/processed/cv_indices.pkl', 'rb') as f:
        cv_indices = pickle.load(f)
    
    print(f"   ✅ Loaded train/val/test splits")
    print(f"   ✅ Loaded {len(cv_indices)} CV folds")
    
    # Show usage pattern
    print(f"\n📝 Production usage pattern:")
    print(f"""
    # 1. Load prepared data
    train_data = pd.read_parquet('data/models/splits/train_data.parquet')
    val_data = pd.read_parquet('data/models/splits/val_data.parquet')
    test_data = pd.read_parquet('data/models/splits/test_data.parquet')
    
    with open('data/processed/cv_indices.pkl', 'rb') as f:
        cv_indices = pickle.load(f)
    
    # 2. Prepare features (exclude target and metadata columns)
    exclude_cols = ['race_id', 'driver_id', 'date_utc', 'is_pole', 'is_race_winner']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    X_train = train_data[feature_cols].fillna(0).values
    y_train = train_data['is_pole'].values  # or 'is_race_winner'
    groups_train = train_data['race_id'].values
    
    # 3. Cross-validation with GroupKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    cv_scores = []
    for train_idx, val_idx in cv_indices:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train[train_idx], y_train[train_idx])
        val_pred = model.predict(X_train[val_idx])
        score = accuracy_score(y_train[val_idx], val_pred)
        cv_scores.append(score)
    
    print(f"CV Score: {{np.mean(cv_scores):.3f}} ± {{np.std(cv_scores):.3f}}")
    
    # 4. Final model training on full training set
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_train, y_train)
    
    # 5. Validation set evaluation
    X_val = val_data[feature_cols].fillna(0).values
    y_val = val_data['is_pole'].values
    val_predictions = final_model.predict(X_val)
    val_score = accuracy_score(y_val, val_predictions)
    
    # 6. Final test set evaluation (only once!)
    X_test = test_data[feature_cols].fillna(0).values
    y_test = test_data['is_pole'].values
    test_predictions = final_model.predict(X_test)
    test_score = accuracy_score(y_test, test_predictions)
    """)
    
    print(f"✅ Production usage demonstration complete")

if __name__ == "__main__":
    # Validate Phase 5 implementation
    success = validate_phase5_results()
    
    if success:
        # Demonstrate production usage
        demonstrate_production_usage()
    else:
        print("❌ Validation failed - fix issues before proceeding")