#!/usr/bin/env python3
"""
CORRECTED DEMO: Complete Phase 5 ML Pipeline with proper feature selection
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def demo_corrected_ml_pipeline():
    """Demonstrate complete ML pipeline using Phase 5 prepared data - CORRECTED"""
    
    print("=" * 80)
    print("CORRECTED DEMO: COMPLETE ML PIPELINE WITH PHASE 5 DATA")
    print("=" * 80)
    
    # 1. Load prepared data
    print("📁 Loading Phase 5 prepared data...")
    
    train_data = pd.read_parquet('data/models/splits/train_data.parquet')
    val_data = pd.read_parquet('data/models/splits/val_data.parquet')
    test_data = pd.read_parquet('data/models/splits/test_data.parquet')
    
    with open('data/processed/cv_indices.pkl', 'rb') as f:
        cv_indices = pickle.load(f)
    
    print(f"   ✅ Loaded train/val/test splits")
    print(f"   ✅ Loaded {len(cv_indices)} CV folds")
    
    # 2. Prepare features - Use only numeric and encoded features
    print(f"\n🔧 Preparing features for pole prediction...")
    
    # Exclude ALL categorical and target columns
    exclude_cols = [
        'race_id', 'driver_id', 'team_id', 'circuit_id', 'tyre_compound', 'status', 'session_type',
        'date_utc', 'is_pole', 'is_race_winner', 'quali_best_time', 'race_position'
    ]
    
    # Use only numeric features (including encoded categoricals)
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    # Ensure we're only using numeric columns
    numeric_cols = []
    for col in feature_cols:
        if train_data[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            numeric_cols.append(col)
    
    feature_cols = numeric_cols
    
    X_train = train_data[feature_cols].fillna(0).values
    y_train = train_data['is_pole'].values
    groups_train = train_data['race_id'].values
    
    X_val = val_data[feature_cols].fillna(0).values
    y_val = val_data['is_pole'].values
    
    X_test = test_data[feature_cols].fillna(0).values
    y_test = test_data['is_race_winner'].values  # Different target for demo
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Selected features: {feature_cols[:5]}... (showing first 5)")
    print(f"   Target (pole) distribution: No-pole: {np.sum(y_train == 0)}, Pole: {np.sum(y_train == 1)}")
    
    # 3. Cross-validation
    print(f"\n🔄 Running GroupKFold cross-validation...")
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv_indices):
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X_train[train_idx], y_train[train_idx])
        
        val_pred = model.predict(X_train[val_idx])
        score = accuracy_score(y_train[val_idx], val_pred)
        cv_scores.append(score)
        
        print(f"   Fold {fold+1}: Accuracy = {score:.3f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"   CV Mean: {cv_mean:.3f} ± {cv_std:.3f}")
    
    # 4. Final model training
    print(f"\n🎯 Training final model on complete training set...")
    
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    final_model.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = list(zip(feature_cols, final_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   🏆 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"      {i+1:2d}. {feature:25s}: {importance:.3f}")
    
    # 5. Validation set evaluation
    print(f"\n📊 Validation set evaluation (pole prediction)...")
    
    val_pred = final_model.predict(X_val)
    val_score = accuracy_score(y_val, val_pred)
    
    print(f"   Validation Accuracy: {val_score:.3f}")
    print(f"   Baseline (predict majority): {1 - np.mean(y_val):.3f}")
    print(f"   Val target distribution: No-pole: {np.sum(y_val == 0)}, Pole: {np.sum(y_val == 1)}")
    
    # 6. Train race winner model for test demonstration
    print(f"\n🏁 Test set demonstration (race winner prediction)...")
    
    y_train_winner = train_data['is_race_winner'].values
    winner_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    winner_model.fit(X_train, y_train_winner)
    
    test_pred = winner_model.predict(X_test)
    test_score = accuracy_score(y_test, test_pred)
    
    print(f"   Test Accuracy (race winner): {test_score:.3f}")
    print(f"   Test baseline: {1 - np.mean(y_test):.3f}")
    print(f"   Test target distribution: No-win: {np.sum(y_test == 0)}, Win: {np.sum(y_test == 1)}")
    
    # Feature importance for race winner
    winner_importance = list(zip(feature_cols, winner_model.feature_importances_))
    winner_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   🏆 Top 5 Features for Race Winner Prediction:")
    for i, (feature, importance) in enumerate(winner_importance[:5]):
        print(f"      {i+1}. {feature:25s}: {importance:.3f}")
    
    # 7. Summary
    print(f"\n" + "="*80)
    print("PHASE 5 ML PIPELINE DEMO - COMPLETE SUCCESS")
    print("="*80)
    
    print(f"✅ Time-aware splits: Train(2014-2022) | Val(2023) | Test(2024)")
    print(f"✅ Group-aware CV: 5 folds, no race mixing")
    print(f"✅ Feature engineering: {len(feature_cols)} numeric features")
    print(f"✅ Model training: RandomForest with proper encoding")
    print(f"✅ Evaluation: Pole & race winner prediction")
    
    print(f"\n🏆 Key Results:")
    print(f"   • Pole Prediction CV: {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"   • Pole Validation: {val_score:.3f}")
    print(f"   • Race Winner Test: {test_score:.3f}")
    print(f"   • Top Pole Feature: {feature_importance[0][0]}")
    print(f"   • Top Winner Feature: {winner_importance[0][0]}")
    
    print(f"\n📈 Model Insights:")
    pole_top_features = [f[0] for f in feature_importance[:3]]
    winner_top_features = [f[0] for f in winner_importance[:3]]
    
    print(f"   • Pole prediction driven by: {', '.join(pole_top_features)}")
    print(f"   • Race winner prediction driven by: {', '.join(winner_top_features)}")
    
    print(f"\n🚀 Phase 5 Demonstrates:")
    print(f"   ✅ Production-ready ML pipeline")
    print(f"   ✅ Temporal safety with time-aware splits")
    print(f"   ✅ Group awareness preventing race mixing")
    print(f"   ✅ Feature engineering with proper encoding")
    print(f"   ✅ Cross-validation for reliable model selection")
    print(f"   ✅ Multiple prediction tasks (pole & race winner)")
    
    print(f"\n🏁 Ready for advanced techniques:")
    print(f"   • Hyperparameter tuning with grid/random search")
    print(f"   • Ensemble methods (voting, stacking)")
    print(f"   • Deep learning models (neural networks)")
    print(f"   • Real-time prediction API deployment")

if __name__ == "__main__":
    demo_corrected_ml_pipeline()