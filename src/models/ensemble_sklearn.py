#!/usr/bin/env python3
"""
PHASE 7.2A: Pure Scikit-learn Ensemble Classifier for Race Winner Prediction
Fast ensemble approach: RandomForest + GradientBoosting + ExtraTrees → Logistic Regression meta-learner
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_stage2_features():
    """Load Stage-2 features prepared in Phase 7.1"""
    
    features_file = Path('data/features/stage2_features.parquet')
    if not features_file.exists():
        raise FileNotFoundError(f"Stage-2 features not found: {features_file}")
    
    print(f"📁 Loading Stage-2 features from {features_file}")
    stage2_data = pd.read_parquet(features_file)
    
    print(f"   Shape: {stage2_data.shape}")
    print(f"   Races: {stage2_data['race_id'].nunique()}")
    print(f"   Winners: {stage2_data['is_winner'].sum()}")
    
    return stage2_data

def prepare_ensemble_data(data):
    """Prepare data for ensemble training"""
    
    print(f"🔧 Preparing ensemble training data...")
    
    # Define feature columns (exclude identifiers and target)
    exclude_cols = ['race_id', 'driver_id', 'team_id', 'date_utc', 'circuit_id', 
                   'is_winner', 'data_split', 'status']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Handle any remaining non-numeric columns
    numeric_cols = []
    for col in feature_cols:
        if data[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
        else:
            print(f"   Skipping non-numeric column: {col}")
    
    feature_cols = numeric_cols
    
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Sample features: {feature_cols[:10]}")
    
    # Prepare features and target
    X = data[feature_cols].copy()
    y = data['is_winner'].copy()
    groups = data['race_id'].values
    
    # Handle missing values more robustly
    print(f"   Handling missing values...")
    nan_count = X.isnull().sum().sum()
    print(f"   Missing values per column: {nan_count}")
    
    if nan_count > 0:
        # More aggressive NaN handling
        print(f"   Columns with NaN: {X.columns[X.isnull().any()].tolist()}")
        
        # Fill with 0 first, then with column statistics
        X = X.fillna(0)
        
        # For any remaining NaN (e.g., all-NaN columns), fill with column mean/median
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
        
        # Final cleanup - replace any remaining NaN with 0
        X = X.fillna(0)
        
        # Verify no NaN values remain
        final_nan_count = X.isnull().sum().sum()
        print(f"   Missing values after aggressive imputation: {final_nan_count}")
        
        if final_nan_count > 0:
            print(f"   WARNING: Still have {final_nan_count} NaN values, replacing with 0")
            X = X.fillna(0)
    
    print(f"   All missing values handled successfully")
    
    # Split by data_split if available
    if 'data_split' in data.columns:
        train_mask = data['data_split'] == 'train'
        val_mask = data['data_split'] == 'val'  
        test_mask = data['data_split'] == 'test'
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        groups_train = groups[train_mask]
        
        X_val = X[val_mask] if val_mask.any() else X[test_mask]
        y_val = y[val_mask] if val_mask.any() else y[test_mask]
        groups_val = groups[val_mask] if val_mask.any() else groups[test_mask]
        
        X_test = X[test_mask]
        y_test = y[test_mask]
        groups_test = groups[test_mask]
        
    else:
        # Simple train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        groups_train = groups[:split_idx]
        groups_test = groups[split_idx:]
        X_val, y_val, groups_val = X_test, y_test, groups_test
    
    print(f"   Train: {X_train.shape[0]} samples, {y_train.sum()} winners")
    print(f"   Val:   {X_val.shape[0]} samples, {y_val.sum()} winners")
    print(f"   Test:  {X_test.shape[0]} samples, {y_test.sum()} winners")
    
    return {
        'X_train': X_train, 'y_train': y_train, 'groups_train': groups_train,
        'X_val': X_val, 'y_val': y_val, 'groups_val': groups_val,
        'X_test': X_test, 'y_test': y_test, 'groups_test': groups_test,
        'feature_cols': feature_cols
    }

def train_base_models(X_train, y_train):
    """Train base models for ensemble using pure scikit-learn"""
    
    print(f"🎯 Training base models for ensemble...")
    
    base_models = {}
    
    # 1. Random Forest
    print(f"   Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    base_models['random_forest'] = rf_model
    
    # 2. Gradient Boosting
    print(f"   Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    base_models['gradient_boosting'] = gb_model
    
    # 3. Extra Trees
    print(f"   Training Extra Trees...")
    et_model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    base_models['extra_trees'] = et_model
    
    print(f"   Base models trained: {list(base_models.keys())}")
    
    return base_models

def generate_base_predictions(base_models, X):
    """Generate predictions from base models"""
    
    base_predictions = {}
    
    for model_name, model in base_models.items():
        pred_proba = model.predict_proba(X)
        # Use probability of positive class (winner)
        base_predictions[model_name] = pred_proba[:, 1]
    
    return base_predictions

def train_meta_learner(base_models, X_train, y_train, X_val, y_val):
    """Train meta-learner on base model predictions"""
    
    print(f"🧠 Training meta-learner...")
    
    # Generate base model predictions on training data
    train_base_preds = generate_base_predictions(base_models, X_train)
    
    # Create meta-features matrix
    meta_features_train = np.column_stack(list(train_base_preds.values()))
    
    print(f"   Meta-features shape: {meta_features_train.shape}")
    
    # Train logistic regression meta-learner
    meta_learner = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    
    # Scale meta-features
    scaler = StandardScaler()
    meta_features_train_scaled = scaler.fit_transform(meta_features_train)
    
    meta_learner.fit(meta_features_train_scaled, y_train)
    
    # Evaluate on validation set
    val_base_preds = generate_base_predictions(base_models, X_val)
    meta_features_val = np.column_stack(list(val_base_preds.values()))
    meta_features_val_scaled = scaler.transform(meta_features_val)
    
    val_predictions = meta_learner.predict_proba(meta_features_val_scaled)[:, 1]
    val_accuracy = accuracy_score(y_val, val_predictions > 0.5)
    val_logloss = log_loss(y_val, val_predictions)
    
    print(f"   Validation accuracy: {val_accuracy:.3f}")
    print(f"   Validation log loss: {val_logloss:.3f}")
    
    return meta_learner, scaler, list(train_base_preds.keys())

def evaluate_ensemble(base_models, meta_learner, scaler, model_names, X_test, y_test, groups_test):
    """Evaluate ensemble model"""
    
    print(f"📊 Evaluating ensemble model...")
    
    # Generate base predictions
    test_base_preds = generate_base_predictions(base_models, X_test)
    
    # Create meta-features
    meta_features_test = np.column_stack([test_base_preds[name] for name in model_names])
    meta_features_test_scaled = scaler.transform(meta_features_test)
    
    # Get ensemble predictions
    ensemble_predictions = meta_learner.predict_proba(meta_features_test_scaled)[:, 1]
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, ensemble_predictions > 0.5)
    test_logloss = log_loss(y_test, ensemble_predictions)
    test_brier = brier_score_loss(y_test, ensemble_predictions)
    
    try:
        test_auc = roc_auc_score(y_test, ensemble_predictions)
    except:
        test_auc = 0.5  # If only one class present
    
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"   Test log loss: {test_logloss:.3f}")
    print(f"   Test Brier score: {test_brier:.3f}")
    print(f"   Test AUC: {test_auc:.3f}")
    
    # Top-K accuracy by race
    print(f"\n   Calculating race-level metrics...")
    race_metrics = calculate_race_metrics(X_test, y_test, groups_test, ensemble_predictions)
    
    return {
        'predictions': ensemble_predictions,
        'accuracy': test_accuracy,
        'log_loss': test_logloss,
        'brier_score': test_brier,
        'auc': test_auc,
        'race_metrics': race_metrics
    }

def calculate_race_metrics(X_test, y_test, groups_test, predictions):
    """Calculate race-level Top-K accuracy metrics"""
    
    race_results = pd.DataFrame({
        'race_id': groups_test,
        'actual_winner': y_test,
        'win_probability': predictions
    })
    
    top1_correct = 0
    top3_correct = 0
    total_races = 0
    
    for race_id, race_group in race_results.groupby('race_id'):
        race_sorted = race_group.sort_values('win_probability', ascending=False)
        
        actual_winner = race_group['actual_winner'].sum() > 0
        if actual_winner:
            # Check if actual winner is in top predictions
            top1_pred = race_sorted.iloc[0]['actual_winner'] == 1
            top3_pred = race_sorted.head(3)['actual_winner'].sum() > 0
            
            if top1_pred:
                top1_correct += 1
            if top3_pred:
                top3_correct += 1
            
            total_races += 1
    
    top1_accuracy = top1_correct / total_races if total_races > 0 else 0
    top3_accuracy = top3_correct / total_races if total_races > 0 else 0
    
    print(f"   Race-level Top-1 accuracy: {top1_accuracy:.3f}")
    print(f"   Race-level Top-3 accuracy: {top3_accuracy:.3f}")
    
    return {
        'top1_accuracy': top1_accuracy,
        'top3_accuracy': top3_accuracy,
        'total_races': total_races
    }

def run_ensemble_classifier():
    """Run complete ensemble classifier training and evaluation"""
    
    print("=" * 80)
    print("PHASE 7.2A: SKLEARN ENSEMBLE CLASSIFIER FOR RACE WINNER PREDICTION")
    print("=" * 80)
    
    # Load Stage-2 features
    try:
        stage2_data = load_stage2_features()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Please run Stage-2 feature preparation first")
        return None
    
    # Prepare ensemble data
    data_splits = prepare_ensemble_data(stage2_data)
    
    # Train base models
    base_models = train_base_models(
        data_splits['X_train'], 
        data_splits['y_train']
    )
    
    # Train meta-learner
    meta_learner, scaler, model_names = train_meta_learner(
        base_models,
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val']
    )
    
    # Evaluate ensemble
    results = evaluate_ensemble(
        base_models, meta_learner, scaler, model_names,
        data_splits['X_test'], data_splits['y_test'], data_splits['groups_test']
    )
    
    # Save ensemble model
    print(f"\n💾 Saving ensemble model...")
    models_dir = Path('data/models')
    models_dir.mkdir(exist_ok=True)
    
    ensemble_model = {
        'base_models': base_models,
        'meta_learner': meta_learner,
        'scaler': scaler,
        'model_names': model_names,
        'feature_cols': data_splits['feature_cols'],
        'results': results
    }
    
    ensemble_file = models_dir / 'stage2_ensemble.pkl'
    joblib.dump(ensemble_model, ensemble_file)
    
    print(f"   Ensemble model saved: {ensemble_file}")
    
    # Save results summary
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    results_summary = pd.DataFrame([{
        'model': 'Ensemble_Classifier',
        'accuracy': results['accuracy'],
        'log_loss': results['log_loss'],
        'brier_score': results['brier_score'],
        'auc': results['auc'],
        'race_top1_accuracy': results['race_metrics']['top1_accuracy'],
        'race_top3_accuracy': results['race_metrics']['top3_accuracy'],
        'total_races_evaluated': results['race_metrics']['total_races']
    }])
    
    summary_file = reports_dir / 'stage2_ensemble_results.csv'
    results_summary.to_csv(summary_file, index=False)
    
    print(f"   Results summary saved: {summary_file}")
    
    print(f"\n✅ Ensemble Classifier Complete!")
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Test Accuracy: {results['accuracy']:.3f}")
    print(f"   Log Loss: {results['log_loss']:.3f}")
    print(f"   Brier Score: {results['brier_score']:.3f}")  
    print(f"   AUC: {results['auc']:.3f}")
    print(f"   Race Top-1 Accuracy: {results['race_metrics']['top1_accuracy']:.3f}")
    print(f"   Race Top-3 Accuracy: {results['race_metrics']['top3_accuracy']:.3f}")
    
    return ensemble_model

if __name__ == "__main__":
    ensemble_model = run_ensemble_classifier()
    
    if ensemble_model:
        print(f"\n🚀 Ensemble classifier ready!")
        print(f"   Next step: Build race simulator (7.2B)")
    else:
        print(f"\n❌ Ensemble classifier training failed!")