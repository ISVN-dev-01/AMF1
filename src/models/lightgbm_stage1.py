#!/usr/bin/env python3
"""
PHASE 6.2: LightGBM Regression Model for Pole Prediction
Train LightGBM to predict qualifying times, then rank for pole prediction
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_prepared_data():
    """Load the prepared train/val/test splits with features"""
    
    splits_dir = Path('data/models/splits')
    
    train_data = pd.read_parquet(splits_dir / 'train_data.parquet')
    val_data = pd.read_parquet(splits_dir / 'val_data.parquet')
    test_data = pd.read_parquet(splits_dir / 'test_data.parquet')
    
    return train_data, val_data, test_data

def prepare_features_and_targets(data, feature_cols=None):
    """Prepare features and target variable for LightGBM"""
    
    # Define feature columns if not provided
    if feature_cols is None:
        # Use all numeric columns except identifiers and target
        exclude_cols = ['race_id', 'driver_id', 'is_pole', 'date_utc', 'circuit_id']
        feature_cols = [col for col in data.columns 
                       if col not in exclude_cols and data[col].dtype in ['int64', 'float64']]
    
    # Prepare features (X) and target (y)
    X = data[feature_cols].copy()
    
    # Target: Use LapTimeSeconds as regression target
    y = data['LapTimeSeconds'].copy()
    
    # Race groups for GroupKFold
    groups = data['race_id'].values
    
    return X, y, groups, feature_cols

def train_lightgbm_regressor(X_train, y_train, groups_train, X_val=None, y_val=None):
    """Train LightGBM regression model with GroupKFold cross-validation"""
    
    print("🚀 Training LightGBM Regression Model...")
    
    # LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Cross-validation with GroupKFold
    print("   🔄 Running GroupKFold cross-validation...")
    
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    feature_importance_sum = np.zeros(X_train.shape[1])
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
        print(f"      Fold {fold + 1}/5", end=' ')
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_fold_val)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        cv_scores.append(rmse)
        
        # Accumulate feature importance
        feature_importance_sum += model.feature_importance(importance_type='gain')
        
        print(f"RMSE: {rmse:.4f}")
    
    print(f"   📊 CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on full training data
    print("   🎯 Training final model on full training data...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        final_model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
    else:
        final_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500
        )
    
    # Feature importance analysis
    feature_importance = feature_importance_sum / 5  # Average across folds
    
    return final_model, cv_scores, feature_importance

def evaluate_regression_model(model, X, y, split_name=""):
    """Evaluate regression model performance"""
    
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"   📈 {split_name} Regression Metrics:")
    print(f"      RMSE: {rmse:.4f}")
    print(f"      MAE:  {mae:.4f}")
    print(f"      R²:   {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def convert_regression_to_pole_predictions(data, y_pred, race_id_col='race_id'):
    """Convert regression predictions to pole predictions by ranking"""
    
    pole_predictions = []
    
    # Add predictions to dataframe
    data_with_pred = data.copy()
    data_with_pred['predicted_time'] = y_pred
    
    # Group by race and rank by predicted time
    for race_id, race_group in data_with_pred.groupby(race_id_col):
        race_data = race_group.copy()
        
        # Rank by predicted qualifying time (lower = better)
        race_data = race_data.sort_values('predicted_time')
        race_data['predicted_position'] = range(1, len(race_data) + 1)
        
        # Pole prediction (position 1)
        race_data['predicted_pole'] = (race_data['predicted_position'] == 1).astype(int)
        
        pole_predictions.append(race_data)
    
    return pd.concat(pole_predictions, ignore_index=True)

def evaluate_pole_predictions(predictions_df):
    """Evaluate pole predictions from regression model"""
    
    results = {}
    
    # Overall accuracy metrics
    total_races = predictions_df['race_id'].nunique()
    
    # Top-K accuracy
    top1_correct = 0
    top3_positions = []
    top5_positions = []
    
    for race_id, race_group in predictions_df.groupby('race_id'):
        # Get actual pole position driver
        actual_pole_driver = race_group[race_group['is_pole'] == 1]['driver_id'].values
        
        # Get predicted ranking
        race_sorted = race_group.sort_values('predicted_time')
        
        if len(actual_pole_driver) > 0:
            actual_driver = actual_pole_driver[0]
            
            # Find actual driver's position in prediction ranking
            driver_positions = race_sorted.reset_index(drop=True)
            try:
                actual_position = driver_positions[driver_positions['driver_id'] == actual_driver].index[0] + 1
            except IndexError:
                actual_position = len(race_sorted) + 1  # Worst case
            
            # Top-1 accuracy
            if actual_position == 1:
                top1_correct += 1
            
            # Top-K positions
            if actual_position <= 3:
                top3_positions.append(actual_position)
            else:
                top3_positions.append(0)
                
            if actual_position <= 5:
                top5_positions.append(actual_position)
            else:
                top5_positions.append(0)
    
    # Calculate metrics
    results['total_races'] = total_races
    results['top1_accuracy'] = top1_correct / total_races if total_races > 0 else 0
    results['top3_accuracy'] = sum(1 for pos in top3_positions if pos > 0) / len(top3_positions) if top3_positions else 0
    results['top5_accuracy'] = sum(1 for pos in top5_positions if pos > 0) / len(top5_positions) if top5_positions else 0
    
    # MRR
    reciprocal_ranks = []
    for pos in top5_positions:
        if pos > 0:
            reciprocal_ranks.append(1.0 / pos)
        else:
            reciprocal_ranks.append(0.0)
    
    results['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    # NDCG@5
    ndcg_scores = []
    for pos in top5_positions:
        if pos > 0:
            dcg = 1.0 / np.log2(pos + 1)
            idcg = 1.0 / np.log2(2)
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    
    results['ndcg_at_5'] = np.mean(ndcg_scores) if ndcg_scores else 0
    
    return results

def plot_feature_importance(feature_names, importance_values, top_n=15):
    """Plot feature importance"""
    
    # Get top N features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance (LightGBM)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    # Save plot
    plot_file = Path('reports/lightgbm_feature_importance.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Feature importance plot saved: {plot_file}")
    
    return importance_df

def run_lightgbm_stage1():
    """Run complete LightGBM Stage-1 pole prediction"""
    
    print("=" * 80)
    print("PHASE 6.2: LIGHTGBM REGRESSION FOR POLE PREDICTION")
    print("=" * 80)
    
    # Load data
    print("📁 Loading prepared data...")
    train_data, val_data, test_data = load_prepared_data()
    
    print(f"   Train: {len(train_data):,} records")
    print(f"   Val:   {len(val_data):,} records")
    print(f"   Test:  {len(test_data):,} records")
    
    # Prepare features and targets
    print("\n🔧 Preparing features and targets...")
    X_train, y_train, groups_train, feature_cols = prepare_features_and_targets(train_data)
    X_val, y_val, groups_val, _ = prepare_features_and_targets(val_data, feature_cols)
    X_test, y_test, groups_test, _ = prepare_features_and_targets(test_data, feature_cols)
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target: qualifying time prediction (LapTimeSeconds)")
    
    # Train LightGBM model
    print(f"\n🎯 Training LightGBM regressor...")
    model, cv_scores, feature_importance = train_lightgbm_regressor(
        X_train, y_train, groups_train, X_val, y_val
    )
    
    # Evaluate regression performance
    print(f"\n📊 Evaluating regression performance...")
    train_results = evaluate_regression_model(model, X_train, y_train, "Train")
    val_results = evaluate_regression_model(model, X_val, y_val, "Validation")
    test_results = evaluate_regression_model(model, X_test, y_test, "Test")
    
    # Convert to pole predictions
    print(f"\n🏁 Converting regression predictions to pole predictions...")
    
    pole_results = {}
    
    for split_name, (data, X, y_pred) in [
        ('train', (train_data, X_train, train_results['predictions'])),
        ('val', (val_data, X_val, val_results['predictions'])),
        ('test', (test_data, X_test, test_results['predictions']))
    ]:
        print(f"   Processing {split_name} set...")
        
        # Convert regression to pole predictions
        pole_predictions = convert_regression_to_pole_predictions(data, y_pred)
        
        # Evaluate pole prediction accuracy
        pole_metrics = evaluate_pole_predictions(pole_predictions)
        
        pole_results[split_name] = {
            'predictions': pole_predictions,
            'metrics': pole_metrics
        }
        
        print(f"      Races: {pole_metrics['total_races']}")
        print(f"      Top-1 accuracy: {pole_metrics['top1_accuracy']:.3f}")
        print(f"      Top-3 accuracy: {pole_metrics['top3_accuracy']:.3f}")
        print(f"      MRR: {pole_metrics['mrr']:.3f}")
    
    # Feature importance analysis
    print(f"\n🔍 Feature importance analysis...")
    importance_df = plot_feature_importance(feature_cols, feature_importance)
    print(f"   Top 5 features:")
    for idx, row in importance_df.head().iterrows():
        print(f"      {idx+1}. {row['feature']}: {row['importance']:.1f}")
    
    # Save model and results
    print(f"\n💾 Saving model and results...")
    
    models_dir = Path('data/models')
    models_dir.mkdir(exist_ok=True)
    
    # Save trained model
    model_file = models_dir / 'lightgbm_stage1_regressor.pkl'
    joblib.dump(model, model_file)
    print(f"   Model saved: {model_file}")
    
    # Save predictions
    reports_dir = Path('reports')
    for split_name, split_results in pole_results.items():
        pred_file = reports_dir / f'lightgbm_pole_predictions_{split_name}.csv'
        split_results['predictions'].to_csv(pred_file, index=False)
        print(f"   {split_name} predictions: {pred_file}")
    
    # Create summary report
    summary_data = []
    for split_name, split_results in pole_results.items():
        metrics = split_results['metrics']
        summary_data.append({
            'split': split_name,
            'total_races': metrics['total_races'],
            'top1_accuracy': metrics['top1_accuracy'],
            'top3_accuracy': metrics['top3_accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'mrr': metrics['mrr'],
            'ndcg_at_5': metrics['ndcg_at_5']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = reports_dir / 'lightgbm_stage1_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"   Summary: {summary_file}")
    
    # Final summary
    print(f"\n✅ LightGBM Stage-1 Complete!")
    print(f"\n📊 Pole Prediction Performance:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # Compare with baseline
    try:
        import pickle
        baseline_file = Path('data/models/fp3_baseline_results.pkl')
        if baseline_file.exists():
            with open(baseline_file, 'rb') as f:
                baseline_results = pickle.load(f)
            
            baseline_train_acc = baseline_results['train']['metrics']['top1_accuracy']
            lgb_train_acc = pole_results['train']['metrics']['top1_accuracy']
            
            improvement = lgb_train_acc - baseline_train_acc
            
            print(f"\n🎯 vs FP3 Baseline:")
            print(f"   Baseline Top-1: {baseline_train_acc:.3f}")
            print(f"   LightGBM Top-1: {lgb_train_acc:.3f}")
            print(f"   Improvement: {improvement:+.3f} ({improvement/baseline_train_acc*100:+.1f}%)")
            
    except Exception as e:
        print(f"   Could not compare with baseline: {e}")
    
    print(f"\n🚀 Ready for Stage-1 advanced models and calibration!")
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'cv_scores': cv_scores,
        'pole_results': pole_results,
        'feature_importance': importance_df
    }

if __name__ == "__main__":
    results = run_lightgbm_stage1()
    
    # Save complete results
    results_file = Path('data/models/lightgbm_stage1_results.pkl')
    with joblib.dump(results, results_file) as f:
        pass
    
    print(f"\n💾 Complete results saved to {results_file}")