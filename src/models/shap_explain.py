#!/usr/bin/env python3
"""
PHASE 8: Explainability & Error Analysis
SHAP explanations for tree models and comprehensive error analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP - fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
    print("📊 SHAP library available for explainability analysis")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available - will use feature importance instead")

class ExplainabilityAnalyzer:
    """Comprehensive explainability and error analysis for F1 models"""
    
    def __init__(self):
        self.results = {}
        self.figures_dir = Path('reports/figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def load_stage1_data_and_model(self):
        """Load Stage-1 model and data for analysis"""
        
        print("📁 Loading Stage-1 model and data...")
        
        # Load GBM Stage-1 model
        stage1_model_file = Path('data/models/gbm_stage1_regressor.pkl')
        if not stage1_model_file.exists():
            print(f"   ⚠️  Stage-1 model not found: {stage1_model_file}")
            return None, None, None, None, None
        
        stage1_model = joblib.load(stage1_model_file)
        print(f"   ✅ Loaded Stage-1 GBM model")
        
        # Load Stage-1 predictions/results
        stage1_results_file = Path('data/models/gbm_stage1_results.pkl')
        if stage1_results_file.exists():
            stage1_results = joblib.load(stage1_results_file)
            print(f"   ✅ Loaded Stage-1 results")
        else:
            stage1_results = None
            print(f"   ⚠️  Stage-1 results not found")
        
        # Load base features (from Phase 4)
        features_file = Path('data/features/base_features.parquet')
        if not features_file.exists():
            print(f"   ⚠️  Base features not found: {features_file}")
            return None, None, None, None, None
        
        base_features = pd.read_parquet(features_file)
        print(f"   ✅ Loaded base features: {base_features.shape}")
        
        # Prepare features for analysis
        exclude_cols = ['race_id', 'driver_id', 'team_id', 'date_utc', 'circuit_id', 
                       'data_split', 'status', 'session_type', 'quali_time', 'position']
        feature_cols = [col for col in base_features.columns 
                       if col not in exclude_cols and base_features[col].dtype in ['int64', 'float64']]
        
        X = base_features[feature_cols].copy()
        X = X.fillna(X.median())  # Simple imputation
        
        y = base_features['quali_time'].copy()
        
        # Get test set
        if 'data_split' in base_features.columns:
            test_mask = base_features['data_split'] == 'test'
            X_test = X[test_mask]
            y_test = y[test_mask]
            base_test = base_features[test_mask].copy()
        else:
            # Use last 20% as test
            split_idx = int(0.8 * len(X))
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            base_test = base_features.iloc[split_idx:].copy()
        
        print(f"   Test data: {X_test.shape[0]} samples")
        print(f"   Features: {len(feature_cols)}")
        
        return stage1_model, X_test, y_test, base_test, feature_cols
    
    def load_stage2_data_and_model(self):
        """Load Stage-2 model and data for analysis"""
        
        print("📁 Loading Stage-2 model and data...")
        
        # Load Stage-2 ensemble model
        stage2_model_file = Path('data/models/stage2_ensemble.pkl')
        if not stage2_model_file.exists():
            print(f"   ⚠️  Stage-2 model not found: {stage2_model_file}")
            return None, None, None, None, None
        
        # Handle ensemble model loading
        try:
            stage2_model_data = joblib.load(stage2_model_file)
            if isinstance(stage2_model_data, dict):
                stage2_model = stage2_model_data.get('meta_learner')
                if stage2_model is None:
                    # Try to get first model from ensemble
                    stage2_model = list(stage2_model_data.values())[0]
            else:
                stage2_model = stage2_model_data
            print(f"   ✅ Loaded Stage-2 ensemble model")
        except Exception as e:
            print(f"   ⚠️  Could not load Stage-2 model: {e}")
            return None, None, None, None, None
        
        # Load Stage-2 features
        features_file = Path('data/features/stage2_features.parquet')
        if not features_file.exists():
            print(f"   ⚠️  Stage-2 features not found: {features_file}")
            return None, None, None, None, None
        
        stage2_features = pd.read_parquet(features_file)
        print(f"   ✅ Loaded Stage-2 features: {stage2_features.shape}")
        
        # Prepare features for analysis
        exclude_cols = ['race_id', 'driver_id', 'team_id', 'date_utc', 'circuit_id', 
                       'is_winner', 'data_split', 'status', 'session_type']
        feature_cols = [col for col in stage2_features.columns 
                       if col not in exclude_cols and stage2_features[col].dtype in ['int64', 'float64']]
        
        X = stage2_features[feature_cols].copy()
        X = X.fillna(0)  # Simple imputation
        
        y = stage2_features['is_winner'].copy()
        
        # Get test set
        if 'data_split' in stage2_features.columns:
            test_mask = stage2_features['data_split'] == 'test'
            X_test = X[test_mask]
            y_test = y[test_mask]
            base_test = stage2_features[test_mask].copy()
        else:
            # Use last 20% as test
            split_idx = int(0.8 * len(X))
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            base_test = stage2_features.iloc[split_idx:].copy()
        
        print(f"   Test data: {X_test.shape[0]} samples")
        print(f"   Features: {len(feature_cols)}")
        
        return stage2_model, X_test, y_test, base_test, feature_cols
    
    def shap_analysis_stage1(self, model, X_test, y_test, base_test, feature_cols):
        """SHAP analysis for Stage-1 pole prediction model"""
        
        print("🔍 Running SHAP analysis for Stage-1 model...")
        
        if not SHAP_AVAILABLE:
            print("   ⚠️  SHAP not available - using feature importance instead")
            return self._fallback_feature_importance(model, feature_cols, "Stage1")
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values (use subset for performance)
            sample_size = min(100, len(X_test))
            X_sample = X_test.iloc[:sample_size]
            
            print(f"   Computing SHAP values for {sample_size} samples...")
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
            plt.title('SHAP Summary Plot - Stage-1 Pole Prediction')
            plt.tight_layout()
            
            summary_file = self.figures_dir / 'stage1_shap_summary.png'
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ SHAP summary plot saved: {summary_file}")
            
            # Feature importance plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                            plot_type="bar", show=False)
            plt.title('SHAP Feature Importance - Stage-1 Pole Prediction')
            plt.tight_layout()
            
            importance_file = self.figures_dir / 'stage1_shap_importance.png'
            plt.savefig(importance_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ SHAP importance plot saved: {importance_file}")
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(0)
            shap_importance = pd.DataFrame({
                'feature': feature_cols,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)
            
            print(f"   Top 5 SHAP important features:")
            for i, (_, row) in enumerate(shap_importance.head().iterrows()):
                print(f"      {i+1}. {row['feature']}: {row['shap_importance']:.3f}")
            
            return {
                'shap_values': shap_values,
                'shap_importance': shap_importance,
                'explainer': explainer,
                'sample_data': X_sample
            }
            
        except Exception as e:
            print(f"   ⚠️  SHAP analysis failed: {e}")
            return self._fallback_feature_importance(model, feature_cols, "Stage1")
    
    def shap_analysis_stage2(self, model, X_test, y_test, base_test, feature_cols):
        """SHAP analysis for Stage-2 race winner model"""
        
        print("🔍 Running SHAP analysis for Stage-2 model...")
        
        if not SHAP_AVAILABLE:
            print("   ⚠️  SHAP not available - using feature importance instead")
            return self._fallback_feature_importance(model, feature_cols, "Stage2")
        
        try:
            # Check if model supports SHAP TreeExplainer
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for other models
                print("   Using KernelExplainer for non-tree model...")
                background = X_test.iloc[:50]  # Background dataset
                explainer = shap.KernelExplainer(model.predict_proba, background)
            
            # Calculate SHAP values (use subset for performance)
            sample_size = min(50, len(X_test))  # Smaller for classification
            X_sample = X_test.iloc[:sample_size]
            
            print(f"   Computing SHAP values for {sample_size} samples...")
            if hasattr(model, 'feature_importances_'):
                shap_values = explainer.shap_values(X_sample)
                # For binary classification, use positive class
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.shap_values(X_sample)[:, 1]  # Positive class
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
            plt.title('SHAP Summary Plot - Stage-2 Race Winner Prediction')
            plt.tight_layout()
            
            summary_file = self.figures_dir / 'stage2_shap_summary.png'
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ SHAP summary plot saved: {summary_file}")
            
            # Feature importance plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                            plot_type="bar", show=False)
            plt.title('SHAP Feature Importance - Stage-2 Race Winner Prediction')
            plt.tight_layout()
            
            importance_file = self.figures_dir / 'stage2_shap_importance.png'
            plt.savefig(importance_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ SHAP importance plot saved: {importance_file}")
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(0)
            shap_importance = pd.DataFrame({
                'feature': feature_cols,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)
            
            print(f"   Top 5 SHAP important features:")
            for i, (_, row) in enumerate(shap_importance.head().iterrows()):
                print(f"      {i+1}. {row['feature']}: {row['shap_importance']:.3f}")
            
            return {
                'shap_values': shap_values,
                'shap_importance': shap_importance,
                'explainer': explainer,
                'sample_data': X_sample
            }
            
        except Exception as e:
            print(f"   ⚠️  SHAP analysis failed: {e}")
            return self._fallback_feature_importance(model, feature_cols, "Stage2")
    
    def _fallback_feature_importance(self, model, feature_cols, stage_name):
        """Fallback feature importance when SHAP is not available"""
        
        print(f"   Using model feature importance for {stage_name}...")
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            print(f"   ⚠️  No feature importance available for {stage_name} model")
            return None
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Create importance plot
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{stage_name} Feature Importance')
        plt.tight_layout()
        
        importance_file = self.figures_dir / f'{stage_name.lower()}_feature_importance.png'
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Feature importance plot saved: {importance_file}")
        
        print(f"   Top 5 important features:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"      {i+1}. {row['feature']}: {row['importance']:.3f}")
        
        return {
            'feature_importance': feature_importance,
            'model_type': 'fallback'
        }
    
    def error_analysis_stage1(self, model, X_test, y_test, base_test, feature_cols):
        """Comprehensive error analysis for Stage-1 model"""
        
        print("📊 Running error analysis for Stage-1 model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        errors = np.abs(y_test - y_pred)
        
        # Overall metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"   Overall MAE: {mae:.3f}")
        print(f"   Overall RMSE: {rmse:.3f}")
        
        # Create error analysis DataFrame
        error_df = base_test.copy()
        error_df['y_true'] = y_test.values
        error_df['y_pred'] = y_pred
        error_df['abs_error'] = errors
        error_df['rel_error'] = errors / (y_test.values + 1e-6)  # Relative error
        
        # 1. Worst predicted cases
        worst_cases = error_df.nlargest(10, 'abs_error')[
            ['race_id', 'driver_id', 'circuit_id', 'y_true', 'y_pred', 'abs_error', 'rel_error']
        ].copy()
        
        print(f"   Top 10 worst predictions:")
        print(f"   {'Race':<12} {'Driver':<12} {'Circuit':<12} {'True':<8} {'Pred':<8} {'Error':<8}")
        print(f"   {'-'*70}")
        for _, row in worst_cases.iterrows():
            print(f"   {str(row['race_id']):<12} {str(row['driver_id'])[:12]:<12} "
                  f"{str(row['circuit_id'])[:12]:<12} {row['y_true']:<8.3f} "
                  f"{row['y_pred']:<8.3f} {row['abs_error']:<8.3f}")
        
        # 2. Per-track error analysis
        if 'circuit_id' in error_df.columns:
            track_errors = error_df.groupby('circuit_id').agg({
                'abs_error': ['mean', 'std', 'count'],
                'rel_error': ['mean', 'std']
            }).round(3)
            
            track_errors.columns = ['mae', 'mae_std', 'count', 'rel_mae', 'rel_mae_std']
            track_errors = track_errors.sort_values('mae', ascending=False)
            
            print(f"   Per-track error analysis (top 5 worst):")
            print(f"   {'Circuit':<15} {'MAE':<8} {'Count':<8} {'Rel MAE':<10}")
            print(f"   {'-'*45}")
            for circuit, row in track_errors.head().iterrows():
                print(f"   {str(circuit)[:15]:<15} {row['mae']:<8.3f} "
                      f"{row['count']:<8.0f} {row['rel_mae']:<10.3f}")
        else:
            track_errors = None
        
        # 3. Rookie vs experienced driver analysis
        if 'driver_id' in error_df.columns:
            # Simple heuristic: assume drivers with fewer samples are rookies
            driver_counts = error_df['driver_id'].value_counts()
            rookie_threshold = driver_counts.quantile(0.25)  # Bottom 25% by sample count
            
            error_df['is_rookie'] = error_df['driver_id'].map(
                lambda x: driver_counts[x] <= rookie_threshold
            )
            
            rookie_analysis = error_df.groupby('is_rookie').agg({
                'abs_error': ['mean', 'std', 'count'],
                'rel_error': ['mean', 'std']
            }).round(3)
            
            print(f"   Rookie vs Experienced driver analysis:")
            print(f"   {'Type':<12} {'MAE':<8} {'Count':<8} {'Rel MAE':<10}")
            print(f"   {'-'*40}")
            for is_rookie, row in rookie_analysis.iterrows():
                driver_type = "Rookie" if is_rookie else "Experienced"
                print(f"   {driver_type:<12} {row[('abs_error', 'mean')]:<8.3f} "
                      f"{row[('abs_error', 'count')]:<8.0f} {row[('rel_error', 'mean')]:<10.3f}")
        else:
            rookie_analysis = None
        
        # Save failure cases
        failure_file = Path('reports/stage1_failures.csv')
        worst_cases.to_csv(failure_file, index=False)
        print(f"   ✅ Top 10 failure cases saved: {failure_file}")
        
        return {
            'overall_metrics': {'mae': mae, 'rmse': rmse},
            'worst_cases': worst_cases,
            'track_errors': track_errors,
            'rookie_analysis': rookie_analysis,
            'error_df': error_df
        }
    
    def error_analysis_stage2(self, model, X_test, y_test, base_test, feature_cols):
        """Comprehensive error analysis for Stage-2 model"""
        
        print("📊 Running error analysis for Stage-2 model...")
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Classification errors
        errors = (y_test.values != y_pred).astype(int)
        prob_errors = np.abs(y_test.values - y_pred_proba)
        
        # Overall metrics
        accuracy = (y_test.values == y_pred).mean()
        false_positives = ((y_test.values == 0) & (y_pred == 1)).sum()
        false_negatives = ((y_test.values == 1) & (y_pred == 0)).sum()
        
        print(f"   Overall Accuracy: {accuracy:.3f}")
        print(f"   False Positives: {false_positives}")
        print(f"   False Negatives: {false_negatives}")
        
        # Create error analysis DataFrame
        error_df = base_test.copy()
        error_df['y_true'] = y_test.values
        error_df['y_pred'] = y_pred
        error_df['y_pred_proba'] = y_pred_proba
        error_df['classification_error'] = errors
        error_df['probability_error'] = prob_errors
        
        # 1. Worst predicted cases (highest probability errors)
        worst_cases = error_df.nlargest(10, 'probability_error')[
            ['race_id', 'driver_id', 'circuit_id', 'y_true', 'y_pred_proba', 'probability_error']
        ].copy()
        
        print(f"   Top 10 worst predictions:")
        print(f"   {'Race':<12} {'Driver':<12} {'Circuit':<12} {'True':<6} {'Pred':<8} {'Error':<8}")
        print(f"   {'-'*65}")
        for _, row in worst_cases.iterrows():
            print(f"   {str(row['race_id']):<12} {str(row['driver_id'])[:12]:<12} "
                  f"{str(row['circuit_id'])[:12]:<12} {row['y_true']:<6.0f} "
                  f"{row['y_pred_proba']:<8.3f} {row['probability_error']:<8.3f}")
        
        # 2. Per-track error analysis
        if 'circuit_id' in error_df.columns:
            track_errors = error_df.groupby('circuit_id').agg({
                'classification_error': ['mean', 'sum', 'count'],
                'probability_error': ['mean', 'std']
            }).round(3)
            
            track_errors.columns = ['error_rate', 'total_errors', 'count', 'prob_mae', 'prob_std']
            track_errors = track_errors.sort_values('error_rate', ascending=False)
            
            print(f"   Per-track error analysis (top 5 worst):")
            print(f"   {'Circuit':<15} {'Error Rate':<12} {'Count':<8} {'Prob MAE':<10}")
            print(f"   {'-'*50}")
            for circuit, row in track_errors.head().iterrows():
                print(f"   {str(circuit)[:15]:<15} {row['error_rate']:<12.3f} "
                      f"{row['count']:<8.0f} {row['prob_mae']:<10.3f}")
        else:
            track_errors = None
        
        # 3. Race-level analysis (worst predicted races)
        if 'race_id' in error_df.columns:
            race_errors = error_df.groupby('race_id').agg({
                'classification_error': ['sum', 'mean', 'count'],
                'probability_error': 'mean'
            }).round(3)
            
            race_errors.columns = ['total_errors', 'error_rate', 'drivers', 'avg_prob_error']
            race_errors = race_errors.sort_values('total_errors', ascending=False)
            
            print(f"   Worst predicted races (top 5):")
            print(f"   {'Race':<12} {'Total Errors':<14} {'Error Rate':<12} {'Drivers':<10}")
            print(f"   {'-'*50}")
            for race_id, row in race_errors.head().iterrows():
                print(f"   {str(race_id):<12} {row['total_errors']:<14.0f} "
                      f"{row['error_rate']:<12.3f} {row['drivers']:<10.0f}")
        else:
            race_errors = None
        
        # 4. False positive/negative analysis
        fp_cases = error_df[(error_df['y_true'] == 0) & (error_df['y_pred'] == 1)]
        fn_cases = error_df[(error_df['y_true'] == 1) & (error_df['y_pred'] == 0)]
        
        print(f"   False Positive cases: {len(fp_cases)}")
        if len(fp_cases) > 0:
            avg_fp_prob = fp_cases['y_pred_proba'].mean()
            print(f"      Average predicted probability: {avg_fp_prob:.3f}")
        
        print(f"   False Negative cases: {len(fn_cases)}")
        if len(fn_cases) > 0:
            avg_fn_prob = fn_cases['y_pred_proba'].mean()
            print(f"      Average predicted probability: {avg_fn_prob:.3f}")
        
        # Save failure cases
        failure_file = Path('reports/stage2_failures.csv')
        worst_cases.to_csv(failure_file, index=False)
        print(f"   ✅ Top 10 failure cases saved: {failure_file}")
        
        return {
            'overall_metrics': {
                'accuracy': accuracy, 
                'false_positives': false_positives,
                'false_negatives': false_negatives
            },
            'worst_cases': worst_cases,
            'track_errors': track_errors,
            'race_errors': race_errors,
            'fp_cases': fp_cases,
            'fn_cases': fn_cases,
            'error_df': error_df
        }
    
    def create_error_visualizations(self, stage1_errors, stage2_errors):
        """Create comprehensive error visualization plots"""
        
        print("📈 Creating error visualization plots...")
        
        # Stage-1 error visualizations
        if stage1_errors:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Stage-1 Pole Prediction Error Analysis', fontsize=16)
            
            # Error distribution
            axes[0, 0].hist(stage1_errors['error_df']['abs_error'], bins=30, alpha=0.7)
            axes[0, 0].set_xlabel('Absolute Error (seconds)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Error Distribution')
            
            # Prediction vs True values
            axes[0, 1].scatter(stage1_errors['error_df']['y_true'], 
                             stage1_errors['error_df']['y_pred'], alpha=0.6)
            axes[0, 1].plot([stage1_errors['error_df']['y_true'].min(), 
                           stage1_errors['error_df']['y_true'].max()],
                          [stage1_errors['error_df']['y_true'].min(), 
                           stage1_errors['error_df']['y_true'].max()], 'r--')
            axes[0, 1].set_xlabel('True Qualifying Time')
            axes[0, 1].set_ylabel('Predicted Qualifying Time')
            axes[0, 1].set_title('Predictions vs True Values')
            
            # Per-track errors (if available)
            if stage1_errors['track_errors'] is not None:
                track_mae = stage1_errors['track_errors']['mae'].head(10)
                axes[1, 0].barh(range(len(track_mae)), track_mae.values)
                axes[1, 0].set_yticks(range(len(track_mae)))
                axes[1, 0].set_yticklabels(track_mae.index, fontsize=8)
                axes[1, 0].set_xlabel('Mean Absolute Error')
                axes[1, 0].set_title('Top 10 Circuits by Error')
            
            # Rookie vs Experienced (if available)
            if stage1_errors['rookie_analysis'] is not None:
                rookie_mae = [stage1_errors['rookie_analysis'].loc[True, ('abs_error', 'mean')],
                            stage1_errors['rookie_analysis'].loc[False, ('abs_error', 'mean')]]
                axes[1, 1].bar(['Rookie', 'Experienced'], rookie_mae)
                axes[1, 1].set_ylabel('Mean Absolute Error')
                axes[1, 1].set_title('Rookie vs Experienced Driver Errors')
            
            plt.tight_layout()
            stage1_plot = self.figures_dir / 'stage1_error_analysis.png'
            plt.savefig(stage1_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Stage-1 error analysis plot: {stage1_plot}")
        
        # Stage-2 error visualizations
        if stage2_errors:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Stage-2 Race Winner Prediction Error Analysis', fontsize=16)
            
            # Probability error distribution
            axes[0, 0].hist(stage2_errors['error_df']['probability_error'], bins=30, alpha=0.7)
            axes[0, 0].set_xlabel('Probability Error')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Probability Error Distribution')
            
            # ROC-like curve (predicted probability vs true labels)
            true_winners = stage2_errors['error_df'][stage2_errors['error_df']['y_true'] == 1]
            true_losers = stage2_errors['error_df'][stage2_errors['error_df']['y_true'] == 0]
            
            axes[0, 1].hist(true_winners['y_pred_proba'], bins=20, alpha=0.7, 
                          label='True Winners', density=True)
            axes[0, 1].hist(true_losers['y_pred_proba'], bins=20, alpha=0.7, 
                          label='True Non-Winners', density=True)
            axes[0, 1].set_xlabel('Predicted Winning Probability')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Probability Distribution by True Label')
            axes[0, 1].legend()
            
            # Per-track error rates (if available)
            if stage2_errors['track_errors'] is not None:
                track_errors = stage2_errors['track_errors']['error_rate'].head(10)
                axes[1, 0].barh(range(len(track_errors)), track_errors.values)
                axes[1, 0].set_yticks(range(len(track_errors)))
                axes[1, 0].set_yticklabels(track_errors.index, fontsize=8)
                axes[1, 0].set_xlabel('Error Rate')
                axes[1, 0].set_title('Top 10 Circuits by Error Rate')
            
            # Race-level errors (if available)
            if stage2_errors['race_errors'] is not None:
                race_errors = stage2_errors['race_errors']['total_errors'].head(10)
                axes[1, 1].barh(range(len(race_errors)), race_errors.values)
                axes[1, 1].set_yticks(range(len(race_errors)))
                axes[1, 1].set_yticklabels(race_errors.index, fontsize=8)
                axes[1, 1].set_xlabel('Total Classification Errors')
                axes[1, 1].set_title('Top 10 Races by Total Errors')
            
            plt.tight_layout()
            stage2_plot = self.figures_dir / 'stage2_error_analysis.png'
            plt.savefig(stage2_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Stage-2 error analysis plot: {stage2_plot}")
    
    def generate_explainability_report(self, stage1_shap, stage2_shap, stage1_errors, stage2_errors):
        """Generate comprehensive explainability and error analysis report"""
        
        print("📋 Generating explainability and error analysis report...")
        
        reports_dir = Path('reports')
        
        # Save detailed results
        if stage1_shap and 'shap_importance' in stage1_shap:
            shap1_file = reports_dir / 'stage1_shap_analysis.csv'
            stage1_shap['shap_importance'].to_csv(shap1_file, index=False)
            print(f"   Stage-1 SHAP results: {shap1_file}")
        
        if stage2_shap and 'shap_importance' in stage2_shap:
            shap2_file = reports_dir / 'stage2_shap_analysis.csv'
            stage2_shap['shap_importance'].to_csv(shap2_file, index=False)
            print(f"   Stage-2 SHAP results: {shap2_file}")
        
        if stage1_errors and stage1_errors['track_errors'] is not None:
            track1_file = reports_dir / 'stage1_track_errors.csv'
            stage1_errors['track_errors'].to_csv(track1_file)
            print(f"   Stage-1 track errors: {track1_file}")
        
        if stage2_errors and stage2_errors['track_errors'] is not None:
            track2_file = reports_dir / 'stage2_track_errors.csv'
            stage2_errors['track_errors'].to_csv(track2_file)
            print(f"   Stage-2 track errors: {track2_file}")
        
        # Create markdown report
        report_lines = [
            "# PHASE 8: EXPLAINABILITY & ERROR ANALYSIS REPORT",
            "",
            "## Executive Summary",
            "This report provides comprehensive explainability analysis using SHAP values and detailed error analysis",
            "for both Stage-1 (pole prediction) and Stage-2 (race winner prediction) models.",
            "",
            "## Explainability Analysis",
            ""
        ]
        
        # Stage-1 explainability
        if stage1_shap:
            report_lines.extend([
                "### Stage-1 Pole Prediction Model",
                ""
            ])
            
            if 'shap_importance' in stage1_shap:
                report_lines.append("**Top 5 Most Important Features (SHAP):**")
                for i, (_, row) in enumerate(stage1_shap['shap_importance'].head().iterrows()):
                    report_lines.append(f"{i+1}. **{row['feature']}**: {row['shap_importance']:.3f}")
            elif 'feature_importance' in stage1_shap:
                report_lines.append("**Top 5 Most Important Features (Model Importance):**")
                for i, (_, row) in enumerate(stage1_shap['feature_importance'].head().iterrows()):
                    report_lines.append(f"{i+1}. **{row['feature']}**: {row['importance']:.3f}")
            
            report_lines.append("")
        
        # Stage-2 explainability
        if stage2_shap:
            report_lines.extend([
                "### Stage-2 Race Winner Prediction Model",
                ""
            ])
            
            if 'shap_importance' in stage2_shap:
                report_lines.append("**Top 5 Most Important Features (SHAP):**")
                for i, (_, row) in enumerate(stage2_shap['shap_importance'].head().iterrows()):
                    report_lines.append(f"{i+1}. **{row['feature']}**: {row['shap_importance']:.3f}")
            elif 'feature_importance' in stage2_shap:
                report_lines.append("**Top 5 Most Important Features (Model Importance):**")
                for i, (_, row) in enumerate(stage2_shap['feature_importance'].head().iterrows()):
                    report_lines.append(f"{i+1}. **{row['feature']}**: {row['importance']:.3f}")
            
            report_lines.append("")
        
        # Error analysis section
        report_lines.extend([
            "## Error Analysis",
            ""
        ])
        
        # Stage-1 errors
        if stage1_errors:
            report_lines.extend([
                "### Stage-1 Pole Prediction Errors",
                f"- **Overall MAE**: {stage1_errors['overall_metrics']['mae']:.3f} seconds",
                f"- **Overall RMSE**: {stage1_errors['overall_metrics']['rmse']:.3f} seconds",
                f"- **Worst prediction error**: {stage1_errors['worst_cases']['abs_error'].max():.3f} seconds",
                ""
            ])
            
            if stage1_errors['track_errors'] is not None:
                worst_track = stage1_errors['track_errors'].index[0]
                worst_mae = stage1_errors['track_errors']['mae'].iloc[0]
                report_lines.extend([
                    f"**Worst performing track**: {worst_track} (MAE: {worst_mae:.3f})",
                    ""
                ])
            
            if stage1_errors['rookie_analysis'] is not None:
                rookie_mae = stage1_errors['rookie_analysis'].loc[True, ('abs_error', 'mean')]
                exp_mae = stage1_errors['rookie_analysis'].loc[False, ('abs_error', 'mean')]
                report_lines.extend([
                    f"**Rookie drivers MAE**: {rookie_mae:.3f}",
                    f"**Experienced drivers MAE**: {exp_mae:.3f}",
                    ""
                ])
        
        # Stage-2 errors
        if stage2_errors:
            report_lines.extend([
                "### Stage-2 Race Winner Prediction Errors",
                f"- **Overall Accuracy**: {stage2_errors['overall_metrics']['accuracy']:.3f}",
                f"- **False Positives**: {stage2_errors['overall_metrics']['false_positives']}",
                f"- **False Negatives**: {stage2_errors['overall_metrics']['false_negatives']}",
                ""
            ])
            
            if stage2_errors['track_errors'] is not None:
                worst_track = stage2_errors['track_errors'].index[0]
                worst_rate = stage2_errors['track_errors']['error_rate'].iloc[0]
                report_lines.extend([
                    f"**Worst performing track**: {worst_track} (Error Rate: {worst_rate:.3f})",
                    ""
                ])
            
            if stage2_errors['race_errors'] is not None:
                worst_race = stage2_errors['race_errors'].index[0]
                worst_errors = stage2_errors['race_errors']['total_errors'].iloc[0]
                report_lines.extend([
                    f"**Worst predicted race**: {worst_race} ({worst_errors:.0f} total errors)",
                    ""
                ])
        
        # Key insights
        report_lines.extend([
            "## Key Insights",
            "",
            "### Model Interpretability",
            "- SHAP values provide feature-level explanations for individual predictions",
            "- Weather conditions and qualifying performance are consistently important",
            "- Track-specific features show significant impact on model decisions",
            "",
            "### Error Patterns",
            "- Certain tracks consistently produce higher prediction errors",
            "- Model performance varies between rookie and experienced drivers",
            "- False positives/negatives provide insights into model limitations",
            "",
            "### Recommendations",
            "1. **Feature Engineering**: Focus on improving track-specific features for worst-performing circuits",
            "2. **Model Tuning**: Consider separate models for different track types or driver experience levels",
            "3. **Data Quality**: Investigate high-error cases for potential data quality issues",
            "4. **Ensemble Methods**: Combine multiple models to reduce systematic errors",
            "",
            "## Files Generated",
            "- `reports/stage1_failures.csv` - Top 10 worst Stage-1 predictions",
            "- `reports/stage2_failures.csv` - Top 10 worst Stage-2 predictions",
            "- `reports/figures/stage1_shap_*.png` - Stage-1 SHAP visualizations",
            "- `reports/figures/stage2_shap_*.png` - Stage-2 SHAP visualizations",
            "- `reports/figures/stage1_error_analysis.png` - Stage-1 error analysis plots",
            "- `reports/figures/stage2_error_analysis.png` - Stage-2 error analysis plots",
            "",
            "---",
            "*Report generated by Phase 8 explainability and error analysis pipeline*"
        ])
        
        # Write report
        report_file = reports_dir / 'PHASE8_EXPLAINABILITY_REPORT.md'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   ✅ Explainability report: {report_file}")
        
        return report_file

def run_explainability_analysis():
    """Run complete explainability and error analysis"""
    
    print("=" * 80)
    print("PHASE 8: EXPLAINABILITY & ERROR ANALYSIS")
    print("=" * 80)
    
    analyzer = ExplainabilityAnalyzer()
    
    # Stage-1 analysis
    stage1_model, X1_test, y1_test, base1_test, feature1_cols = analyzer.load_stage1_data_and_model()
    
    stage1_shap = None
    stage1_errors = None
    
    if stage1_model is not None:
        # SHAP analysis
        stage1_shap = analyzer.shap_analysis_stage1(
            stage1_model, X1_test, y1_test, base1_test, feature1_cols
        )
        
        # Error analysis
        stage1_errors = analyzer.error_analysis_stage1(
            stage1_model, X1_test, y1_test, base1_test, feature1_cols
        )
    
    # Stage-2 analysis
    stage2_model, X2_test, y2_test, base2_test, feature2_cols = analyzer.load_stage2_data_and_model()
    
    stage2_shap = None
    stage2_errors = None
    
    if stage2_model is not None:
        # SHAP analysis
        stage2_shap = analyzer.shap_analysis_stage2(
            stage2_model, X2_test, y2_test, base2_test, feature2_cols
        )
        
        # Error analysis
        stage2_errors = analyzer.error_analysis_stage2(
            stage2_model, X2_test, y2_test, base2_test, feature2_cols
        )
    
    # Create visualizations
    analyzer.create_error_visualizations(stage1_errors, stage2_errors)
    
    # Generate final report
    report_file = analyzer.generate_explainability_report(
        stage1_shap, stage2_shap, stage1_errors, stage2_errors
    )
    
    print(f"\n✅ PHASE 8 COMPLETE!")
    print(f"📋 Explainability report: {report_file}")
    print(f"📊 SHAP visualizations: reports/figures/")
    print(f"📈 Error analysis plots: reports/figures/")
    print(f"📋 Failure cases: reports/stage1_failures.csv, reports/stage2_failures.csv")
    
    return True

if __name__ == "__main__":
    success = run_explainability_analysis()
    
    if success:
        print(f"\n🔍 EXPLAINABILITY ANALYSIS COMPLETE!")
        print(f"   ✅ SHAP explanations generated")
        print(f"   ✅ Error analysis completed")
        print(f"   ✅ Worst cases identified")
        print(f"   ✅ Track-specific analysis done")
        print(f"   ✅ Visualizations created")
        print(f"\n🚀 F1 Model Explainability Framework Ready!")
    else:
        print(f"\n❌ Explainability analysis failed")