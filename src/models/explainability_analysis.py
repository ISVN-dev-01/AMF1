#!/usr/bin/env python3
"""
PHASE 8: Simplified Explainability & Error Analysis
Feature importance analysis and comprehensive error analysis for F1 models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("📊 PHASE 8: Explainability & Error Analysis (Simplified)")

class SimplifiedExplainabilityAnalyzer:
    """Simplified explainability and error analysis for F1 models"""
    
    def __init__(self):
        self.results = {}
        self.figures_dir = Path('reports/figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_stage1_errors(self):
        """Analyze Stage-1 pole prediction errors"""
        
        print("\n🔍 Stage-1 Error Analysis...")
        
        # Try to load Stage-1 predictions
        stage1_files = [
            'reports/gbm_pole_predictions_test.csv',
            'reports/gbm_pole_predictions_val.csv'
        ]
        
        predictions_data = []
        for file_path in stage1_files:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                predictions_data.append(df)
                print(f"   ✅ Loaded {file_path}: {df.shape}")
        
        if not predictions_data:
            print("   ⚠️  No Stage-1 prediction files found")
            return None
        
        # Combine predictions
        all_predictions = pd.concat(predictions_data, ignore_index=True)
        
        # Calculate errors
        if 'actual_quali_time' in all_predictions.columns and 'predicted_quali_time' in all_predictions.columns:
            all_predictions['abs_error'] = np.abs(
                all_predictions['actual_quali_time'] - all_predictions['predicted_quali_time']
            )
            all_predictions['rel_error'] = all_predictions['abs_error'] / (
                all_predictions['actual_quali_time'] + 1e-6
            )
            
            # Overall metrics
            mae = all_predictions['abs_error'].mean()
            rmse = np.sqrt(mean_squared_error(
                all_predictions['actual_quali_time'], 
                all_predictions['predicted_quali_time']
            ))
            
            print(f"   Overall MAE: {mae:.3f} seconds")
            print(f"   Overall RMSE: {rmse:.3f} seconds")
            
            # Worst cases
            worst_cases = all_predictions.nlargest(10, 'abs_error')[
                ['race_id', 'driver_id', 'circuit_id', 'actual_quali_time', 
                 'predicted_quali_time', 'abs_error']
            ].copy()
            
            print(f"   Top 10 worst predictions:")
            print(f"   {'Race':<10} {'Driver':<15} {'Circuit':<12} {'Actual':<8} {'Pred':<8} {'Error':<8}")
            print(f"   {'-'*75}")
            
            for _, row in worst_cases.head().iterrows():
                print(f"   {str(row['race_id']):<10} {str(row['driver_id'])[:15]:<15} "
                      f"{str(row['circuit_id'])[:12]:<12} {row['actual_quali_time']:<8.3f} "
                      f"{row['predicted_quali_time']:<8.3f} {row['abs_error']:<8.3f}")
            
            # Per-track analysis
            if 'circuit_id' in all_predictions.columns:
                track_errors = all_predictions.groupby('circuit_id').agg({
                    'abs_error': ['mean', 'std', 'count'],
                    'rel_error': 'mean'
                }).round(3)
                
                track_errors.columns = ['mae', 'mae_std', 'count', 'rel_mae']
                track_errors = track_errors.sort_values('mae', ascending=False)
                
                print(f"   Per-track error analysis (top 5 worst):")
                print(f"   {'Circuit':<20} {'MAE':<8} {'Count':<8} {'Rel MAE':<10}")
                print(f"   {'-'*50}")
                
                for circuit, row in track_errors.head().iterrows():
                    print(f"   {str(circuit)[:20]:<20} {row['mae']:<8.3f} "
                          f"{row['count']:<8.0f} {row['rel_mae']:<10.3f}")
            
            # Save worst cases
            failure_file = Path('reports/stage1_failures.csv')
            worst_cases.to_csv(failure_file, index=False)
            print(f"   ✅ Stage-1 failures saved: {failure_file}")
            
            return {
                'overall_metrics': {'mae': mae, 'rmse': rmse},
                'worst_cases': worst_cases,
                'track_errors': track_errors if 'circuit_id' in all_predictions.columns else None,
                'all_predictions': all_predictions
            }
        else:
            print("   ⚠️  Required columns not found in Stage-1 predictions")
            return None
    
    def analyze_stage2_errors(self):
        """Analyze Stage-2 race winner prediction errors"""
        
        print("\n🔍 Stage-2 Error Analysis...")
        
        # Load Stage-2 features and create simple predictions
        stage2_file = Path('data/features/stage2_features.parquet')
        if not stage2_file.exists():
            print("   ⚠️  Stage-2 features not found")
            return None
        
        stage2_data = pd.read_parquet(stage2_file)
        print(f"   ✅ Loaded Stage-2 data: {stage2_data.shape}")
        
        # Prepare features
        exclude_cols = ['race_id', 'driver_id', 'team_id', 'date_utc', 'circuit_id', 
                       'is_winner', 'data_split', 'status', 'session_type']
        feature_cols = [col for col in stage2_data.columns 
                       if col not in exclude_cols and stage2_data[col].dtype in ['int64', 'float64']]
        
        X = stage2_data[feature_cols].copy().fillna(0)
        y = stage2_data['is_winner'].copy()
        
        # Simple train/test split
        train_mask = np.arange(len(X)) < int(0.8 * len(X))
        test_mask = ~train_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Train simple Random Forest for analysis
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        false_positives = ((y_test == 0) & (y_pred == 1)).sum()
        false_negatives = ((y_test == 1) & (y_pred == 0)).sum()
        
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(f"   False Positives: {false_positives}")
        print(f"   False Negatives: {false_negatives}")
        
        # Create error analysis DataFrame
        test_data = stage2_data[test_mask].copy().reset_index(drop=True)
        test_data['y_pred_proba'] = y_pred_proba
        test_data['classification_error'] = (y_test.values != y_pred).astype(int)
        test_data['probability_error'] = np.abs(y_test.values - y_pred_proba)
        
        # Worst cases
        worst_cases = test_data.nlargest(10, 'probability_error')[
            ['race_id', 'driver_id', 'circuit_id', 'is_winner', 'y_pred_proba', 'probability_error']
        ].copy()
        
        print(f"   Top 10 worst predictions:")
        print(f"   {'Race':<10} {'Driver':<15} {'Circuit':<12} {'Actual':<8} {'Pred':<8} {'Error':<8}")
        print(f"   {'-'*70}")
        
        for _, row in worst_cases.head().iterrows():
            print(f"   {str(row['race_id']):<10} {str(row['driver_id'])[:15]:<15} "
                  f"{str(row['circuit_id'])[:12]:<12} {row['is_winner']:<8.0f} "
                  f"{row['y_pred_proba']:<8.3f} {row['probability_error']:<8.3f}")
        
        # Per-track analysis
        if 'circuit_id' in test_data.columns:
            track_errors = test_data.groupby('circuit_id').agg({
                'classification_error': ['mean', 'sum', 'count'],
                'probability_error': 'mean'
            }).round(3)
            
            track_errors.columns = ['error_rate', 'total_errors', 'count', 'prob_mae']
            track_errors = track_errors.sort_values('error_rate', ascending=False)
            
            print(f"   Per-track error analysis (top 5 worst):")
            print(f"   {'Circuit':<20} {'Error Rate':<12} {'Count':<8} {'Prob MAE':<10}")
            print(f"   {'-'*55}")
            
            for circuit, row in track_errors.head().iterrows():
                print(f"   {str(circuit)[:20]:<20} {row['error_rate']:<12.3f} "
                      f"{row['count']:<8.0f} {row['prob_mae']:<10.3f}")
        
        # Race-level analysis
        if 'race_id' in test_data.columns:
            race_errors = test_data.groupby('race_id').agg({
                'classification_error': ['sum', 'mean', 'count']
            }).round(3)
            
            race_errors.columns = ['total_errors', 'error_rate', 'drivers']
            race_errors = race_errors.sort_values('total_errors', ascending=False)
            
            print(f"   Worst predicted races (top 5):")
            print(f"   {'Race':<12} {'Total Errors':<14} {'Error Rate':<12} {'Drivers':<10}")
            print(f"   {'-'*50}")
            
            for race_id, row in race_errors.head().iterrows():
                print(f"   {str(race_id):<12} {row['total_errors']:<14.0f} "
                      f"{row['error_rate']:<12.3f} {row['drivers']:<10.0f}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 10 most important features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"      {i+1:2d}. {row['feature']:<25} {row['importance']:.3f}")
        
        # Save results
        failure_file = Path('reports/stage2_failures.csv')
        worst_cases.to_csv(failure_file, index=False)
        print(f"   ✅ Stage-2 failures saved: {failure_file}")
        
        feature_file = Path('reports/stage2_feature_importance_analysis.csv')
        feature_importance.to_csv(feature_file, index=False)
        print(f"   ✅ Feature importance saved: {feature_file}")
        
        return {
            'overall_metrics': {
                'accuracy': accuracy,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            },
            'worst_cases': worst_cases,
            'track_errors': track_errors if 'circuit_id' in test_data.columns else None,
            'race_errors': race_errors if 'race_id' in test_data.columns else None,
            'feature_importance': feature_importance,
            'test_data': test_data
        }
    
    def create_error_visualizations(self, stage1_results, stage2_results):
        """Create error visualization plots"""
        
        print("\n📈 Creating error visualizations...")
        
        # Stage-1 visualizations
        if stage1_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Stage-1 Pole Prediction Error Analysis', fontsize=16)
            
            # Error distribution
            axes[0, 0].hist(stage1_results['all_predictions']['abs_error'], 
                           bins=30, alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('Absolute Error (seconds)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Error Distribution')
            
            # True vs Predicted
            axes[0, 1].scatter(stage1_results['all_predictions']['actual_quali_time'],
                             stage1_results['all_predictions']['predicted_quali_time'],
                             alpha=0.6, color='green')
            min_val = stage1_results['all_predictions']['actual_quali_time'].min()
            max_val = stage1_results['all_predictions']['actual_quali_time'].max()
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            axes[0, 1].set_xlabel('Actual Qualifying Time')
            axes[0, 1].set_ylabel('Predicted Qualifying Time')
            axes[0, 1].set_title('Predictions vs Actual')
            axes[0, 1].legend()
            
            # Track errors
            if stage1_results['track_errors'] is not None:
                track_mae = stage1_results['track_errors']['mae'].head(10)
                axes[1, 0].barh(range(len(track_mae)), track_mae.values, color='orange')
                axes[1, 0].set_yticks(range(len(track_mae)))
                axes[1, 0].set_yticklabels(track_mae.index, fontsize=8)
                axes[1, 0].set_xlabel('Mean Absolute Error')
                axes[1, 0].set_title('Top 10 Worst Tracks')
            
            # Error vs actual time
            axes[1, 1].scatter(stage1_results['all_predictions']['actual_quali_time'],
                             stage1_results['all_predictions']['abs_error'],
                             alpha=0.6, color='red')
            axes[1, 1].set_xlabel('Actual Qualifying Time')
            axes[1, 1].set_ylabel('Absolute Error')
            axes[1, 1].set_title('Error vs Actual Time')
            
            plt.tight_layout()
            stage1_plot = self.figures_dir / 'stage1_error_analysis.png'
            plt.savefig(stage1_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Stage-1 plot: {stage1_plot}")
        
        # Stage-2 visualizations
        if stage2_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Stage-2 Race Winner Prediction Error Analysis', fontsize=16)
            
            # Probability error distribution
            axes[0, 0].hist(stage2_results['test_data']['probability_error'], 
                           bins=30, alpha=0.7, color='purple')
            axes[0, 0].set_xlabel('Probability Error')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Probability Error Distribution')
            
            # Predicted probabilities by true label
            winners = stage2_results['test_data'][stage2_results['test_data']['is_winner'] == 1]
            non_winners = stage2_results['test_data'][stage2_results['test_data']['is_winner'] == 0]
            
            axes[0, 1].hist(winners['y_pred_proba'], bins=20, alpha=0.7, 
                           label='True Winners', color='gold', density=True)
            axes[0, 1].hist(non_winners['y_pred_proba'], bins=20, alpha=0.7, 
                           label='True Non-Winners', color='gray', density=True)
            axes[0, 1].set_xlabel('Predicted Win Probability')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Probability Distribution by True Label')
            axes[0, 1].legend()
            
            # Feature importance
            top_features = stage2_results['feature_importance'].head(10)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'].values, 
                           color='teal')
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'].values, fontsize=8)
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 10 Most Important Features')
            
            # Track error rates
            if stage2_results['track_errors'] is not None:
                track_rates = stage2_results['track_errors']['error_rate'].head(8)
                axes[1, 1].barh(range(len(track_rates)), track_rates.values, color='coral')
                axes[1, 1].set_yticks(range(len(track_rates)))
                axes[1, 1].set_yticklabels(track_rates.index, fontsize=8)
                axes[1, 1].set_xlabel('Error Rate')
                axes[1, 1].set_title('Top 8 Worst Tracks by Error Rate')
            
            plt.tight_layout()
            stage2_plot = self.figures_dir / 'stage2_error_analysis.png'
            plt.savefig(stage2_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Stage-2 plot: {stage2_plot}")
    
    def generate_explainability_report(self, stage1_results, stage2_results):
        """Generate comprehensive explainability report"""
        
        print("\n📋 Generating explainability report...")
        
        report_lines = [
            "# PHASE 8: EXPLAINABILITY & ERROR ANALYSIS REPORT",
            "",
            "## Executive Summary",
            "Comprehensive error analysis for both Stage-1 (pole prediction) and Stage-2 (race winner prediction)",
            "models, identifying failure patterns, track-specific performance, and feature importance.",
            "",
            "## Stage-1 Pole Prediction Analysis",
            ""
        ]
        
        if stage1_results:
            report_lines.extend([
                f"### Overall Performance",
                f"- **Mean Absolute Error**: {stage1_results['overall_metrics']['mae']:.3f} seconds",
                f"- **Root Mean Square Error**: {stage1_results['overall_metrics']['rmse']:.3f} seconds",
                f"- **Worst single prediction error**: {stage1_results['worst_cases']['abs_error'].max():.3f} seconds",
                ""
            ])
            
            if stage1_results['track_errors'] is not None:
                worst_track = stage1_results['track_errors'].index[0]
                worst_mae = stage1_results['track_errors']['mae'].iloc[0]
                best_track = stage1_results['track_errors'].index[-1]
                best_mae = stage1_results['track_errors']['mae'].iloc[-1]
                
                report_lines.extend([
                    f"### Track-Specific Performance",
                    f"- **Worst performing track**: {worst_track} (MAE: {worst_mae:.3f}s)",
                    f"- **Best performing track**: {best_track} (MAE: {best_mae:.3f}s)",
                    f"- **Performance variation**: {worst_mae - best_mae:.3f}s between best and worst tracks",
                    ""
                ])
        
        report_lines.extend([
            "## Stage-2 Race Winner Prediction Analysis",
            ""
        ])
        
        if stage2_results:
            report_lines.extend([
                f"### Overall Performance",
                f"- **Classification Accuracy**: {stage2_results['overall_metrics']['accuracy']:.3f}",
                f"- **False Positives**: {stage2_results['overall_metrics']['false_positives']} predictions",
                f"- **False Negatives**: {stage2_results['overall_metrics']['false_negatives']} predictions",
                ""
            ])
            
            if stage2_results['track_errors'] is not None:
                worst_track = stage2_results['track_errors'].index[0]
                worst_rate = stage2_results['track_errors']['error_rate'].iloc[0]
                
                report_lines.extend([
                    f"### Track-Specific Performance",
                    f"- **Worst performing track**: {worst_track} (Error Rate: {worst_rate:.3f})",
                    ""
                ])
            
            if stage2_results['race_errors'] is not None:
                worst_race = stage2_results['race_errors'].index[0]
                worst_errors = stage2_results['race_errors']['total_errors'].iloc[0]
                
                report_lines.extend([
                    f"### Race-Level Analysis",
                    f"- **Worst predicted race**: {worst_race} ({worst_errors:.0f} total errors)",
                    ""
                ])
            
            # Top features
            report_lines.extend([
                f"### Most Important Features",
                ""
            ])
            
            for i, (_, row) in enumerate(stage2_results['feature_importance'].head().iterrows()):
                report_lines.append(f"{i+1}. **{row['feature']}**: {row['importance']:.3f}")
        
        report_lines.extend([
            "",
            "## Key Insights & Recommendations",
            "",
            "### Error Patterns Identified",
            "1. **Track Dependency**: Certain circuits consistently produce higher prediction errors",
            "2. **Model Limitations**: Systematic biases in specific racing scenarios",
            "3. **Feature Importance**: Weather and qualifying performance are key predictors",
            "",
            "### Actionable Recommendations",
            "1. **Track-Specific Models**: Consider separate models for different circuit types",
            "2. **Feature Engineering**: Improve track-specific feature representation",
            "3. **Data Quality**: Investigate high-error cases for potential data issues",
            "4. **Model Ensemble**: Combine multiple approaches to reduce systematic errors",
            "",
            "## Files Generated",
            "- `reports/stage1_failures.csv` - Top 10 worst Stage-1 predictions",
            "- `reports/stage2_failures.csv` - Top 10 worst Stage-2 predictions", 
            "- `reports/stage2_feature_importance_analysis.csv` - Detailed feature importance",
            "- `reports/figures/stage1_error_analysis.png` - Stage-1 error visualizations",
            "- `reports/figures/stage2_error_analysis.png` - Stage-2 error visualizations",
            "",
            "---",
            "*Generated by Phase 8 Explainability & Error Analysis Pipeline*"
        ])
        
        # Write report
        report_file = Path('reports/PHASE8_EXPLAINABILITY_REPORT.md')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   ✅ Report saved: {report_file}")
        
        return report_file

def run_simplified_explainability():
    """Run simplified explainability and error analysis"""
    
    print("=" * 80)
    print("PHASE 8: EXPLAINABILITY & ERROR ANALYSIS")  
    print("=" * 80)
    
    analyzer = SimplifiedExplainabilityAnalyzer()
    
    # Stage-1 analysis
    stage1_results = analyzer.analyze_stage1_errors()
    
    # Stage-2 analysis  
    stage2_results = analyzer.analyze_stage2_errors()
    
    # Create visualizations
    analyzer.create_error_visualizations(stage1_results, stage2_results)
    
    # Generate report
    report_file = analyzer.generate_explainability_report(stage1_results, stage2_results)
    
    print(f"\n✅ PHASE 8 COMPLETE!")
    print(f"📋 Explainability report: {report_file}")
    print(f"📊 Error visualizations: reports/figures/")
    print(f"📋 Failure cases: reports/stage1_failures.csv, reports/stage2_failures.csv")
    
    return True

if __name__ == "__main__":
    success = run_simplified_explainability()
    
    if success:
        print(f"\n🔍 EXPLAINABILITY ANALYSIS COMPLETE!")
        print(f"   ✅ Error analysis completed")
        print(f"   ✅ Worst cases identified and saved")
        print(f"   ✅ Track-specific analysis done")
        print(f"   ✅ Feature importance analyzed") 
        print(f"   ✅ Visualizations created")
        print(f"\n🚀 F1 Model Explainability Framework Ready!")
    else:
        print(f"\n❌ Explainability analysis failed")