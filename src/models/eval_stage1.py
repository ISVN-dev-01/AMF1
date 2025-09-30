#!/usr/bin/env python3
"""
PHASE 6.5: Comprehensive Evaluation Pipeline for Stage-1 Models
Compare all models with detailed per-track analysis and visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_all_model_results():
    """Load results from all Stage-1 models"""
    
    results = {}
    
    # Load FP3 baseline results
    baseline_file = Path('data/models/fp3_baseline_results.pkl')
    if baseline_file.exists():
        with open(baseline_file, 'rb') as f:
            results['fp3_baseline'] = pickle.load(f)
    
    # Load GBM results
    gbm_file = Path('data/models/gbm_stage1_results.pkl')
    if gbm_file.exists():
        results['gbm'] = joblib.load(gbm_file)
    
    return results

def create_model_comparison_summary(all_results):
    """Create comprehensive model comparison summary"""
    
    summary_data = []
    
    for model_name, model_results in all_results.items():
        
        if model_name == 'fp3_baseline':
            # Handle baseline format
            for split_name in ['train', 'val', 'test']:
                if split_name in model_results:
                    metrics = model_results[split_name]['metrics']
                    summary_data.append({
                        'model': 'FP3_Baseline',
                        'split': split_name,
                        'total_races': metrics['total_races'],
                        'top1_accuracy': metrics['top1_accuracy'],
                        'top3_accuracy': metrics['top3_accuracy'],
                        'top5_accuracy': metrics['top5_accuracy'],
                        'mrr': metrics['mrr'],
                        'ndcg_at_5': metrics['ndcg_at_5']
                    })
        
        else:
            # Handle ML model format
            pole_results = model_results.get('pole_results', {})
            for split_name in ['train', 'val', 'test']:
                if split_name in pole_results:
                    metrics = pole_results[split_name]['metrics']
                    summary_data.append({
                        'model': model_name.upper(),
                        'split': split_name,
                        'total_races': metrics['total_races'],
                        'top1_accuracy': metrics['top1_accuracy'],
                        'top3_accuracy': metrics['top3_accuracy'],
                        'top5_accuracy': metrics['top5_accuracy'],
                        'mrr': metrics['mrr'],
                        'ndcg_at_5': metrics['ndcg_at_5']
                    })
    
    return pd.DataFrame(summary_data)

def analyze_per_track_performance(all_results):
    """Analyze performance per track/circuit"""
    
    track_analysis = []
    
    for model_name, model_results in all_results.items():
        
        if model_name == 'fp3_baseline':
            # Handle baseline format
            for split_name in ['train', 'val', 'test']:
                if split_name in model_results and 'predictions' in model_results[split_name]:
                    predictions_df = model_results[split_name]['predictions']
                    
                    # Group by circuit
                    for circuit_id, circuit_group in predictions_df.groupby('circuit_id'):
                        races_count = circuit_group['race_id'].nunique()
                        
                        # Calculate track-specific metrics
                        track_top1_acc = 0
                        track_positions = []
                        
                        for race_id, race_group in circuit_group.groupby('race_id'):
                            actual_pole = race_group[race_group['actual_pole'] == 1]['driver_id'].values
                            predicted_pole = race_group['predicted_pole_driver'].iloc[0]
                            
                            if len(actual_pole) > 0:
                                if actual_pole[0] == predicted_pole:
                                    track_top1_acc += 1
                                
                                # Find position of actual pole sitter in FP3 ranking
                                race_sorted = race_group.sort_values('fp3_time')
                                try:
                                    actual_pos = race_sorted[race_sorted['driver_id'] == actual_pole[0]].index[0]
                                    track_positions.append(actual_pos + 1)
                                except:
                                    track_positions.append(len(race_sorted) + 1)
                        
                        track_analysis.append({
                            'model': 'FP3_Baseline',
                            'split': split_name,
                            'circuit_id': circuit_id,
                            'races_count': races_count,
                            'top1_accuracy': track_top1_acc / races_count if races_count > 0 else 0,
                            'avg_position': np.mean(track_positions) if track_positions else 0,
                            'mrr': np.mean([1/pos if pos <= 5 else 0 for pos in track_positions]) if track_positions else 0
                        })
    
    return pd.DataFrame(track_analysis)

def create_performance_visualizations(summary_df, output_dir):
    """Create performance comparison visualizations"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model comparison across splits
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Stage-1 Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['top1_accuracy', 'top3_accuracy', 'mrr', 'ndcg_at_5']
    metric_names = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Mean Reciprocal Rank', 'NDCG@5']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        # Filter to train split for main comparison
        train_data = summary_df[summary_df['split'] == 'train']
        
        if not train_data.empty:
            sns.barplot(data=train_data, x='model', y=metric, ax=ax)
            ax.set_title(f'{metric_name} (Train Set)')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(train_data[metric]):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stage1_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance across splits
    if len(summary_df['split'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot_data = summary_df.pivot(index='model', columns='split', values='top1_accuracy')
        pivot_data.plot(kind='bar', ax=ax)
        ax.set_title('Top-1 Accuracy Across Train/Val/Test Splits', fontsize=14, fontweight='bold')
        ax.set_ylabel('Top-1 Accuracy')
        ax.legend(title='Split')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'stage1_split_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"   📊 Visualizations saved to {output_dir}")

def generate_detailed_report(summary_df, track_df, output_dir):
    """Generate detailed performance report"""
    
    output_dir = Path(output_dir)
    
    report_lines = []
    report_lines.append("# FORMULA 1 POLE PREDICTION - STAGE-1 EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## EXECUTIVE SUMMARY")
    report_lines.append("")
    
    train_summary = summary_df[summary_df['split'] == 'train']
    if not train_summary.empty:
        best_model = train_summary.loc[train_summary['top1_accuracy'].idxmax()]
        report_lines.append(f"**Best Model:** {best_model['model']}")
        report_lines.append(f"**Best Top-1 Accuracy:** {best_model['top1_accuracy']:.3f} ({best_model['top1_accuracy']*100:.1f}%)")
        report_lines.append(f"**Best MRR:** {best_model['mrr']:.3f}")
        report_lines.append("")
    
    # Model Performance Table
    report_lines.append("## MODEL PERFORMANCE COMPARISON")
    report_lines.append("")
    report_lines.append("### Train Set Performance")
    report_lines.append("")
    
    if not train_summary.empty:
        train_table = train_summary[['model', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy', 'mrr', 'ndcg_at_5']].round(3)
        report_lines.append(train_table.to_string(index=False))
        report_lines.append("")
    
    # Detailed Analysis
    report_lines.append("## DETAILED ANALYSIS")
    report_lines.append("")
    
    # Compare models
    if len(train_summary) > 1:
        baseline_acc = train_summary[train_summary['model'] == 'FP3_BASELINE']['top1_accuracy'].iloc[0] if 'FP3_BASELINE' in train_summary['model'].values else 0
        
        report_lines.append("### Model vs Baseline Comparison")
        report_lines.append("")
        
        for _, row in train_summary.iterrows():
            if row['model'] != 'FP3_BASELINE':
                improvement = row['top1_accuracy'] - baseline_acc
                report_lines.append(f"**{row['model']}:**")
                report_lines.append(f"- Top-1 Accuracy: {row['top1_accuracy']:.3f} (vs baseline: {improvement:+.3f})")
                report_lines.append(f"- Improvement: {improvement/baseline_acc*100:+.1f}%" if baseline_acc > 0 else "- Improvement: N/A")
                report_lines.append("")
    
    # Key Insights
    report_lines.append("## KEY INSIGHTS")
    report_lines.append("")
    
    if not train_summary.empty:
        max_acc = train_summary['top1_accuracy'].max()
        max_mrr = train_summary['mrr'].max()
        
        if max_acc < 0.1:
            report_lines.append("- **Low pole prediction accuracy** - Room for significant improvement")
        elif max_acc < 0.3:
            report_lines.append("- **Moderate pole prediction accuracy** - Models showing promise")
        else:
            report_lines.append("- **Good pole prediction accuracy** - Models performing well")
        
        if max_mrr > 0.6:
            report_lines.append("- **Strong ranking performance** - Pole sitter often in top 3 predictions")
        else:
            report_lines.append("- **Moderate ranking performance** - Need better ranking of candidates")
    
    # Data Quality Issues
    val_summary = summary_df[summary_df['split'] == 'val']
    test_summary = summary_df[summary_df['split'] == 'test']
    
    if not val_summary.empty and val_summary['top1_accuracy'].max() == 0:
        report_lines.append("- **Data Quality Issue:** Validation set contains no pole positions")
    
    if not test_summary.empty and test_summary['top1_accuracy'].max() == 0:
        report_lines.append("- **Data Quality Issue:** Test set contains no pole positions")
    
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## RECOMMENDATIONS")
    report_lines.append("")
    report_lines.append("1. **Fix data splits** - Ensure validation/test sets contain pole positions")
    report_lines.append("2. **Feature engineering** - Add circuit-specific and weather features")
    report_lines.append("3. **Advanced models** - Try neural networks and ensemble methods")
    report_lines.append("4. **Calibration** - Implement probability calibration for confidence scores")
    report_lines.append("")
    
    # Save report
    report_file = output_dir / 'stage1_evaluation_report.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   📋 Detailed report saved: {report_file}")

def run_comprehensive_evaluation():
    """Run complete Stage-1 evaluation pipeline"""
    
    print("=" * 80)
    print("PHASE 6.5: COMPREHENSIVE STAGE-1 EVALUATION")
    print("=" * 80)
    
    # Load all model results
    print("📁 Loading all model results...")
    all_results = load_all_model_results()
    
    print(f"   Models loaded: {list(all_results.keys())}")
    
    if not all_results:
        print("❌ No model results found. Please run baseline and ML models first.")
        return
    
    # Create model comparison summary
    print("\n📊 Creating model comparison summary...")
    summary_df = create_model_comparison_summary(all_results)
    
    print(f"   Summary shape: {summary_df.shape}")
    print(f"   Models: {summary_df['model'].unique()}")
    print(f"   Splits: {summary_df['split'].unique()}")
    
    # Per-track analysis
    print("\n🏁 Analyzing per-track performance...")
    track_df = analyze_per_track_performance(all_results)
    
    print(f"   Track analysis shape: {track_df.shape}")
    if not track_df.empty:
        print(f"   Circuits analyzed: {track_df['circuit_id'].nunique()}")
    
    # Save results
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Save summary tables
    summary_file = reports_dir / 'stage1_model_comparison.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Model comparison saved: {summary_file}")
    
    if not track_df.empty:
        track_file = reports_dir / 'stage1_per_track_analysis.csv'
        track_df.to_csv(track_file, index=False)
        print(f"   Per-track analysis saved: {track_file}")
    
    # Create visualizations
    print("\n🎨 Creating performance visualizations...")
    create_performance_visualizations(summary_df, reports_dir)
    
    # Generate detailed report
    print("\n📋 Generating detailed evaluation report...")
    generate_detailed_report(summary_df, track_df, reports_dir)
    
    # Display summary
    print("\n✅ Comprehensive Evaluation Complete!")
    print("\n📊 FINAL STAGE-1 RESULTS:")
    print("=" * 60)
    
    # Show train set performance
    train_summary = summary_df[summary_df['split'] == 'train']
    if not train_summary.empty:
        print("Train Set Performance:")
        display_cols = ['model', 'top1_accuracy', 'top3_accuracy', 'mrr', 'ndcg_at_5']
        print(train_summary[display_cols].to_string(index=False, float_format='%.3f'))
    
    # Show best model
    if not train_summary.empty:
        best_model = train_summary.loc[train_summary['top1_accuracy'].idxmax()]
        print(f"\n🏆 BEST MODEL: {best_model['model']}")
        print(f"   Top-1 Accuracy: {best_model['top1_accuracy']:.3f} ({best_model['top1_accuracy']*100:.1f}%)")
        print(f"   Top-3 Accuracy: {best_model['top3_accuracy']:.3f} ({best_model['top3_accuracy']*100:.1f}%)")
        print(f"   MRR: {best_model['mrr']:.3f}")
        print(f"   NDCG@5: {best_model['ndcg_at_5']:.3f}")
    
    # Highlight data issues
    val_summary = summary_df[summary_df['split'] == 'val']
    if not val_summary.empty and val_summary['top1_accuracy'].max() == 0:
        print(f"\n⚠️  DATA QUALITY ISSUE: Validation set has no pole positions")
    
    test_summary = summary_df[summary_df['split'] == 'test']
    if not test_summary.empty and test_summary['top1_accuracy'].max() == 0:
        print(f"⚠️  DATA QUALITY ISSUE: Test set has no pole positions")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Fix data quality issues in validation/test splits")
    print(f"   2. Implement Stage-2 classification models")
    print(f"   3. Add probability calibration")
    print(f"   4. Build ensemble methods")
    
    return {
        'summary': summary_df,
        'track_analysis': track_df,
        'all_results': all_results
    }

if __name__ == "__main__":
    evaluation_results = run_comprehensive_evaluation()
    
    # Save evaluation results
    eval_file = Path('data/models/stage1_evaluation_results.pkl')
    joblib.dump(evaluation_results, eval_file)
    
    print(f"\n💾 Evaluation results saved to {eval_file}")
    print("Ready for Stage-2 development!")