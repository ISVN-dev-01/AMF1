#!/usr/bin/env python3
"""
PHASE 6.1: Baseline FP3 Heuristic for Pole Prediction
Simple baseline: Driver with minimal FP3 time gets predicted pole
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

def load_prepared_data():
    """Load the prepared train/val/test splits"""
    
    splits_dir = Path('data/models/splits')
    
    train_data = pd.read_parquet(splits_dir / 'train_data.parquet')
    val_data = pd.read_parquet(splits_dir / 'val_data.parquet')
    test_data = pd.read_parquet(splits_dir / 'test_data.parquet')
    
    return train_data, val_data, test_data

def compute_fp3_baseline(data):
    """
    Compute FP3 baseline predictions
    For each race, predict pole for driver with minimal FP3 time (LapTimeSeconds)
    """
    
    baseline_results = []
    
    # Group by race
    for race_id, race_group in data.groupby('race_id'):
        race_data = race_group.copy()
        
        # Use LapTimeSeconds as proxy for FP3 time
        if 'LapTimeSeconds' in race_data.columns:
            # Find driver with minimal lap time
            min_time_idx = race_data['LapTimeSeconds'].idxmin()
            predicted_pole_driver = race_data.loc[min_time_idx, 'driver_id']
            min_time = race_data.loc[min_time_idx, 'LapTimeSeconds']
            
            # Create predictions for all drivers in this race
            for idx, row in race_data.iterrows():
                baseline_results.append({
                    'race_id': race_id,
                    'driver_id': row['driver_id'],
                    'actual_pole': row['is_pole'],
                    'predicted_pole': 1 if row['driver_id'] == predicted_pole_driver else 0,
                    'fp3_time': row['LapTimeSeconds'],
                    'predicted_pole_driver': predicted_pole_driver,
                    'min_fp3_time': min_time,
                    'circuit_id': row.get('circuit_id', 'unknown'),
                    'date_utc': row.get('date_utc', None)
                })
    
    return pd.DataFrame(baseline_results)

def evaluate_baseline_predictions(predictions_df):
    """Evaluate baseline predictions with multiple metrics"""
    
    results = {}
    
    # Overall accuracy metrics
    total_races = predictions_df['race_id'].nunique()
    
    # Top-1 accuracy (exact pole prediction)
    top1_correct = 0
    top3_positions = []
    top5_positions = []
    
    for race_id, race_group in predictions_df.groupby('race_id'):
        # Get actual pole position driver
        actual_pole_driver = race_group[race_group['actual_pole'] == 1]['driver_id'].values
        predicted_pole_driver = race_group['predicted_pole_driver'].iloc[0]
        
        # Top-1 accuracy
        if len(actual_pole_driver) > 0 and actual_pole_driver[0] == predicted_pole_driver:
            top1_correct += 1
        
        # For Top-3 and Top-5, we need to rank by FP3 time
        race_sorted = race_group.sort_values('fp3_time')
        top3_drivers = race_sorted.head(3)['driver_id'].tolist()
        top5_drivers = race_sorted.head(5)['driver_id'].tolist()
        
        if len(actual_pole_driver) > 0:
            actual_driver = actual_pole_driver[0]
            
            # Top-3 accuracy
            if actual_driver in top3_drivers:
                top3_positions.append(top3_drivers.index(actual_driver) + 1)
            else:
                top3_positions.append(0)  # Not in top-3
            
            # Top-5 accuracy  
            if actual_driver in top5_drivers:
                top5_positions.append(top5_drivers.index(actual_driver) + 1)
            else:
                top5_positions.append(0)  # Not in top-5
    
    # Calculate accuracies
    results['total_races'] = total_races
    results['top1_accuracy'] = top1_correct / total_races if total_races > 0 else 0
    results['top3_accuracy'] = sum(1 for pos in top3_positions if pos > 0) / len(top3_positions) if top3_positions else 0
    results['top5_accuracy'] = sum(1 for pos in top5_positions if pos > 0) / len(top5_positions) if top5_positions else 0
    
    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for pos in top5_positions:
        if pos > 0:
            reciprocal_ranks.append(1.0 / pos)
        else:
            reciprocal_ranks.append(0.0)
    
    results['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    # NDCG@5 calculation
    ndcg_scores = []
    for pos in top5_positions:
        if pos > 0:
            # DCG for single relevant item at position pos
            dcg = 1.0 / np.log2(pos + 1)
            # IDCG (perfect ranking) = 1.0 / log2(2) 
            idcg = 1.0 / np.log2(2)
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    
    results['ndcg_at_5'] = np.mean(ndcg_scores) if ndcg_scores else 0
    
    return results

def run_fp3_baseline():
    """Run complete FP3 baseline evaluation"""
    
    print("=" * 80)
    print("PHASE 6.1: FP3 BASELINE HEURISTIC")
    print("=" * 80)
    
    # Load data
    print("📁 Loading prepared data...")
    train_data, val_data, test_data = load_prepared_data()
    
    print(f"   Train: {len(train_data):,} records")
    print(f"   Val:   {len(val_data):,} records") 
    print(f"   Test:  {len(test_data):,} records")
    
    # Compute baseline predictions for each split
    results = {}
    
    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        print(f"\n🔧 Computing FP3 baseline for {split_name} set...")
        
        baseline_predictions = compute_fp3_baseline(data)
        evaluation_results = evaluate_baseline_predictions(baseline_predictions)
        
        results[split_name] = {
            'predictions': baseline_predictions,
            'metrics': evaluation_results
        }
        
        print(f"   Races evaluated: {evaluation_results['total_races']}")
        print(f"   Top-1 accuracy: {evaluation_results['top1_accuracy']:.3f}")
        print(f"   Top-3 accuracy: {evaluation_results['top3_accuracy']:.3f}")
        print(f"   Top-5 accuracy: {evaluation_results['top5_accuracy']:.3f}")
        print(f"   MRR: {evaluation_results['mrr']:.3f}")
        print(f"   NDCG@5: {evaluation_results['ndcg_at_5']:.3f}")
    
    # Save baseline results
    print(f"\n💾 Saving baseline results...")
    
    results_dir = Path('reports')
    results_dir.mkdir(exist_ok=True)
    
    # Save predictions for each split
    for split_name, split_results in results.items():
        pred_file = results_dir / f'fp3_baseline_predictions_{split_name}.csv'
        split_results['predictions'].to_csv(pred_file, index=False)
        print(f"   Saved {split_name} predictions: {pred_file}")
    
    # Create summary report
    summary_data = []
    for split_name, split_results in results.items():
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
    summary_file = results_dir / 'fp3_baseline_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    print(f"   Saved summary: {summary_file}")
    
    # Summary
    print(f"\n✅ FP3 Baseline Complete!")
    print(f"\n📊 Overall Performance Summary:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    print(f"\n🎯 Key Insights:")
    train_metrics = results['train']['metrics']
    
    print(f"   • FP3 time predicts pole with {train_metrics['top1_accuracy']:.1%} accuracy (train set)")
    print(f"   • Actual pole sitter in top-3 FP3 times {train_metrics['top3_accuracy']:.1%} of races")
    
    if train_metrics['mrr'] > 0:
        print(f"   • Mean position of pole sitter in FP3 ranking: {1/train_metrics['mrr']:.1f}")
    
    if train_metrics['top1_accuracy'] > 0.6:
        print(f"   🏆 Strong baseline - FP3 times highly predictive!")
    elif train_metrics['top1_accuracy'] > 0.4:
        print(f"   👍 Decent baseline - room for ML improvement")
    else:
        print(f"   🔧 Weak baseline - significant opportunity for ML models")
    
    return results

if __name__ == "__main__":
    results = run_fp3_baseline()
    
    # Store results for use by other models
    import pickle
    baseline_file = Path('data/models/fp3_baseline_results.pkl')
    with open(baseline_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Baseline results saved to {baseline_file}")
    print("Ready for Stage-1 model training!")