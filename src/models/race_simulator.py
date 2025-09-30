#!/usr/bin/env python3
"""
PHASE 7.2B: Race Simulator for Winner Prediction
Lightweight Monte Carlo race simulator with stochastic overtakes, pit stops, and safety cars
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

class F1RaceSimulator:
    """Lightweight F1 race simulator for winner prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Simulation parameters
        self.default_race_distance = 55  # laps
        self.pit_stop_time_loss = 25.0   # seconds
        self.safety_car_duration = 8     # laps
        
    def simulate_race(self, race_data, n_simulations=500):
        """
        Simulate a single race multiple times
        
        Args:
            race_data: DataFrame with driver data for one race
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            dict: Win probabilities for each driver
        """
        
        drivers = race_data['driver_id'].values
        n_drivers = len(drivers)
        
        # Extract race features
        grid_positions = race_data['predicted_grid_position'].values
        overtake_skills = race_data['driver_overtake_skill'].values
        reliability = race_data['team_reliability'].values
        safety_car_prob = race_data['safety_car_prob_track'].iloc[0]
        race_distance = race_data['race_distance_laps'].iloc[0]
        
        # Initialize win counters
        win_counts = {driver: 0 for driver in drivers}
        
        # Run simulations
        for sim in range(n_simulations):
            race_result = self._simulate_single_race(
                drivers, grid_positions, overtake_skills, 
                reliability, safety_car_prob, race_distance
            )
            
            # Award win to race winner
            if race_result:
                winner = race_result[0]  # First place
                win_counts[winner] += 1
        
        # Convert to probabilities
        win_probabilities = {
            driver: count / n_simulations 
            for driver, count in win_counts.items()
        }
        
        return win_probabilities
    
    def _simulate_single_race(self, drivers, grid_positions, overtake_skills, 
                             reliability, safety_car_prob, race_distance):
        """Simulate a single race instance"""
        
        n_drivers = len(drivers)
        
        # Initialize race state
        current_positions = grid_positions.copy()
        lap_times = np.random.normal(90.0, 2.0, n_drivers)  # Base lap times
        pit_stops_completed = np.zeros(n_drivers, dtype=bool)
        
        # Add grid position advantage to lap times (better grid = faster lap)
        position_advantage = 1.0 / np.sqrt(grid_positions)
        lap_times = lap_times - position_advantage * 0.5
        
        # Reliability check - drivers may DNF
        dnf_mask = np.random.random(n_drivers) > reliability
        
        # Simulate race progression
        total_race_time = np.zeros(n_drivers)
        
        for lap in range(int(race_distance)):
            
            # Add random lap time variation
            lap_time_variation = np.random.normal(0, 0.5, n_drivers)
            current_lap_times = lap_times + lap_time_variation
            
            # Safety car probability
            if np.random.random() < safety_car_prob / race_distance:
                # Safety car neutralizes race - equalize lap times
                current_lap_times = np.full(n_drivers, 92.0)
            
            # Overtaking simulation (simplified)
            if lap > 5:  # Overtakes more likely after first few laps
                self._simulate_overtakes(
                    current_positions, overtake_skills, current_lap_times
                )
            
            # Pit stop strategy (around mid-race)
            pit_window_start = race_distance * 0.3
            pit_window_end = race_distance * 0.7
            
            if pit_window_start <= lap <= pit_window_end:
                for i in range(n_drivers):
                    if not pit_stops_completed[i] and np.random.random() < 0.15:
                        # Pit this driver
                        current_lap_times[i] += self.pit_stop_time_loss
                        pit_stops_completed[i] = True
            
            # Accumulate race time
            total_race_time += current_lap_times
            
            # Apply DNF mask
            total_race_time[dnf_mask] = np.inf
        
        # Mandatory pit stop - penalize drivers who didn't pit
        for i in range(n_drivers):
            if not pit_stops_completed[i]:
                total_race_time[i] += 60.0  # Time penalty
        
        # Final race order (fastest time wins)
        race_order = np.argsort(total_race_time)
        
        return [drivers[i] for i in race_order if not dnf_mask[i]]
    
    def _simulate_overtakes(self, positions, overtake_skills, lap_times):
        """Simulate position changes during a lap"""
        
        n_drivers = len(positions)
        
        # Create position-speed pairs
        driver_data = list(zip(range(n_drivers), positions, overtake_skills, lap_times))
        
        # Sort by current position
        driver_data.sort(key=lambda x: x[1])
        
        # Simulate overtakes between adjacent positions
        for i in range(len(driver_data) - 1):
            behind_idx, behind_pos, behind_skill, behind_time = driver_data[i]
            ahead_idx, ahead_pos, ahead_skill, ahead_time = driver_data[i + 1]
            
            # Overtake probability based on skill difference and lap time
            skill_diff = behind_skill - ahead_skill
            time_diff = ahead_time - behind_time  # Negative if behind driver is faster
            
            overtake_prob = 0.05 + 0.1 * skill_diff + 0.05 * time_diff
            overtake_prob = max(0, min(0.3, overtake_prob))  # Cap at 30%
            
            if np.random.random() < overtake_prob:
                # Swap positions
                positions[behind_idx], positions[ahead_idx] = ahead_pos, behind_pos
    
    def simulate_race_batch(self, stage2_data, n_simulations=500):
        """Simulate all races in the dataset"""
        
        print(f"🏁 Running race simulations...")
        print(f"   Simulations per race: {n_simulations}")
        
        all_results = []
        
        for race_id in stage2_data['race_id'].unique():
            race_data = stage2_data[stage2_data['race_id'] == race_id].copy()
            
            if len(race_data) == 0:
                continue
                
            print(f"   Simulating race {race_id} with {len(race_data)} drivers...")
            
            # Run simulation for this race
            win_probs = self.simulate_race(race_data, n_simulations)
            
            # Create results for this race
            for _, row in race_data.iterrows():
                driver_id = row['driver_id']
                sim_win_prob = win_probs.get(driver_id, 0.0)
                
                result = {
                    'race_id': race_id,
                    'driver_id': driver_id,
                    'actual_winner': row['is_winner'],
                    'simulator_win_prob': sim_win_prob,
                    'predicted_grid_position': row['predicted_grid_position'],
                    'driver_overtake_skill': row['driver_overtake_skill'],
                    'team_reliability': row['team_reliability'],
                    'safety_car_prob_track': row['safety_car_prob_track']
                }
                
                all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        
        print(f"   Simulation complete!")
        print(f"   Total race-driver combinations: {len(results_df)}")
        print(f"   Average win probability: {results_df['simulator_win_prob'].mean():.3f}")
        
        return results_df

def load_stage2_features():
    """Load Stage-2 features"""
    
    features_file = Path('data/features/stage2_features.parquet')
    if not features_file.exists():
        raise FileNotFoundError(f"Stage-2 features not found: {features_file}")
    
    print(f"📁 Loading Stage-2 features from {features_file}")
    stage2_data = pd.read_parquet(features_file)
    
    return stage2_data

def prepare_simulator_features(stage2_data, simulator_results):
    """Combine original features with simulator outputs"""
    
    print(f"🔧 Preparing simulator-enhanced features...")
    
    # Merge simulator results with original data
    enhanced_data = stage2_data.merge(
        simulator_results[['race_id', 'driver_id', 'simulator_win_prob']], 
        on=['race_id', 'driver_id'], 
        how='left'
    )
    
    enhanced_data['simulator_win_prob'] = enhanced_data['simulator_win_prob'].fillna(0.0)
    
    print(f"   Enhanced data shape: {enhanced_data.shape}")
    print(f"   Simulator probabilities range: {enhanced_data['simulator_win_prob'].min():.3f} - {enhanced_data['simulator_win_prob'].max():.3f}")
    
    return enhanced_data

def train_simulator_classifier(enhanced_data):
    """Train classifier on simulator-enhanced features"""
    
    print(f"🎯 Training simulator-enhanced classifier...")
    
    # Prepare features - include simulator probabilities
    exclude_cols = ['race_id', 'driver_id', 'team_id', 'date_utc', 'circuit_id', 
                   'is_winner', 'data_split', 'status', 'session_type']
    feature_cols = [col for col in enhanced_data.columns 
                   if col not in exclude_cols and enhanced_data[col].dtype in ['int64', 'float64']]
    
    print(f"   Features: {len(feature_cols)} (including simulator_win_prob)")
    
    # Handle missing values
    X = enhanced_data[feature_cols].copy()
    X = X.fillna(0)
    
    y = enhanced_data['is_winner'].copy()
    groups = enhanced_data['race_id'].values
    
    # Split data
    if 'data_split' in enhanced_data.columns:
        train_mask = enhanced_data['data_split'] == 'train'
        test_mask = enhanced_data['data_split'] == 'test'
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_test = groups[test_mask]
    else:
        # Simple split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        groups_test = groups[split_idx:]
    
    print(f"   Train: {X_train.shape[0]} samples, {y_train.sum()} winners")
    print(f"   Test:  {X_test.shape[0]} samples, {y_test.sum()} winners")
    
    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate
    test_predictions = rf_classifier.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, test_predictions > 0.5)
    test_logloss = log_loss(y_test, test_predictions)
    
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"   Test log loss: {test_logloss:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"   Top 5 features:")
    for idx, (_, row) in enumerate(feature_importance.head().iterrows()):
        print(f"      {idx+1}. {row['feature']}: {row['importance']:.3f}")
    
    # Race-level evaluation
    race_metrics = calculate_race_level_metrics(X_test, y_test, groups_test, test_predictions)
    
    return {
        'model': rf_classifier,
        'feature_cols': feature_cols,
        'feature_importance': feature_importance,
        'test_accuracy': test_accuracy,
        'test_logloss': test_logloss,
        'race_metrics': race_metrics,
        'predictions': test_predictions
    }

def calculate_race_level_metrics(X_test, y_test, groups_test, predictions):
    """Calculate race-level performance metrics"""
    
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
        
        if race_group['actual_winner'].sum() > 0:
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

def run_race_simulator():
    """Run complete race simulator pipeline"""
    
    print("=" * 80)
    print("PHASE 7.2B: RACE SIMULATOR FOR WINNER PREDICTION")
    print("=" * 80)
    
    # Load Stage-2 features
    try:
        stage2_data = load_stage2_features()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    
    # Initialize simulator
    simulator = F1RaceSimulator(random_state=42)
    
    # Run simulations
    simulator_results = simulator.simulate_race_batch(stage2_data, n_simulations=1000)
    
    # Prepare enhanced features
    enhanced_data = prepare_simulator_features(stage2_data, simulator_results)
    
    # Train simulator-enhanced classifier
    results = train_simulator_classifier(enhanced_data)
    
    # Save simulator and results
    print(f"\n💾 Saving simulator and results...")
    
    models_dir = Path('data/models')
    models_dir.mkdir(exist_ok=True)
    
    # Save simulator
    simulator_file = models_dir / 'race_simulator.pkl'
    joblib.dump(simulator, simulator_file)
    print(f"   Race simulator saved: {simulator_file}")
    
    # Save enhanced classifier
    classifier_file = models_dir / 'stage2_simulator_classifier.pkl'
    joblib.dump(results['model'], classifier_file)
    print(f"   Simulator classifier saved: {classifier_file}")
    
    # Save results
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Simulator results
    sim_results_file = reports_dir / 'race_simulator_results.csv'
    simulator_results.to_csv(sim_results_file, index=False)
    print(f"   Simulator results saved: {sim_results_file}")
    
    # Feature importance
    importance_file = reports_dir / 'simulator_feature_importance.csv'
    results['feature_importance'].to_csv(importance_file, index=False)
    print(f"   Feature importance saved: {importance_file}")
    
    # Performance summary
    summary_data = [{
        'model': 'Simulator_Enhanced_RF',
        'test_accuracy': results['test_accuracy'],  
        'test_logloss': results['test_logloss'],
        'race_top1_accuracy': results['race_metrics']['top1_accuracy'],
        'race_top3_accuracy': results['race_metrics']['top3_accuracy'],
        'total_races': results['race_metrics']['total_races']
    }]
    
    summary_file = reports_dir / 'stage2_simulator_results.csv'
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"   Performance summary saved: {summary_file}")
    
    print(f"\n✅ Race Simulator Complete!")
    print(f"\n🏆 SIMULATOR RESULTS:")
    print(f"   Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"   Test Log Loss: {results['test_logloss']:.3f}")
    print(f"   Race Top-1 Accuracy: {results['race_metrics']['top1_accuracy']:.3f}")
    print(f"   Race Top-3 Accuracy: {results['race_metrics']['top3_accuracy']:.3f}")
    
    # Compare with ensemble
    try:
        ensemble_file = reports_dir / 'stage2_ensemble_results.csv'
        if ensemble_file.exists():
            ensemble_results = pd.read_csv(ensemble_file)
            ensemble_top1 = ensemble_results['race_top1_accuracy'].iloc[0]
            simulator_top1 = results['race_metrics']['top1_accuracy']
            
            improvement = simulator_top1 - ensemble_top1
            print(f"\n🎯 vs Ensemble Classifier:")
            print(f"   Ensemble Top-1: {ensemble_top1:.3f}")
            print(f"   Simulator Top-1: {simulator_top1:.3f}")
            print(f"   Improvement: {improvement:+.3f}")
            
            if improvement > 0.05:
                print(f"   🚀 Significant improvement with race simulation!")
            elif improvement > 0:
                print(f"   👍 Simulator shows improvement")
            else:
                print(f"   🔧 Ensemble performed better")
    except Exception as e:
        print(f"   Could not compare with ensemble: {e}")
    
    return {
        'simulator': simulator,
        'classifier': results['model'],
        'results': results,
        'simulator_results': simulator_results
    }

if __name__ == "__main__":
    simulator_results = run_race_simulator()
    
    if simulator_results:
        print(f"\n🚀 Race simulator ready!")
        print(f"   Next step: Final evaluation and comparison (7.3)")
    else:
        print(f"\n❌ Race simulator training failed!")