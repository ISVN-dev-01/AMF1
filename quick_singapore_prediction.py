#!/usr/bin/env python3
"""
Quick Singapore GP 2025 Prediction CLI
Fast access to Stage-2 model predictions
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def quick_singapore_prediction():
    """Generate quick Singapore GP 2025 prediction"""
    
    print("🏁 Singapore GP 2025 - Quick Prediction")
    print("=" * 45)
    
    # Check if model exists
    model_path = Path("models/production/stage2_marina_rf.pkl")
    if not model_path.exists():
        print("❌ Model not found. Please train first:")
        print("   python3 src/experiments/train_stage2_marina_bay_simplified.py --train")
        return
    
    # Load model
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        print("✅ Stage-2 Marina Bay model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Singapore GP 2025 data
    drivers = {
        63: {"name": "George Russell", "team": "Mercedes", "grid": 1},
        1: {"name": "Max Verstappen", "team": "Red Bull Racing", "grid": 3},
        81: {"name": "Oscar Piastri", "team": "McLaren", "grid": 4},
        4: {"name": "Lando Norris", "team": "McLaren", "grid": 2},
        16: {"name": "Charles Leclerc", "team": "Ferrari", "grid": 5},
        55: {"name": "Carlos Sainz", "team": "Ferrari", "grid": 6},
        44: {"name": "Lewis Hamilton", "team": "Mercedes", "grid": 7},
        14: {"name": "Fernando Alonso", "team": "Aston Martin", "grid": 8},
        11: {"name": "Sergio Pérez", "team": "Red Bull Racing", "grid": 9},
        18: {"name": "Lance Stroll", "team": "Aston Martin", "grid": 10}
    }
    
    # Create feature matrix
    features_data = []
    for driver_id, info in drivers.items():
        # Marina Bay features (simplified)
        if driver_id == 44:  # Hamilton - Marina Bay specialist
            marina_features = {
                'marina_count': 8, 'marina_wins': 3, 'marina_podiums': 6,
                'marina_avg_finish': 3.2, 'marina_recency_finish': 3.5, 'marina_best_last3': 1.0
            }
        elif driver_id == 63:  # Russell - Mercedes strength
            marina_features = {
                'marina_count': 3, 'marina_wins': 0, 'marina_podiums': 1,
                'marina_avg_finish': 6.0, 'marina_recency_finish': 5.5, 'marina_best_last3': 4.0
            }
        elif driver_id == 1:  # Verstappen
            marina_features = {
                'marina_count': 5, 'marina_wins': 0, 'marina_podiums': 2,
                'marina_avg_finish': 4.8, 'marina_recency_finish': 4.2, 'marina_best_last3': 3.0
            }
        else:
            marina_features = {
                'marina_count': 3, 'marina_wins': 0, 'marina_podiums': 1,
                'marina_avg_finish': 8.5, 'marina_recency_finish': 8.0, 'marina_best_last3': 6.0
            }
        
        # 2025 season form (based on actual performance)
        if driver_id == 4:  # Norris - championship leader
            season_features = {
                'season_points_to_date': 350, 'season_podiums': 12,
                'season_avg_finish_last3': 2.3, 'season_quali_avg_last3': 3.1
            }
        elif driver_id == 81:  # Piastri
            season_features = {
                'season_points_to_date': 320, 'season_podiums': 10,
                'season_avg_finish_last3': 3.1, 'season_quali_avg_last3': 4.2
            }
        elif driver_id == 1:  # Verstappen
            season_features = {
                'season_points_to_date': 290, 'season_podiums': 11,
                'season_avg_finish_last3': 3.8, 'season_quali_avg_last3': 2.9
            }
        elif driver_id == 63:  # Russell
            season_features = {
                'season_points_to_date': 180, 'season_podiums': 6,
                'season_avg_finish_last3': 4.7, 'season_quali_avg_last3': 4.5
            }
        else:
            season_features = {
                'season_points_to_date': 85, 'season_podiums': 2,
                'season_avg_finish_last3': 8.5, 'season_quali_avg_last3': 9.2
            }
        
        # Combine features
        driver_features = {
            **marina_features,
            **season_features,
            'grid_position': info['grid'],
            'safety_car_prob': 0.75,
            'is_marina_bay': 1
        }
        
        features_data.append(driver_features)
    
    # Create DataFrame and predict
    X = pd.DataFrame(features_data)[feature_cols].fillna(0)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create results
    results = []
    for i, (driver_id, info) in enumerate(drivers.items()):
        results.append({
            'driver': info['name'],
            'team': info['team'],
            'grid': info['grid'],
            'probability': probabilities[i] * 100
        })
    
    # Sort by probability
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    # Display results
    print("\n🏆 TOP 5 PREDICTIONS:")
    for i, result in enumerate(results[:5], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏁"
        print(f"{emoji} {result['driver']} ({result['team']})")
        print(f"   Grid: P{result['grid']} | Win Probability: {result['probability']:.1f}%")
    
    print(f"\n🎯 PREDICTED WINNER: {results[0]['driver']} ({results[0]['probability']:.1f}%)")
    print("💡 Prediction based on Marina Bay specialization + 2025 form")

def main():
    parser = argparse.ArgumentParser(description='Quick Singapore GP 2025 Prediction')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    quick_singapore_prediction()
    
    if args.detailed:
        print("\n📊 For detailed analysis, run:")
        print("   python3 singapore_gp_2025_final_prediction.py")

if __name__ == "__main__":
    main()