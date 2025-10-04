#!/usr/bin/env python3
"""
Stage-2 Marina Bay Race Winner Prediction Pipeline (Simplified)
Uses RandomForest instead of LightGBM for better compatibility
Trains cutoff-aware models using 2020-2024 + 2025 up to qualifying
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timezone
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# Configuration
SEASONS_HISTORY = [2020, 2021, 2022, 2023, 2024]
CURRENT_SEASON = 2025
CUTOFF = pd.Timestamp("2025-10-04 23:59:59", tz="UTC")  # End of Singapore qualifying

# Marina Bay circuit ID
MARINA_BAY_CIRCUIT = "marina_bay"
SINGAPORE_RACE_ID = "singapore_2025"

class Stage2ModelTrainerSimplified:
    def __init__(self, data_path="data/processed/master_dataset.parquet"):
        self.data_path = data_path
        self.df = None
        self.feature_cols = []
        self.model = None
        self.calibrator = None
        self.scaler = StandardScaler()
        
    def load_and_filter_data(self):
        """Load data and filter to relevant seasons with cutoff awareness"""
        print("🔄 Loading and filtering data...")
        
        try:
            self.df = pd.read_parquet(self.data_path)
        except FileNotFoundError:
            print("❌ Master dataset not found. Creating mock data for demonstration...")
            self.df = self._create_mock_dataset()
        
        # Ensure race_date is timezone-aware
        if not pd.api.types.is_datetime64_any_dtype(self.df['race_date']):
            self.df['race_date'] = pd.to_datetime(self.df['race_date'])
        
        if self.df['race_date'].dt.tz is None:
            self.df['race_date'] = self.df['race_date'].dt.tz_localize('UTC')
        
        # Filter historical seasons
        df_hist = self.df[self.df['season'].isin(SEASONS_HISTORY)].copy()
        
        # Filter current season up to cutoff
        df_current = self.df[
            (self.df['season'] == CURRENT_SEASON) & 
            (self.df['race_date'] <= CUTOFF)
        ].copy()
        
        # Combine datasets
        self.df = pd.concat([df_hist, df_current], ignore_index=True)
        
        print(f"✅ Loaded {len(self.df)} records from {SEASONS_HISTORY + [CURRENT_SEASON]}")
        print(f"📅 Data cutoff: {CUTOFF}")
        
    def _create_mock_dataset(self):
        """Create comprehensive mock F1 dataset for demonstration"""
        np.random.seed(42)
        
        # Real F1 driver numbers and teams
        drivers = [1, 11, 16, 55, 44, 63, 4, 81, 14, 18, 22, 23, 31, 3, 24, 27, 10, 77, 2, 20]
        teams = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
        
        circuits = [
            'marina_bay', 'monza', 'silverstone', 'spa', 'monaco', 'interlagos',
            'suzuka', 'austin', 'bahrain', 'melbourne', 'imola', 'spain',
            'canada', 'austria', 'hungary', 'netherlands', 'baku', 'miami',
            'las_vegas', 'abu_dhabi', 'jeddah', 'qatar'
        ]
        
        data = []
        race_id_counter = 0
        
        for season in SEASONS_HISTORY + [CURRENT_SEASON]:
            num_races = 20 if season < CURRENT_SEASON else 16  # 2025 partial season
            
            for race_num in range(num_races):
                circuit = circuits[race_num % len(circuits)]
                race_date = pd.Timestamp(f"{season}-{3 + race_num//2:02d}-{15 + (race_num % 2) * 14:02d}", tz="UTC")
                
                # Skip future races for 2025
                if season == CURRENT_SEASON and race_date > CUTOFF:
                    continue
                
                race_id_counter += 1
                race_id = f"race_{season}_{race_num:02d}"
                
                # Create realistic driver performance with some randomness
                # But maintain some consistency for Marina Bay specifically
                performance_order = list(range(len(drivers)))
                
                # Marina Bay specific adjustments (some drivers historically better)
                if circuit == MARINA_BAY_CIRCUIT:
                    # Hamilton, Russell, Verstappen historically good at Singapore
                    marina_bonus = {44: -2, 63: -1, 1: -1, 4: -0.5}  # Lower = better
                    for i, driver in enumerate(drivers):
                        if driver in marina_bonus:
                            performance_order[i] += marina_bonus[driver]
                
                # Add randomness
                performance_order = [p + np.random.normal(0, 1.5) for p in performance_order]
                
                # Sort by performance (lower = better finish)
                driver_performance = list(zip(drivers, teams, performance_order))
                driver_performance.sort(key=lambda x: x[2])
                
                for pos, (driver_id, team_id, _) in enumerate(driver_performance):
                    finish_pos = pos + 1
                    
                    # DNF probability
                    dnf_prob = 0.08 if circuit == MARINA_BAY_CIRCUIT else 0.12
                    dnf = 1 if np.random.random() < dnf_prob else 0
                    
                    if dnf:
                        race_position = None
                        points = 0
                    else:
                        race_position = finish_pos
                        # F1 points system
                        point_values = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10
                        points = point_values[min(pos, len(point_values) - 1)]
                    
                    is_winner = 1 if race_position == 1 else 0
                    podium = 1 if race_position and race_position <= 3 else 0
                    
                    data.append({
                        'season': season,
                        'race_id': race_id,
                        'race_date': race_date,
                        'session_type': 'R',
                        'driver_id': driver_id,
                        'team_id': team_id,
                        'circuit_id': circuit,
                        'lap_time_sec': 90 + np.random.normal(0, 2),
                        'quali_rank': pos + 1,  # Simplified: quali = race order
                        'race_position': race_position,
                        'points': points,
                        'podium': podium,
                        'dnf': dnf,
                        'is_race_winner': is_winner
                    })
        
        print(f"🏗️ Generated {len(data)} mock race records")
        return pd.DataFrame(data)
    
    def get_past_events(self, target_date):
        """Get events before target date for cutoff-aware features"""
        return self.df[self.df['race_date'] < target_date]
    
    def compute_marina_features(self, target_race_date, driver_list):
        """Compute Marina Bay track-specific features with recency weighting"""
        print("🏁 Computing Marina Bay track features...")
        
        past = self.get_past_events(target_race_date)
        marina_past = past[
            (past['circuit_id'] == MARINA_BAY_CIRCUIT) & 
            (past['session_type'] == 'R')
        ].copy()
        
        features = []
        
        for driver_id in driver_list:
            driver_data = marina_past[
                marina_past['driver_id'] == driver_id
            ].sort_values('race_date', ascending=False)
            
            count = len(driver_data)
            wins = (driver_data['race_position'] == 1).sum() if count > 0 else 0
            podiums = (driver_data['race_position'] <= 3).sum() if count > 0 else 0
            
            # Average finish position
            avg_finish = driver_data['race_position'].mean() if count > 0 else 15.0  # Default mid-field
            
            # Recency-weighted finish (exponential decay)
            decay_factor = 0.5
            if count > 0:
                weights = np.exp(-decay_factor * np.arange(len(driver_data)))
                recency_finish = np.average(driver_data['race_position'], weights=weights)
            else:
                recency_finish = 15.0
            
            # Best finish in last 3 appearances
            best_last3 = driver_data['race_position'].head(3).min() if count > 0 else 20.0
            
            features.append({
                'driver_id': driver_id,
                'marina_count': count,
                'marina_wins': int(wins),
                'marina_podiums': int(podiums),
                'marina_avg_finish': avg_finish,
                'marina_recency_finish': recency_finish,
                'marina_best_last3': best_last3
            })
        
        return pd.DataFrame(features)
    
    def compute_season_form(self, season, cutoff_date, driver_list):
        """Compute current season form features"""
        print(f"📊 Computing season {season} form features...")
        
        season_data = self.df[
            (self.df['season'] == season) & 
            (self.df['race_date'] < cutoff_date) &
            (self.df['session_type'] == 'R')
        ]
        
        features = []
        
        for driver_id in driver_list:
            driver_races = season_data[
                season_data['driver_id'] == driver_id
            ].sort_values('race_date', ascending=False)
            
            points_total = driver_races['points'].sum()
            podiums_count = (driver_races['race_position'] <= 3).sum()
            
            # Last 3 races average finish
            last3_finish = driver_races['race_position'].head(3).mean() if len(driver_races) > 0 else 15.0
            
            # Last 3 qualifying average
            last3_quali = driver_races['quali_rank'].head(3).mean() if len(driver_races) > 0 else 15.0
            
            features.append({
                'driver_id': driver_id,
                'season_points_to_date': points_total,
                'season_podiums': int(podiums_count),
                'season_avg_finish_last3': last3_finish,
                'season_quali_avg_last3': last3_quali
            })
        
        return pd.DataFrame(features)
    
    def build_stage2_dataset(self):
        """Build complete Stage-2 training dataset with cutoff-aware features"""
        print("🏗️  Building Stage-2 dataset...")
        
        rows = []
        
        # Get unique races sorted by date
        races = self.df[self.df['session_type'] == 'R'].groupby('race_id').agg({
            'race_date': 'first',
            'season': 'first',
            'circuit_id': 'first'
        }).reset_index().sort_values('race_date')
        
        print(f"Processing {len(races)} races...")
        
        for idx, (_, race) in enumerate(races.iterrows()):
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(races)} races...")
            
            race_id = race['race_id']
            race_date = race['race_date']
            season = race['season']
            circuit_id = race['circuit_id']
            
            # Get drivers in this race
            race_drivers = self.df[
                (self.df['race_id'] == race_id) & 
                (self.df['session_type'] == 'R')
            ]['driver_id'].unique()
            
            # Compute Marina Bay features
            marina_features = self.compute_marina_features(race_date, race_drivers)
            
            # Compute season form features
            season_form = self.compute_season_form(season, race_date, race_drivers)
            
            # Get race results for labels
            race_results = self.df[
                (self.df['race_id'] == race_id) & 
                (self.df['session_type'] == 'R')
            ][['driver_id', 'race_position', 'quali_rank', 'points', 'dnf']]
            
            # Combine features
            race_features = marina_features.merge(season_form, on='driver_id', how='left')
            race_features = race_features.merge(race_results, on='driver_id', how='left')
            
            # Add race metadata
            race_features['race_id'] = race_id
            race_features['race_date'] = race_date
            race_features['season'] = season
            race_features['circuit_id'] = circuit_id
            race_features['is_marina_bay'] = int(circuit_id == MARINA_BAY_CIRCUIT)
            
            # Add grid position (qualifying rank)
            race_features['grid_position'] = race_features['quali_rank']
            
            # Add safety car probability (track-specific)
            safety_car_probs = {
                'marina_bay': 0.75,
                'monaco': 0.70,
                'baku': 0.65,
                'jeddah': 0.60,
                'miami': 0.45,
                'default': 0.30
            }
            race_features['safety_car_prob'] = safety_car_probs.get(circuit_id, safety_car_probs['default'])
            
            # Create target variable
            race_features['is_race_winner'] = (race_features['race_position'] == 1).astype(int)
            
            rows.append(race_features)
        
        # Combine all races
        dataset = pd.concat(rows, ignore_index=True)
        
        # Remove rows with missing critical data
        dataset = dataset.dropna(subset=['race_position', 'quali_rank'])
        
        print(f"✅ Built dataset with {len(dataset)} driver-race records")
        print(f"📊 Winners: {dataset['is_race_winner'].sum()}, Non-winners: {(dataset['is_race_winner'] == 0).sum()}")
        
        return dataset
    
    def train_model(self, dataset):
        """Train Stage-2 RandomForest classifier with GroupKFold"""
        print("🤖 Training Stage-2 race winner model...")
        
        # Define feature columns
        self.feature_cols = [
            'marina_count', 'marina_wins', 'marina_podiums', 'marina_avg_finish',
            'marina_recency_finish', 'marina_best_last3',
            'season_points_to_date', 'season_podiums', 'season_avg_finish_last3',
            'season_quali_avg_last3', 'grid_position', 'safety_car_prob', 'is_marina_bay'
        ]
        
        # Prepare data
        X = dataset[self.feature_cols].copy()
        y = dataset['is_race_winner'].copy()
        groups = dataset['race_id'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Group K-Fold to prevent race leakage
        gkf = GroupKFold(n_splits=5)
        
        oof_preds = np.zeros(len(X))
        models = []
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            print(f"  Training fold {fold + 1}/5...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # RandomForest parameters optimized for imbalanced data
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict validation set
            val_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds
            
            # Calculate fold metrics
            val_logloss = log_loss(y_val, val_preds)
            val_brier = brier_score_loss(y_val, val_preds)
            
            # Top-1 accuracy per race
            val_df = dataset.iloc[val_idx][['race_id', 'driver_id']].copy()
            val_df['y_true'] = y_val.values
            val_df['y_pred'] = val_preds
            
            race_accs = []
            for race_id, race_group in val_df.groupby('race_id'):
                if race_group['y_true'].sum() > 0:  # Only races with winners
                    pred_winner = race_group.loc[race_group['y_pred'].idxmax(), 'driver_id']
                    true_winner = race_group.loc[race_group['y_true'].idxmax(), 'driver_id']
                    race_accs.append(int(pred_winner == true_winner))
            
            top1_acc = np.mean(race_accs) if race_accs else 0
            
            fold_scores.append({
                'fold': fold + 1,
                'logloss': val_logloss,
                'brier': val_brier,
                'top1_acc': top1_acc
            })
            
            print(f"    Fold {fold + 1} - LogLoss: {val_logloss:.4f}, Brier: {val_brier:.4f}, Top-1: {top1_acc:.3f}")
            
            models.append(model)
        
        # Overall metrics
        overall_logloss = log_loss(y, oof_preds)
        overall_brier = brier_score_loss(y, oof_preds)
        
        # Overall top-1 accuracy
        oof_df = dataset[['race_id', 'driver_id']].copy()
        oof_df['y_true'] = y.values
        oof_df['y_pred'] = oof_preds
        
        overall_race_accs = []
        for race_id, race_group in oof_df.groupby('race_id'):
            if race_group['y_true'].sum() > 0:
                pred_winner = race_group.loc[race_group['y_pred'].idxmax(), 'driver_id']
                true_winner = race_group.loc[race_group['y_true'].idxmax(), 'driver_id']
                overall_race_accs.append(int(pred_winner == true_winner))
        
        overall_top1 = np.mean(overall_race_accs) if overall_race_accs else 0
        
        print(f"\n✅ Training completed!")
        print(f"📊 Overall Metrics:")
        print(f"   LogLoss: {overall_logloss:.4f}")
        print(f"   Brier Score: {overall_brier:.4f}")
        print(f"   Top-1 Accuracy: {overall_top1:.3f}")
        
        # Store models
        self.models = models
        self.model = models[0]  # Use first model for simplicity
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top Feature Importance:")
        for _, row in importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return fold_scores
    
    def predict_singapore(self):
        """Generate predictions for Singapore GP 2025"""
        print("🏎️ Predicting Singapore GP 2025...")
        
        # Singapore GP driver list (using actual 2025 entries)
        singapore_drivers = [1, 11, 16, 55, 44, 63, 4, 81, 14, 18]
        
        # Compute features for Singapore race
        marina_features = self.compute_marina_features(CUTOFF, singapore_drivers)
        season_form = self.compute_season_form(CURRENT_SEASON, CUTOFF, singapore_drivers)
        
        # Combine features
        singapore_features = marina_features.merge(season_form, on='driver_id', how='left')
        
        # Add qualifying results (George Russell pole position)
        quali_results = {
            63: 1,  # George Russell - Pole
            4: 2,   # Lando Norris - P2
            1: 3,   # Max Verstappen - P3
            81: 4,  # Oscar Piastri - P4
            16: 5,  # Charles Leclerc - P5
            55: 6,  # Carlos Sainz - P6
            44: 7,  # Lewis Hamilton - P7
            14: 8,  # Fernando Alonso - P8
            11: 9,  # Sergio Pérez - P9
            18: 10  # Lance Stroll - P10
        }
        
        singapore_features['grid_position'] = singapore_features['driver_id'].map(quali_results)
        singapore_features['is_marina_bay'] = 1
        singapore_features['safety_car_prob'] = 0.75  # High for Singapore
        
        # Fill missing values with median
        for col in self.feature_cols:
            if col in singapore_features.columns:
                singapore_features[col] = singapore_features[col].fillna(singapore_features[col].median())
        
        # Make predictions
        X_singapore = singapore_features[self.feature_cols]
        probs = self.model.predict_proba(X_singapore)[:, 1]
        
        # Create results dataframe
        results = singapore_features[['driver_id']].copy()
        results['win_probability'] = probs
        results = results.sort_values('win_probability', ascending=False)
        
        # Add driver names
        driver_names = {
            1: "Max Verstappen",
            11: "Sergio Pérez", 
            16: "Charles Leclerc",
            55: "Carlos Sainz",
            44: "Lewis Hamilton",
            63: "George Russell",
            4: "Lando Norris",
            81: "Oscar Piastri",
            14: "Fernando Alonso",
            18: "Lance Stroll"
        }
        
        results['driver_name'] = results['driver_id'].map(driver_names)
        results['grid_position'] = results['driver_id'].map(quali_results)
        
        return results
    
    def save_model(self, model_path="models/production/stage2_marina_rf.pkl"):
        """Save trained model and components"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'model_type': 'RandomForest'
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Stage-2 Marina Bay Race Winner Model (Simplified)')
    parser.add_argument('--build-dataset', action='store_true', help='Build training dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict Singapore GP')
    parser.add_argument('--model-out', default='models/production/stage2_marina_rf.pkl')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Stage2ModelTrainerSimplified()
    
    # Load and filter data
    trainer.load_and_filter_data()
    
    if args.build_dataset or args.train:
        # Build dataset
        dataset = trainer.build_stage2_dataset()
        
        if args.train:
            # Train model
            fold_scores = trainer.train_model(dataset)
            
            # Save model
            trainer.save_model(args.model_out)
    
    if args.predict:
        # Load model if not training
        if not args.train:
            print("Loading saved model...")
            model_data = joblib.load(args.model_out)
            trainer.model = model_data['model']
            trainer.feature_cols = model_data['feature_cols']
        
        # Predict Singapore GP
        results = trainer.predict_singapore()
        
        print(f"\n🏆 SINGAPORE GP 2025 PREDICTIONS:")
        print("=" * 50)
        for i, (_, row) in enumerate(results.head(5).iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏁"
            print(f"{emoji} {i}. {row['driver_name']} (P{row['grid_position']}) - {row['win_probability']:.1%}")

if __name__ == "__main__":
    main()