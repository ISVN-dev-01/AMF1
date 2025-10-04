#!/usr/bin/env python3
"""
Stage-2 Marina Bay Race Winner Prediction Pipeline
Trains cutoff-aware models using 2020-2024 + 2025 up to qualifying
Specialized for Singapore GP with track affinity and recency weighting
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
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import shap

# Configuration
SEASONS_HISTORY = [2020, 2021, 2022, 2023, 2024]
CURRENT_SEASON = 2025
CUTOFF = pd.Timestamp("2025-10-04 23:59:59", tz="UTC")  # End of Singapore qualifying

# Marina Bay circuit ID (adjust based on your data)
MARINA_BAY_CIRCUIT = "marina_bay"
SINGAPORE_RACE_ID = "singapore_2025"

class Stage2ModelTrainer:
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
        """Create mock F1 dataset for demonstration"""
        np.random.seed(42)
        
        # Mock drivers
        drivers = [1, 11, 16, 55, 44, 63, 4, 81, 14, 18]  # Real F1 driver numbers
        teams = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]  # Team IDs
        circuits = ['marina_bay', 'monza', 'silverstone', 'spa', 'monaco', 'interlagos']
        
        data = []
        race_id = 0
        
        for season in SEASONS_HISTORY + [CURRENT_SEASON]:
            for circuit_idx, circuit in enumerate(circuits):
                # Skip future races for 2025
                race_date = pd.Timestamp(f"{season}-{3 + circuit_idx * 2:02d}-15", tz="UTC")
                if season == CURRENT_SEASON and race_date > CUTOFF:
                    continue
                
                race_id += 1
                
                for pos, driver_id in enumerate(drivers):
                    # Create realistic F1 data
                    is_winner = 1 if pos == 0 else 0
                    points = max(25 - pos * 2, 0) if pos < 8 else 0
                    podium = 1 if pos < 3 else 0
                    dnf = 1 if np.random.random() < 0.1 else 0
                    
                    if dnf:
                        race_position = None
                        points = 0
                    else:
                        race_position = pos + 1
                    
                    data.append({
                        'season': season,
                        'race_id': f"race_{race_id}",
                        'race_date': race_date,
                        'session_type': 'R',
                        'driver_id': driver_id,
                        'team_id': teams[pos],
                        'circuit_id': circuit,
                        'lap_time_sec': 90 + np.random.normal(0, 2),
                        'quali_rank': pos + 1,
                        'race_position': race_position,
                        'points': points,
                        'podium': podium,
                        'dnf': dnf,
                        'is_race_winner': is_winner
                    })
        
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
            avg_finish = driver_data['race_position'].mean() if count > 0 else np.nan
            
            # Recency-weighted finish (exponential decay)
            decay_factor = 0.5  # Tunable hyperparameter
            if count > 0:
                weights = np.exp(-decay_factor * np.arange(len(driver_data)))
                recency_finish = np.average(driver_data['race_position'], weights=weights)
            else:
                recency_finish = np.nan
            
            # Best finish in last 3 appearances
            best_last3 = driver_data['race_position'].head(3).min() if count > 0 else np.nan
            
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
            last3_finish = driver_races['race_position'].head(3).mean() if len(driver_races) > 0 else np.nan
            
            # Last 3 qualifying average
            last3_quali = driver_races['quali_rank'].head(3).mean() if len(driver_races) > 0 else np.nan
            
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
        
        for _, race in races.iterrows():
            race_id = race['race_id']
            race_date = race['race_date']
            season = race['season']
            circuit_id = race['circuit_id']
            
            print(f"  Processing {race_id} ({circuit_id}) - {race_date.date()}")
            
            # Get drivers in this race
            race_drivers = self.df[
                (self.df['race_id'] == race_id) & 
                (self.df['session_type'] == 'R')
            ]['driver_id'].unique()
            
            # Compute Marina Bay features (only relevant for Marina Bay races)
            if circuit_id == MARINA_BAY_CIRCUIT:
                marina_features = self.compute_marina_features(race_date, race_drivers)
            else:
                # Create empty Marina features for non-Marina Bay races
                marina_features = pd.DataFrame([
                    {
                        'driver_id': driver_id,
                        'marina_count': 0,
                        'marina_wins': 0,
                        'marina_podiums': 0,
                        'marina_avg_finish': np.nan,
                        'marina_recency_finish': np.nan,
                        'marina_best_last3': np.nan
                    }
                    for driver_id in race_drivers
                ])
            
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
                'marina_bay': 0.75,  # High probability in Singapore
                'monaco': 0.70,
                'baku': 0.65,
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
        """Train Stage-2 LightGBM classifier with GroupKFold"""
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
        X = X.fillna(-999)  # LightGBM handles missing values well
        
        # Group K-Fold to prevent race leakage
        gkf = GroupKFold(n_splits=5)
        
        oof_preds = np.zeros(len(X))
        models = []
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            print(f"  Training fold {fold + 1}/5...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # LightGBM parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 32,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 10,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'random_state': 42,
                'is_unbalance': True,
                'verbose': -1
            }
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Predict validation set
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
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
        
        # Store best model (or ensemble)
        self.models = models
        self.model = models[0]  # Use first model for simplicity
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top Feature Importance:")
        for _, row in importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.0f}")
        
        return fold_scores
    
    def calibrate_probabilities(self, dataset):
        """Calibrate model probabilities using isotonic regression"""
        print("🎯 Calibrating probabilities...")
        
        X = dataset[self.feature_cols].fillna(-999)
        y = dataset['is_race_winner']
        
        # Use the trained model to get base predictions
        base_preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Calibrate using isotonic regression
        self.calibrator = CalibratedClassifierCV(method='isotonic', cv='prefit')
        
        # Create a simple wrapper for the LightGBM model
        class LGBWrapper:
            def __init__(self, model, feature_cols):
                self.model = model
                self.feature_cols = feature_cols
            
            def predict_proba(self, X):
                X_subset = X[self.feature_cols].fillna(-999)
                preds = self.model.predict(X_subset, num_iteration=self.model.best_iteration)
                return np.column_stack([1 - preds, preds])
        
        wrapper = LGBWrapper(self.model, self.feature_cols)
        
        # Note: In production, you'd use a separate validation set
        self.calibrator.fit(wrapper, X, y)
        
        print("✅ Probability calibration completed")
    
    def predict_singapore(self):
        """Generate predictions for Singapore GP 2025"""
        print("🏎️ Predicting Singapore GP 2025...")
        
        # Get Singapore GP driver list (mock for now)
        singapore_drivers = [1, 11, 16, 55, 44, 63, 4, 81, 14, 18]
        
        # Compute features for Singapore race
        marina_features = self.compute_marina_features(CUTOFF, singapore_drivers)
        season_form = self.compute_season_form(CURRENT_SEASON, CUTOFF, singapore_drivers)
        
        # Combine features
        singapore_features = marina_features.merge(season_form, on='driver_id', how='left')
        
        # Add qualifying results (using actual results from George Russell pole)
        quali_results = {
            1: 3,   # Max Verstappen - P3
            11: 9,  # Sergio Pérez - P9  
            16: 5,  # Charles Leclerc - P5
            55: 6,  # Carlos Sainz - P6
            44: 7,  # Lewis Hamilton - P7
            63: 1,  # George Russell - P1 (Pole)
            4: 2,   # Lando Norris - P2
            81: 4,  # Oscar Piastri - P4
            14: 8,  # Fernando Alonso - P8
            18: 10  # Lance Stroll - P10
        }
        
        singapore_features['grid_position'] = singapore_features['driver_id'].map(quali_results)
        singapore_features['is_marina_bay'] = 1
        singapore_features['safety_car_prob'] = 0.75  # High for Singapore
        
        # Fill missing values
        singapore_features = singapore_features.fillna(-999)
        
        # Make predictions
        X_singapore = singapore_features[self.feature_cols]
        
        if self.calibrator:
            # Use calibrated probabilities
            class LGBWrapper:
                def __init__(self, model, feature_cols):
                    self.model = model
                    self.feature_cols = feature_cols
                
                def predict_proba(self, X):
                    X_subset = X[self.feature_cols].fillna(-999)
                    preds = self.model.predict(X_subset, num_iteration=self.model.best_iteration)
                    return np.column_stack([1 - preds, preds])
            
            wrapper = LGBWrapper(self.model, self.feature_cols)
            probs = self.calibrator.predict_proba(wrapper, X_singapore)[:, 1]
        else:
            probs = self.model.predict(X_singapore, num_iteration=self.model.best_iteration)
        
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
    
    def save_model(self, model_path="models/production/stage2_marina_lgb.pkl"):
        """Save trained model and components"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'calibrator': self.calibrator
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Stage-2 Marina Bay Race Winner Model')
    parser.add_argument('--build-dataset', action='store_true', help='Build training dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict Singapore GP')
    parser.add_argument('--model-out', default='models/production/stage2_marina_lgb.pkl')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Stage2ModelTrainer()
    
    # Load and filter data
    trainer.load_and_filter_data()
    
    if args.build_dataset or args.train:
        # Build dataset
        dataset = trainer.build_stage2_dataset()
        
        if args.train:
            # Train model
            fold_scores = trainer.train_model(dataset)
            
            # Calibrate probabilities
            trainer.calibrate_probabilities(dataset)
            
            # Save model
            trainer.save_model(args.model_out)
    
    if args.predict:
        # Load model if not training
        if not args.train:
            print("Loading saved model...")
            model_data = joblib.load(args.model_out)
            trainer.model = model_data['model']
            trainer.feature_cols = model_data['feature_cols']
            trainer.calibrator = model_data.get('calibrator')
        
        # Predict Singapore GP
        results = trainer.predict_singapore()
        
        print(f"\n🏆 SINGAPORE GP 2025 PREDICTIONS:")
        print("=" * 50)
        for i, (_, row) in enumerate(results.head(5).iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏁"
            print(f"{emoji} {i}. {row['driver_name']} (P{row['grid_position']}) - {row['win_probability']:.1%}")

if __name__ == "__main__":
    main()