#!/usr/bin/env python3
"""
PHASE 10.1: Save Trained Models & Feature Pipeline
Production model serialization for deployment
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("🔄 PHASE 10.1: Model & Feature Pipeline Serialization")

# Import ensemble classes from separate module for pickling
from .model_classes import Stage1Ensemble, Stage2Ensemble

class ProductionModelSaver:
    """Save trained models and preprocessing pipeline for production deployment"""
    
    def __init__(self):
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path('data/features')
        self.reports_dir = Path('reports')
        
    def load_training_data(self):
        """Load and prepare training data"""
        
        print("📁 Loading training data...")
        
        # Load feature data
        features_file = self.data_dir / 'complete_features.parquet'
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        data = pd.read_parquet(features_file)
        print(f"   ✅ Loaded features: {data.shape}")
        
        # Prepare targets
        data['stage1_target'] = data['quali_best_time']  # Qualifying time
        data['stage2_target'] = data.get('is_race_winner_x', 0)  # Race winner
        
        # Clean data
        data['stage1_target'] = data['stage1_target'].fillna(data['stage1_target'].median())
        data['stage2_target'] = data['stage2_target'].fillna(0)
        
        print(f"   Stage-1 targets: {data['stage1_target'].notna().sum()}")
        print(f"   Stage-2 winners: {data['stage2_target'].sum()}")
        
        return data
    
    def prepare_feature_pipeline(self, data):
        """Create and fit feature preprocessing pipeline"""
        
        print("🔧 Creating feature preprocessing pipeline...")
        
        # Feature columns (exclude metadata and targets)
        exclude_cols = [
            'race_id', 'driver_id', 'team_id', 'date_utc', 'circuit_id', 
            'stage1_target', 'stage2_target', 'status', 'session_type',
            'is_pole_x', 'is_race_winner_x', 'is_pole_y', 'is_race_winner_y'
        ]
        
        feature_cols = [col for col in data.columns 
                       if col not in exclude_cols and data[col].dtype in ['int64', 'float64']]
        
        print(f"   Feature columns: {len(feature_cols)}")
        
        # Create preprocessing pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        
        # Numeric features preprocessing
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessing
        preprocessor = ColumnTransformer([
            ('numeric', numeric_transformer, feature_cols)
        ])
        
        # Fit on training data
        X_train = data[feature_cols]
        preprocessor.fit(X_train)
        
        # Save feature names for serving
        feature_metadata = {
            'feature_columns': feature_cols,
            'numeric_features': feature_cols,
            'target_columns': ['stage1_target', 'stage2_target'],
            'preprocessing_fitted': True
        }
        
        print(f"   ✅ Pipeline fitted on {len(X_train)} samples")
        return preprocessor, feature_cols, feature_metadata
    
    def train_stage1_model(self, data, preprocessor, feature_cols):
        """Train Stage-1 qualifying time prediction model"""
        
        print("🏁 Training Stage-1 Model (Qualifying Prediction)...")
        
        # Prepare data  
        X = data[feature_cols]
        y = data['stage1_target']
        
        # Remove missing targets
        valid_mask = y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        print(f"   Training samples: {len(X_valid)}")
        
        # Preprocess features
        X_processed = preprocessor.transform(X_valid)
        
        # Train ensemble of models
        stage1_models = []
        
        # Model 1: Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_processed, y_valid)
        stage1_models.append(('random_forest', rf_model))
        
        # Model 2: Extra Trees Regressor  
        from sklearn.ensemble import ExtraTreesRegressor
        et_model = ExtraTreesRegressor(
            n_estimators=80,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(X_processed, y_valid)
        stage1_models.append(('extra_trees', et_model))
        
        # Create ensemble wrapper
        stage1_ensemble = Stage1Ensemble(stage1_models)
        
        # Validate model
        train_pred = stage1_ensemble.predict(X_processed)
        mae = np.mean(np.abs(train_pred - y_valid))
        rmse = np.sqrt(np.mean((train_pred - y_valid) ** 2))
        
        print(f"   Training MAE: {mae:.3f} seconds")
        print(f"   Training RMSE: {rmse:.3f} seconds")
        print(f"   ✅ Stage-1 ensemble trained successfully")
        
        return stage1_ensemble
    
    def train_stage2_model(self, data, preprocessor, feature_cols, stage1_model):
        """Train Stage-2 race winner prediction model"""
        
        print("🏆 Training Stage-2 Model (Race Winner Prediction)...")
        
        # Prepare data
        X = data[feature_cols]
        y = data['stage2_target']
        
        # Add Stage-1 predictions as feature
        X_processed = preprocessor.transform(X)
        stage1_preds = stage1_model.predict(X_processed)
        
        # Combine features
        X_stage2 = np.column_stack([X_processed, stage1_preds])
        
        print(f"   Training samples: {len(X_stage2)}")
        print(f"   Feature dimensions: {X_stage2.shape[1]} (including Stage-1 predictions)")
        print(f"   Positive samples (winners): {y.sum()}")
        
        # Train ensemble of classifiers
        stage2_models = []
        
        # Model 1: Random Forest Classifier
        rf_classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_classifier.fit(X_stage2, y)
        stage2_models.append(('random_forest', rf_classifier))
        
        # Model 2: Extra Trees Classifier
        from sklearn.ensemble import ExtraTreesClassifier
        et_classifier = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        et_classifier.fit(X_stage2, y)
        stage2_models.append(('extra_trees', et_classifier))
        
        # Create ensemble wrapper
        stage2_ensemble = Stage2Ensemble(stage2_models)
        
        # Validate model
        train_pred = stage2_ensemble.predict(X_stage2)
        train_proba = stage2_ensemble.predict_proba(X_stage2)[:, 1]
        
        accuracy = np.mean(train_pred == y)
        
        # Winner-specific metrics
        winners_mask = y == 1
        if winners_mask.sum() > 0:
            winner_recall = np.mean(train_pred[winners_mask] == 1)
            winner_precision = np.sum((train_pred == 1) & (y == 1)) / max(np.sum(train_pred == 1), 1)
        else:
            winner_recall = winner_precision = 0
        
        print(f"   Training Accuracy: {accuracy:.3f}")
        print(f"   Winner Recall: {winner_recall:.3f}")
        print(f"   Winner Precision: {winner_precision:.3f}")
        print(f"   ✅ Stage-2 ensemble trained successfully")
        
        return stage2_ensemble
    
    def save_production_models(self, preprocessor, stage1_model, stage2_model, feature_metadata):
        """Save all models and pipeline for production"""
        
        print("💾 Saving production models...")
        
        # Save preprocessor
        preprocessor_file = self.models_dir / 'preprocessor.pkl'
        joblib.dump(preprocessor, preprocessor_file)
        print(f"   ✅ Preprocessor saved: {preprocessor_file}")
        
        # Save Stage-1 model (as ensemble for compatibility)
        stage1_file = self.models_dir / 'stage1_lgb_ensemble.pkl'
        joblib.dump(stage1_model, stage1_file)
        print(f"   ✅ Stage-1 model saved: {stage1_file}")
        
        # Save Stage-2 model
        stage2_file = self.models_dir / 'stage2_ensemble.pkl'
        joblib.dump(stage2_model, stage2_file)
        print(f"   ✅ Stage-2 model saved: {stage2_file}")
        
        # Save feature metadata
        metadata_file = self.models_dir / 'feature_metadata.pkl'
        joblib.dump(feature_metadata, metadata_file)
        print(f"   ✅ Feature metadata saved: {metadata_file}")
        
        # Create model registry
        model_registry = {
            'stage1_model_path': str(stage1_file),
            'stage2_model_path': str(stage2_file),
            'preprocessor_path': str(preprocessor_file),
            'metadata_path': str(metadata_file),
            'model_version': '1.0.0',
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_count': len(feature_metadata['feature_columns']),
            'model_type': 'two_stage_ensemble'
        }
        
        registry_file = self.models_dir / 'model_registry.pkl'
        joblib.dump(model_registry, registry_file)
        print(f"   ✅ Model registry saved: {registry_file}")
        
        return model_registry
    
    def validate_saved_models(self, model_registry):
        """Validate that saved models can be loaded and used"""
        
        print("🔍 Validating saved models...")
        
        try:
            # Load all components
            preprocessor = joblib.load(model_registry['preprocessor_path'])
            stage1_model = joblib.load(model_registry['stage1_model_path'])
            stage2_model = joblib.load(model_registry['stage2_model_path'])
            metadata = joblib.load(model_registry['metadata_path'])
            
            print(f"   ✅ All models loaded successfully")
            print(f"   ✅ Feature columns: {len(metadata['feature_columns'])}")
            print(f"   ✅ Model version: {model_registry['model_version']}")
            
            # Create test sample
            feature_cols = metadata['feature_columns']
            test_sample = pd.DataFrame({
                col: [np.random.randn()] for col in feature_cols
            })
            
            # Test preprocessing
            X_processed = preprocessor.transform(test_sample)
            print(f"   ✅ Preprocessing works: {X_processed.shape}")
            
            # Test Stage-1 prediction
            stage1_pred = stage1_model.predict(X_processed)
            print(f"   ✅ Stage-1 prediction: {stage1_pred[0]:.3f} seconds")
            
            # Test Stage-2 prediction (with Stage-1 feature)
            X_stage2 = np.column_stack([X_processed, stage1_pred])
            stage2_proba = stage2_model.predict_proba(X_stage2)
            print(f"   ✅ Stage-2 prediction: {stage2_proba[0, 1]:.3f} win probability")
            
            print(f"   ✅ Model validation successful!")
            return True
            
        except Exception as e:
            print(f"   ❌ Model validation failed: {e}")
            return False

def save_production_models():
    """Main function to train and save production models"""
    
    print("=" * 80)
    print("PHASE 10.1: PRODUCTION MODEL SERIALIZATION")
    print("=" * 80)
    
    saver = ProductionModelSaver()
    
    try:
        # Load training data
        data = saver.load_training_data()
        
        # Create preprocessing pipeline
        preprocessor, feature_cols, feature_metadata = saver.prepare_feature_pipeline(data)
        
        # Train Stage-1 model
        stage1_model = saver.train_stage1_model(data, preprocessor, feature_cols)
        
        # Train Stage-2 model
        stage2_model = saver.train_stage2_model(data, preprocessor, feature_cols, stage1_model)
        
        # Save all models
        model_registry = saver.save_production_models(
            preprocessor, stage1_model, stage2_model, feature_metadata
        )
        
        # Validate saved models
        validation_success = saver.validate_saved_models(model_registry)
        
        if validation_success:
            print(f"\n✅ PHASE 10.1 COMPLETE!")
            print(f"📦 Models saved to: {saver.models_dir}")
            print(f"🔧 Preprocessor: models/preprocessor.pkl")
            print(f"🏁 Stage-1 Model: models/stage1_lgb_ensemble.pkl")
            print(f"🏆 Stage-2 Model: models/stage2_ensemble.pkl")
            print(f"📋 Model Registry: models/model_registry.pkl")
            return True
        else:
            print(f"\n❌ Model validation failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Model saving failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = save_production_models()
    
    if success:
        print(f"\n🚀 PRODUCTION MODELS READY FOR DEPLOYMENT!")
        print(f"   ✅ Two-stage ensemble models trained and saved")
        print(f"   ✅ Feature preprocessing pipeline serialized")
        print(f"   ✅ Model validation successful")
        print(f"\n📋 Ready for Phase 10.2: FastAPI Server Implementation")
    else:
        print(f"\n❌ Production model preparation failed")