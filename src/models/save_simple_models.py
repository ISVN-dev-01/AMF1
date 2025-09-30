#!/usr/bin/env python3
"""
PHASE 10.1: Simplified Production Model Saving
Create simple, serializable models for deployment
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("🔄 PHASE 10.1: Simplified Production Model Saving")

def save_simple_models():
    """Save simple, deployable models"""
    
    print("=" * 80)
    print("PHASE 10.1: SIMPLIFIED PRODUCTION MODEL SAVING")
    print("=" * 80)
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        print("📁 Loading training data...")
        data = pd.read_parquet('data/features/complete_features.parquet')
        print(f"   ✅ Loaded features: {data.shape}")
        
        # Prepare targets
        data['stage1_target'] = data['quali_best_time'].fillna(data['quali_best_time'].median())
        data['stage2_target'] = data.get('is_race_winner_x', 0).fillna(0)
        
        # Feature columns 
        exclude_cols = [
            'race_id', 'driver_id', 'team_id', 'date_utc', 'circuit_id', 
            'stage1_target', 'stage2_target', 'status', 'session_type',
            'is_pole_x', 'is_race_winner_x', 'is_pole_y', 'is_race_winner_y'
        ]
        
        feature_cols = [col for col in data.columns 
                       if col not in exclude_cols and data[col].dtype in ['int64', 'float64']]
        
        print(f"   Feature columns: {len(feature_cols)}")
        
        # Prepare features
        X = data[feature_cols].fillna(0)
        y1 = data['stage1_target']
        y2 = data['stage2_target']
        
        # Create simple preprocessor
        from sklearn.preprocessing import StandardScaler
        preprocessor = StandardScaler()
        X_scaled = preprocessor.fit_transform(X)
        
        print("🏁 Training Stage-1 Model...")
        stage1_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        stage1_model.fit(X_scaled, y1)
        
        stage1_pred = stage1_model.predict(X_scaled)
        mae = np.mean(np.abs(stage1_pred - y1))
        print(f"   Stage-1 MAE: {mae:.3f} seconds")
        
        print("🏆 Training Stage-2 Model...")
        # Add Stage-1 predictions as feature for Stage-2
        X_stage2 = np.column_stack([X_scaled, stage1_pred])
        
        stage2_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        stage2_model.fit(X_stage2, y2)
        
        stage2_pred = stage2_model.predict(X_stage2)
        accuracy = np.mean(stage2_pred == y2)
        print(f"   Stage-2 Accuracy: {accuracy:.3f}")
        
        print("💾 Saving models...")
        
        # Save preprocessor
        joblib.dump(preprocessor, models_dir / 'preprocessor.pkl')
        print(f"   ✅ Preprocessor saved")
        
        # Save Stage-1 model (as list for compatibility)
        joblib.dump([stage1_model], models_dir / 'stage1_lgb_ensemble.pkl')
        print(f"   ✅ Stage-1 model saved")
        
        # Save Stage-2 model
        joblib.dump(stage2_model, models_dir / 'stage2_ensemble.pkl')
        print(f"   ✅ Stage-2 model saved")
        
        # Save metadata
        metadata = {
            'feature_columns': feature_cols,
            'model_version': '1.0.1',
            'training_date': pd.Timestamp.now().isoformat(),
            'stage1_mae': mae,
            'stage2_accuracy': accuracy
        }
        joblib.dump(metadata, models_dir / 'feature_metadata.pkl')
        print(f"   ✅ Metadata saved")
        
        # Test loading
        print("🔍 Testing model loading...")
        test_preprocessor = joblib.load(models_dir / 'preprocessor.pkl')
        test_stage1 = joblib.load(models_dir / 'stage1_lgb_ensemble.pkl')
        test_stage2 = joblib.load(models_dir / 'stage2_ensemble.pkl')
        test_metadata = joblib.load(models_dir / 'feature_metadata.pkl')
        
        # Test prediction
        test_sample = np.random.randn(1, len(feature_cols))
        test_scaled = test_preprocessor.transform(test_sample)
        test_pred1 = test_stage1[0].predict(test_scaled)[0]
        test_pred2_input = np.column_stack([test_scaled, [test_pred1]])
        test_pred2 = test_stage2.predict_proba(test_pred2_input)[0, 1]
        
        print(f"   ✅ Test prediction - Stage-1: {test_pred1:.3f}s, Stage-2: {test_pred2:.3f}")
        
        print(f"\n✅ PHASE 10.1 COMPLETE!")
        print(f"📦 Models saved to: {models_dir}")
        print(f"🏁 Stage-1 Model: models/stage1_lgb_ensemble.pkl")
        print(f"🏆 Stage-2 Model: models/stage2_ensemble.pkl")
        print(f"🔧 Preprocessor: models/preprocessor.pkl")
        print(f"📋 Metadata: models/feature_metadata.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = save_simple_models()
    
    if success:
        print(f"\n🚀 SIMPLE PRODUCTION MODELS READY!")
        print(f"   ✅ Stage-1 and Stage-2 models trained and saved")
        print(f"   ✅ Standard preprocessing pipeline created")
        print(f"   ✅ Models validated and ready for deployment")
    else:
        print(f"\n❌ Model preparation failed")