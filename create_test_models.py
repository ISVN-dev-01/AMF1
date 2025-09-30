#!/usr/bin/env python3
"""
Create minimal test models for CI/CD pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_test_models():
    """Create minimal test models for CI pipeline"""
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    print("Creating test models for CI pipeline...")
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Generate sample features
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Stage 1: Qualifying time prediction (regression)
    y_stage1 = np.random.uniform(60, 90, n_samples)  # lap times in seconds
    
    # Stage 2: Race winner prediction (classification) 
    y_stage2 = np.random.randint(0, 2, n_samples)  # binary classification
    
    # Create and train Stage 1 model (qualifying)
    stage1_model = RandomForestRegressor(n_estimators=10, random_state=42)
    stage1_model.fit(X, y_stage1)
    
    # Save Stage 1 model
    joblib.dump(stage1_model, models_dir / 'stage1_lgb_ensemble.pkl')
    print("✅ Created stage1_lgb_ensemble.pkl")
    
    # Create and train Stage 2 model (race winner)
    stage2_model = RandomForestClassifier(n_estimators=10, random_state=42)
    stage2_model.fit(X, y_stage2)
    
    # Save Stage 2 model
    joblib.dump(stage2_model, models_dir / 'stage2_ensemble.pkl')
    print("✅ Created stage2_ensemble.pkl")
    
    # Create preprocessor
    preprocessor = StandardScaler()
    preprocessor.fit(X)
    
    # Save preprocessor
    joblib.dump(preprocessor, models_dir / 'preprocessor.pkl')
    print("✅ Created preprocessor.pkl")
    
    # Create feature names file
    with open(models_dir / 'feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print("✅ Created feature_names.txt")
    
    # Create model metadata
    metadata = {
        'stage1_model': {
            'type': 'RandomForestRegressor',
            'target': 'qualifying_time',
            'features': n_features,
            'created': pd.Timestamp.now().isoformat()
        },
        'stage2_model': {
            'type': 'RandomForestClassifier', 
            'target': 'race_winner',
            'features': n_features,
            'created': pd.Timestamp.now().isoformat()
        }
    }
    
    import json
    with open(models_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Created model_metadata.json")
    
    print(f"\n🎯 Test models created successfully in {models_dir.absolute()}")
    print("These are minimal models for CI/CD testing only.")
    
    return True

if __name__ == '__main__':
    try:
        create_test_models()
        print("✅ Test model creation completed successfully")
    except Exception as e:
        print(f"❌ Error creating test models: {e}")
        exit(1)