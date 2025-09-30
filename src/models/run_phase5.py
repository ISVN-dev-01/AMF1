#!/usr/bin/env python3
"""
PHASE 5 Orchestrator: Run complete data preparation pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

from prepare_data import F1DataPreparer

def run_phase5():
    """Run Phase 5: Complete data preparation pipeline"""
    
    print("=" * 80)
    print("PHASE 5 ORCHESTRATOR: DATA PREPARATION FOR ML TRAINING")
    print("=" * 80)
    
    try:
        # Initialize data preparer
        preparer = F1DataPreparer()
        
        # Run complete preparation for pole prediction
        print("🎯 Preparing data for POLE PREDICTION...")
        pole_results = preparer.run_complete_preparation(
            target_task='pole_prediction', 
            cv_splits=5
        )
        
        print(f"\n" + "="*60)
        print("POLE PREDICTION - PREPARATION COMPLETE")
        print("="*60)
        print(f"✅ Training samples: {len(pole_results['X_train']):,}")
        print(f"✅ Features: {pole_results['X_train'].shape[1]}")
        print(f"✅ CV folds: {len(pole_results['cv_indices'])}")
        print(f"✅ Files saved: {len(pole_results['files'])}")
        
        # Run complete preparation for race winner prediction
        print(f"\n🎯 Preparing data for RACE WINNER PREDICTION...")
        winner_results = preparer.run_complete_preparation(
            target_task='race_winner_prediction',
            cv_splits=5
        )
        
        print(f"\n" + "="*60)
        print("RACE WINNER PREDICTION - PREPARATION COMPLETE")
        print("="*60)
        print(f"✅ Training samples: {len(winner_results['X_train']):,}")
        print(f"✅ Features: {winner_results['X_train'].shape[1]}")
        print(f"✅ CV folds: {len(winner_results['cv_indices'])}")
        
        # Summary
        print(f"\n" + "="*80)
        print("PHASE 5 ORCHESTRATOR - COMPLETE SUCCESS")
        print("="*80)
        
        print(f"📊 Data Preparation Summary:")
        print(f"   • Comprehensive dataset: 2014-2024 (11 seasons)")
        print(f"   • Time-aware splits: Train(2014-2022) | Val(2023) | Test(2024)")
        print(f"   • Group-aware CV: 5 folds with race_id grouping")
        print(f"   • Target tasks: pole_prediction, race_winner_prediction")
        print(f"   • Features: {pole_results['X_train'].shape[1]} engineered features")
        
        print(f"\n📁 Generated Files:")
        print(f"   • data/models/splits/ - Train/val/test parquet files")
        print(f"   • data/processed/cv_indices.pkl - Cross-validation indices")
        print(f"   • data/models/split_metadata.json - Configuration metadata")
        print(f"   • data/models/label_encoders.pkl - Categorical encoders")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Model training: Use GroupKFold CV with saved indices")
        print(f"   2. Hyperparameter tuning: Leverage temporal safety guarantees")
        print(f"   3. Feature importance: Analyze driver vs team vs track factors")
        print(f"   4. Model deployment: FastAPI endpoints for real-time prediction")
        
        print(f"\n✅ Phase 5 Complete - Ready for Model Training! 🏁")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 5 orchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_phase5()
    if not success:
        sys.exit(1)