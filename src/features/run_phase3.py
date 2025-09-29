#!/usr/bin/env python3
"""
Phase 3 orchestrator - Label generation
Creates target variables for machine learning models
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        print(f"✅ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_prerequisites():
    """Check if required input files exist"""
    print("Checking prerequisites...")
    
    required_files = [
        'data/processed/master_dataset.parquet'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024*1024)
            print(f"✅ {file_path} ({size_mb:.2f} MB)")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} (missing)")
    
    if missing_files:
        print(f"\nMissing prerequisites: {missing_files}")
        print("Please run previous phases first:")
        print("  Phase 1: python src/data_collection/run_phase1.py")
        print("  Phase 2: python src/data_collection/run_phase2.py")
        return False
    
    return True

def check_deliverables():
    """Check if expected deliverables exist"""
    print(f"\n{'='*60}")
    print("DELIVERABLES CHECK")
    print('='*60)
    
    deliverables = [
        "data/processed/labels.parquet",
    ]
    
    all_exist = True
    for deliverable in deliverables:
        if Path(deliverable).exists():
            size_mb = Path(deliverable).stat().st_size / (1024*1024)
            print(f"✅ {deliverable} ({size_mb:.2f} MB)")
            
            # Show label statistics
            if 'labels.parquet' in deliverable:
                try:
                    import pandas as pd
                    labels_df = pd.read_parquet(deliverable)
                    print(f"    Records: {len(labels_df):,}")
                    print(f"    Columns: {list(labels_df.columns)}")
                    if 'is_pole' in labels_df.columns:
                        print(f"    Pole positions: {labels_df['is_pole'].sum():,}")
                    if 'is_race_winner' in labels_df.columns:
                        print(f"    Race wins: {labels_df['is_race_winner'].sum():,}")
                except Exception as e:
                    print(f"    Warning: Could not read file details: {e}")
        else:
            print(f"❌ {deliverable} (missing)")
            all_exist = False
    
    return all_exist

def create_directories():
    """Create necessary directories"""
    directories = [
        'src/features',
        'data/processed'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    """Main orchestrator function"""
    print("🏎️  F1 AMF Label Generation - Phase 3")
    print("=" * 50)
    print("Goal: Create ML target variables")
    print("Targets: is_pole, quali_best_time, race_position, is_race_winner")
    
    # Create directories
    create_directories()
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Define processing steps
    steps = [
        ("src/features/create_labels.py", "Label Generation"),
    ]
    
    # Track success/failure
    results = {}
    
    # Run each step
    for script_path, description in steps:
        if Path(script_path).exists():
            success = run_script(script_path, description)
            results[description] = success
        else:
            print(f"❌ Script not found: {script_path}")
            results[description] = False
    
    # Check deliverables
    deliverables_exist = check_deliverables()
    
    # Summary
    print(f"\n{'='*60}")
    print("PHASE 3 LABEL GENERATION SUMMARY")
    print('='*60)
    
    successful = 0
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {step}")
        if success:
            successful += 1
    
    print(f"\n{successful}/{len(results)} steps completed successfully")
    
    if deliverables_exist and successful == len(results):
        print(f"\n🎉 Phase 3 completed successfully!")
        print(f"📁 Labels available at: data/processed/labels.parquet")
        print(f"🔍 Ready for Phase 4 - Feature Engineering & Model Training")
        
        # Show next steps
        print(f"\n📋 Generated Target Variables:")
        print(f"  • is_pole: Binary flag for pole position (qualifying 1st)")
        print(f"  • quali_best_time: Best qualifying lap time in seconds")
        print(f"  • race_position: Final race finishing position")  
        print(f"  • is_race_winner: Binary flag for race winner (1st place)")
        
    else:
        print(f"\n⚠️  Phase 3 completed with issues")
        if not deliverables_exist:
            print("  - Missing expected output files")
        if successful < len(results):
            print("  - Some processing steps failed")
        
    return successful == len(results) and deliverables_exist

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)