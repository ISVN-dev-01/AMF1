"""
Phase 2 orchestrator - Data cleaning & validation
Runs cleaning and validation steps in sequence
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

def run_tests():
    """Run validation tests using pytest"""
    print(f"\n{'='*60}")
    print("Running Validation Tests")
    print('='*60)
    
    try:
        # Run validation report first
        result = subprocess.run([sys.executable, 'tests/test_clean_master.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Run pytest
        result = subprocess.run([sys.executable, 'tests/test_clean_master.py', '--test'], 
                              capture_output=True, text=True)
        
        print("Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Test Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All validation tests passed")
            return True
        else:
            print("⚠️  Some validation tests failed or had warnings")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Validation tests failed to run")
        print(f"Error: {e}")
        return False

def check_deliverables():
    """Check if expected deliverables exist"""
    print(f"\n{'='*60}")
    print("DELIVERABLES CHECK")
    print('='*60)
    
    deliverables = [
        "data/processed/master_dataset.parquet",
    ]
    
    all_exist = True
    for deliverable in deliverables:
        if Path(deliverable).exists():
            size_mb = Path(deliverable).stat().st_size / (1024*1024)
            print(f"✅ {deliverable} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {deliverable} (missing)")
            all_exist = False
    
    return all_exist

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/processed',
        'tests'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    """Main orchestrator function"""
    print("🏎️  F1 AMF Data Processing - Phase 2")
    print("=" * 50)
    print("Goal: Clean and validate master dataset")
    
    # Create directories
    create_directories()
    
    # Check prerequisites
    input_file = Path('data/raw/master_dataset.parquet')
    if not input_file.exists():
        print(f"\n❌ Input file missing: {input_file}")
        print("Please run Phase 1 data collection first:")
        print("  python src/data_collection/run_phase1.py")
        return False
    
    print(f"\n✅ Input file found: {input_file}")
    
    # Define processing steps
    steps = [
        ("src/data_collection/clean_master.py", "Data Cleaning & Normalization"),
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
    
    # Run validation tests
    test_success = run_tests()
    results["Validation Tests"] = test_success
    
    # Check deliverables
    deliverables_exist = check_deliverables()
    
    # Summary
    print(f"\n{'='*60}")
    print("PHASE 2 DATA PROCESSING SUMMARY")
    print('='*60)
    
    successful = 0
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {step}")
        if success:
            successful += 1
    
    print(f"\n{successful}/{len(results)} steps completed successfully")
    
    if deliverables_exist and successful == len(results):
        print(f"\n🎉 Phase 2 completed successfully!")
        print(f"📁 Clean dataset available at: data/processed/master_dataset.parquet")
        print(f"🔍 Ready for Phase 3 - Feature Engineering")
    else:
        print(f"\n⚠️  Phase 2 completed with issues")
        
    return successful == len(results) and deliverables_exist

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)