"""
Main data collection orchestrator for F1 AMF project
Runs all Phase 1 data collection steps in sequence
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
    """Check if required packages are installed"""
    print("Checking prerequisites...")
    
    required_packages = ['requests', 'pandas', 'fastf1', 'python-dotenv', 'pyarrow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/raw/weather', 
        'cache/fastf1'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def main():
    """Main orchestrator function"""
    print("🏎️  F1 AMF Data Collection - Phase 1")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return False
    
    # Create directories
    create_directories()
    
    # Define collection steps
    steps = [
        ("src/data_collection/ergast_collect.py", "Ergast Historical Data Collection"),
        ("src/data_collection/fastf1_collect.py", "FastF1 Practice & Telemetry Collection"), 
        ("src/data_collection/weather_collect.py", "Weather Data Collection"),
        ("src/data_collection/create_master_dataset.py", "Master Dataset Creation")
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
    
    # Summary
    print(f"\n{'='*60}")
    print("PHASE 1 DATA COLLECTION SUMMARY")
    print('='*60)
    
    successful = 0
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {step}")
        if success:
            successful += 1
    
    print(f"\n{successful}/{len(results)} steps completed successfully")
    
    # Check deliverables
    print(f"\n{'='*60}")
    print("DELIVERABLES CHECK")
    print('='*60)
    
    deliverables = [
        "data/raw/races.parquet",
        "data/raw/qualifying.parquet", 
        "data/raw/results.parquet",
        "data/raw/fastf1_q_monaco_2024.parquet",
        "data/raw/weather/weather_2024.parquet",
        "data/raw/master_dataset.parquet"
    ]
    
    for deliverable in deliverables:
        if Path(deliverable).exists():
            size_mb = Path(deliverable).stat().st_size / (1024*1024)
            print(f"✅ {deliverable} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {deliverable} (missing)")
    
    return successful == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)