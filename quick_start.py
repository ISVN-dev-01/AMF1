#!/usr/bin/env python3
"""
F1 ML System Quick Start - Test Predictions
"""

import sys
import subprocess
import time
import requests
import json

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi, uvicorn, joblib, pandas, numpy
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting F1 Prediction API server...")
    print("   (This will take a few seconds to load models)")
    
    # Start server in background
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.serve.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8080",
        "--reload"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8080/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server started successfully!")
                return process
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"⏳ Waiting for server... ({i+1}s)")
    
    print("❌ Server failed to start")
    process.terminate()
    return None

def test_prediction():
    """Test a simple prediction"""
    print("\n🎯 Testing F1 predictions...")
    
    # Sample data for Hamilton vs Verstappen
    test_data = {
        "drivers": [
            {
                "driver_id": 44,  # Hamilton
                "team_id": 2,
                "circuit_id": 1,
                "weather_condition": "dry",
                "temperature": 25.0,
                "humidity": 50.0
            },
            {
                "driver_id": 1,   # Verstappen
                "team_id": 1,
                "circuit_id": 1, 
                "weather_condition": "dry",
                "temperature": 25.0,
                "humidity": 50.0
            }
        ]
    }
    
    try:
        # Test qualifying prediction
        response = requests.post("http://localhost:8080/predict_quali", 
                               json=test_data,
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Qualifying prediction successful!")
            print(f"   Hamilton (44): {result[0]['predicted_time']:.3f}s")
            print(f"   Verstappen (1): {result[1]['predicted_time']:.3f}s")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    """Main quick start function"""
    print("🏎️  F1 ML PREDICTION SYSTEM - QUICK START")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Start server
    server_process = start_server()
    if not server_process:
        return
    
    try:
        # Step 3: Test prediction
        if test_prediction():
            print("\n🎉 SUCCESS! Your F1 ML system is working!")
            print("\n📋 What's next:")
            print("   1. Keep this terminal open (server running)")
            print("   2. Open browser: http://localhost:8080/docs")
            print("   3. Try the interactive API docs")
            print("   4. Run: python make_prediction.py")
            print("\n🛑 Press Ctrl+C to stop the server")
            
            # Keep server running
            try:
                server_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Shutting down server...")
                server_process.terminate()
        else:
            print("\n❌ Quick start failed - check the error messages above")
            server_process.terminate()
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server_process.terminate()

if __name__ == "__main__":
    main()