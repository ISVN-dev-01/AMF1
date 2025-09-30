#!/usr/bin/env python3
"""
PHASE 10.2: FastAPI Server for F1 ML Predictions
Production-ready API server for qualifying and race winner predictions
"""

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import time

# Import monitoring system
sys.path.append(str(Path(__file__).parent.parent))
from monitoring.metrics_collector import f1_monitor, update_race_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="F1 Prediction API",
    description="Formula 1 Machine Learning Prediction Service - Qualifying Times & Race Winners",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model storage
models = {}

class DriverFeatures(BaseModel):
    """Pydantic model for driver feature input"""
    driver_id: int
    team_id: Optional[int] = None
    circuit_id: Optional[int] = None
    quali_best_time: Optional[float] = None
    weather_condition: Optional[str] = "dry"
    temperature: Optional[float] = 25.0
    humidity: Optional[float] = 50.0
    wind_speed: Optional[float] = 5.0
    rain_probability: Optional[float] = 0.0
    track_temperature: Optional[float] = 30.0
    air_pressure: Optional[float] = 1013.25
    # Additional features can be added as needed
    additional_features: Optional[Dict[str, float]] = {}

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    drivers: List[DriverFeatures]
    session_type: Optional[str] = "qualifying"
    race_id: Optional[int] = None

class QualifyingPrediction(BaseModel):
    """Response model for qualifying predictions"""
    driver_id: int
    predicted_time: float
    probability_score: float
    grid_position_estimate: int

class RaceWinnerPrediction(BaseModel):
    """Response model for race winner predictions"""
    driver_id: int
    win_probability: float
    confidence_score: float
    ranking: int

class PredictionResponse(BaseModel):
    """Combined response model"""
    qualifying_predictions: List[QualifyingPrediction]
    race_winner_predictions: List[RaceWinnerPrediction]
    metadata: Dict[str, Any]

def load_models():
    """Load all trained models and preprocessing pipeline"""
    try:
        models_dir = Path('models')
        
        # Load all components directly
        models['preprocessor'] = joblib.load(models_dir / 'preprocessor.pkl')
        models['stage1_model'] = joblib.load(models_dir / 'stage1_lgb_ensemble.pkl')  # This is a list
        models['stage2_model'] = joblib.load(models_dir / 'stage2_ensemble.pkl')
        models['metadata'] = joblib.load(models_dir / 'feature_metadata.pkl')
        
        logger.info(f"✅ All models loaded successfully")
        logger.info(f"   Features: {len(models['metadata']['feature_columns'])}")
        logger.info(f"   Model version: {models['metadata']['model_version']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return False

def prepare_features_from_request(drivers: List[DriverFeatures]) -> pd.DataFrame:
    """Convert request data to feature DataFrame"""
    
    feature_columns = models['metadata']['feature_columns']
    
    # Initialize DataFrame with required columns
    df_data = {}
    for col in feature_columns:
        df_data[col] = []
    
    for driver in drivers:
        # Map common features
        driver_dict = driver.dict()
        
        for col in feature_columns:
            if col in driver_dict:
                value = driver_dict[col]
            elif col in driver.additional_features:
                value = driver.additional_features[col]
            else:
                # Default values for missing features
                if 'temperature' in col:
                    value = 25.0
                elif 'humidity' in col:
                    value = 50.0
                elif 'wind' in col:
                    value = 5.0
                elif 'rain' in col or 'weather' in col:
                    value = 0.0
                elif 'pressure' in col:
                    value = 1013.25
                elif 'time' in col:
                    value = 80.0  # Default lap time
                else:
                    value = 0.0
            
            df_data[col].append(value)
    
    return pd.DataFrame(df_data)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting F1 Prediction API...")
    success = load_models()
    if not success:
        logger.error("❌ Failed to load models - API may not work correctly")
    else:
        logger.info("✅ API ready for predictions")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "F1 Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "models_loaded": len(models) > 0
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "model_version": models.get('metadata', {}).get('model_version', 'unknown'),
        "feature_count": len(models.get('metadata', {}).get('feature_columns', [])),
        "endpoints": ["/predict_quali", "/predict_race", "/predict_full"]
    }

@app.post("/predict_quali", response_model=List[QualifyingPrediction])
async def predict_qualifying(request: PredictionRequest):
    """Predict qualifying times (Stage-1)"""
    
    start_time = time.time()
    
    if 'stage1_model' not in models:
        f1_monitor.metrics.record_prediction_request("predict_quali", "error")
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Record request
        f1_monitor.metrics.record_prediction_request("predict_quali", "success")
        
        # Prepare features
        df = prepare_features_from_request(request.drivers)
        logger.info(f"Processing {len(df)} drivers for qualifying prediction")
        
        # Preprocess features
        X_processed = models['preprocessor'].transform(df)
        
        # Predict qualifying times (stage1_model is a list with one model)
        pred_times = models['stage1_model'][0].predict(X_processed)
        
        # Create predictions with ranking
        predictions = []
        sorted_indices = np.argsort(pred_times)  # Faster times get better positions
        
        for i, (idx, driver) in enumerate(zip(sorted_indices, request.drivers)):
            predictions.append(QualifyingPrediction(
                driver_id=driver.driver_id,
                predicted_time=float(pred_times[idx]),
                probability_score=float(1.0 / (1.0 + np.exp((pred_times[idx] - np.mean(pred_times)) * 0.1))),
                grid_position_estimate=i + 1
            ))
        
        # Record latency
        duration = time.time() - start_time
        f1_monitor.metrics.record_prediction_latency("predict_quali", "stage1", duration)
        
        logger.info(f"✅ Generated {len(predictions)} qualifying predictions")
        return predictions
        
    except Exception as e:
        f1_monitor.metrics.record_prediction_request("predict_quali", "error")
        logger.error(f"❌ Qualifying prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_race", response_model=List[RaceWinnerPrediction])
async def predict_race_winners(request: PredictionRequest):
    """Predict race winner probabilities (Stage-2)"""
    
    start_time = time.time()
    
    if 'stage2_model' not in models:
        f1_monitor.metrics.record_prediction_request("predict_race", "error")
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Record request
        f1_monitor.metrics.record_prediction_request("predict_race", "success")
        
        # Prepare features
        df = prepare_features_from_request(request.drivers)
        logger.info(f"Processing {len(df)} drivers for race winner prediction")
        
        # Preprocess features
        X_processed = models['preprocessor'].transform(df)
        
        # Get Stage-1 predictions first (stage1_model is a list with one model)
        stage1_preds = models['stage1_model'][0].predict(X_processed)
        
        # Combine features for Stage-2
        X_stage2 = np.column_stack([X_processed, stage1_preds])
        
        # Predict win probabilities
        win_probabilities = models['stage2_model'].predict_proba(X_stage2)[:, 1]
        
        # Create predictions with ranking
        predictions = []
        sorted_indices = np.argsort(win_probabilities)[::-1]  # Higher probabilities ranked first
        
        for i, (idx, driver) in enumerate(zip(sorted_indices, request.drivers)):
            confidence = min(win_probabilities[idx] * 2, 1.0)  # Scale confidence
            
            predictions.append(RaceWinnerPrediction(
                driver_id=driver.driver_id,
                win_probability=float(win_probabilities[idx]),
                confidence_score=float(confidence),
                ranking=i + 1
            ))
        
        # Record latency
        duration = time.time() - start_time
        f1_monitor.metrics.record_prediction_latency("predict_race", "stage2", duration)
        
        logger.info(f"✅ Generated {len(predictions)} race winner predictions")
        return predictions
        
    except Exception as e:
        f1_monitor.metrics.record_prediction_request("predict_race", "error")
        logger.error(f"❌ Race winner prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_full", response_model=PredictionResponse)
async def predict_full_race(request: PredictionRequest):
    """Complete two-stage F1 prediction pipeline"""
    
    if 'stage1_model' not in models or 'stage2_model' not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        logger.info(f"Processing full race prediction for {len(request.drivers)} drivers")
        
        # Get both predictions
        qualifying_preds = await predict_qualifying(request)
        race_winner_preds = await predict_race_winners(request)
        
        # Create metadata
        metadata = {
            "model_version": models['metadata']['model_version'],
            "prediction_timestamp": pd.Timestamp.now().isoformat(),
            "driver_count": len(request.drivers),
            "session_type": request.session_type,
            "race_id": request.race_id,
            "feature_count": len(models['metadata']['feature_columns'])
        }
        
        response = PredictionResponse(
            qualifying_predictions=qualifying_preds,
            race_winner_predictions=race_winner_preds,
            metadata=metadata
        )
        
        logger.info(f"✅ Generated full race predictions")
        return response
        
    except Exception as e:
        logger.error(f"❌ Full race prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about loaded models"""
    
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    metadata = models.get('metadata', {})
    
    return {
        "model_version": metadata.get('model_version', 'unknown'),
        "training_date": metadata.get('training_date', 'unknown'),
        "feature_count": len(metadata.get('feature_columns', [])),
        "model_type": "two_stage_ensemble",
        "feature_columns": metadata.get('feature_columns', []),
        "stage1_mae": metadata.get('stage1_mae', 'unknown'),
        "stage2_accuracy": metadata.get('stage2_accuracy', 'unknown')
    }

@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    try:
        metrics_output = f1_monitor.metrics.generate_metrics()
        return Response(content=metrics_output, media_type="text/plain")
    except Exception as e:
        logger.error(f"❌ Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics generation failed")

@app.post("/submit_race_results")
async def submit_race_results(race_results: Dict[str, Any]):
    """Submit actual race results for monitoring"""
    try:
        race_id = race_results.get('race_id')
        predictions = race_results.get('predictions', [])
        actuals = race_results.get('actuals', [])
        
        if not race_id or not predictions or not actuals:
            raise HTTPException(status_code=400, detail="Missing required race result data")
        
        # Create features DataFrame from predictions (simplified)
        features_data = []
        for pred in predictions:
            features_data.append({
                'temperature': pred.get('temperature', 25.0),
                'humidity': pred.get('humidity', 60.0),
                'wind_speed': pred.get('wind_speed', 10.0)
            })
        
        features_df = pd.DataFrame(features_data)
        
        # Update monitoring metrics
        update_race_metrics(race_id, predictions, actuals, features_df)
        
        # Check if retraining is needed
        retrain_check = f1_monitor.check_retrain_triggers()
        
        return {
            "status": "success",
            "race_id": race_id,
            "metrics_updated": True,
            "retrain_needed": retrain_check['should_retrain'],
            "retrain_reasons": retrain_check.get('reasons', [])
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to submit race results: {e}")
        raise HTTPException(status_code=500, detail=f"Race results submission failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🏁 Starting F1 Prediction API Server...")
    print("📊 Endpoints available:")
    print("   - GET  /              : Health check")
    print("   - GET  /health        : Detailed health check")
    print("   - POST /predict_quali : Qualifying time predictions")
    print("   - POST /predict_race  : Race winner predictions")
    print("   - POST /predict_full  : Full two-stage predictions")
    print("   - GET  /model_info    : Model information")
    print("   - GET  /docs         : Interactive API documentation")
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )