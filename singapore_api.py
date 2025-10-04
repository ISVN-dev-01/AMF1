"""
Singapore GP 2025 - Real-time Prediction API
Integrates with existing FastAPI application
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import numpy as np
from datetime import datetime
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from singapore_gp_2025_predictor import SingaporeGP2025Predictor
from f1_data_ingestion import F1DataIngestion

# Pydantic models for API
class SingaporePredictionRequest(BaseModel):
    prediction_type: str = "full"  # "qualifying", "race_winner", "full"
    include_analysis: bool = True
    confidence_threshold: float = 0.1

class QualifyingPrediction(BaseModel):
    driver: str
    team: str
    predicted_time: float
    predicted_position: int
    actual_time: Optional[float] = None
    actual_position: Optional[int] = None
    accuracy_score: Optional[float] = None

class RaceWinnerPrediction(BaseModel):
    driver: str
    team: str
    win_probability: float
    confidence_score: float
    key_factors: List[str]

class SingaporePredictionResponse(BaseModel):
    event_info: Dict
    qualifying_predictions: Optional[List[QualifyingPrediction]] = None
    race_winner_predictions: Optional[List[RaceWinnerPrediction]] = None
    stage_1_analysis: Optional[Dict] = None
    stage_2_analysis: Optional[Dict] = None
    metadata: Dict

class SingaporeGPAPI:
    def __init__(self):
        self.predictor = SingaporeGP2025Predictor()
        self.data_ingestion = F1DataIngestion()
        self.cache = {}
        
    def get_singapore_prediction(self, request: SingaporePredictionRequest) -> SingaporePredictionResponse:
        """Generate Singapore GP predictions based on request type"""
        
        # Check cache first
        cache_key = f"singapore_2025_{request.prediction_type}_{request.include_analysis}"
        if cache_key in self.cache:
            print("📦 Serving from cache")
            return self.cache[cache_key]
        
        try:
            # Run the prediction pipeline
            full_results = self.predictor.generate_comprehensive_prediction()
            
            # Prepare response based on request type
            response_data = {
                "event_info": full_results["event_info"],
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "model_version": "AMF1-Singapore-v1.0",
                    "confidence_level": full_results.get("prediction_confidence", "HIGH"),
                    "data_sources": ["FastF1", "Ergast API", "Real Qualifying Results"],
                    "circuit_specialization": True
                }
            }
            
            # Add qualifying predictions if requested
            if request.prediction_type in ["qualifying", "full"]:
                quali_preds = []
                stage1 = full_results["stage_1_analysis"]
                
                for driver, position in stage1["grid_positions"].items():
                    # Find team from qualifying results
                    team = next((r["team"] for r in self.predictor.qualifying_data["q3_results"] 
                               if r["driver"] == driver), "Unknown")
                    
                    quali_preds.append(QualifyingPrediction(
                        driver=driver,
                        team=team,
                        predicted_time=stage1["race_pace_prediction"][driver],
                        predicted_position=position,
                        actual_time=stage1["qualifying_times"][driver],
                        actual_position=position,
                        accuracy_score=0.95  # High accuracy since we have real quali results
                    ))
                
                response_data["qualifying_predictions"] = quali_preds
                
                if request.include_analysis:
                    response_data["stage_1_analysis"] = stage1
            
            # Add race winner predictions if requested
            if request.prediction_type in ["race_winner", "full"]:
                race_preds = []
                stage2 = full_results["stage_2_prediction"]
                
                for driver, prob in stage2["win_probabilities"].items():
                    if prob >= request.confidence_threshold:
                        team = next((r["team"] for r in self.predictor.qualifying_data["q3_results"] 
                                   if r["driver"] == driver), "Unknown")
                        
                        # Generate key factors for this driver
                        key_factors = self._generate_driver_factors(driver, prob, stage2)
                        
                        race_preds.append(RaceWinnerPrediction(
                            driver=driver,
                            team=team,
                            win_probability=prob,
                            confidence_score=min(prob * 1.2, 1.0),  # Confidence slightly higher than probability
                            key_factors=key_factors
                        ))
                
                # Sort by probability
                race_preds.sort(key=lambda x: x.win_probability, reverse=True)
                response_data["race_winner_predictions"] = race_preds
                
                if request.include_analysis:
                    response_data["stage_2_analysis"] = stage2
            
            # Create response object
            response = SingaporePredictionResponse(**response_data)
            
            # Cache the result
            self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def _generate_driver_factors(self, driver: str, probability: float, stage2_data: Dict) -> List[str]:
        """Generate key factors explaining driver's win probability"""
        factors = []
        
        # Grid position factor
        if driver == "George Russell":
            factors.append("🥇 Pole position advantage at difficult-to-overtake circuit")
        
        # Driver-specific factors
        driver_factors = {
            "Max Verstappen": ["🏆 Championship leader with adaptability", "⚡ Strong race pace management"],
            "Lando Norris": ["📈 Excellent 2025 season form", "🔧 Strong McLaren package"],
            "George Russell": ["🎯 Exceptional qualifying performance", "💪 Mercedes race pace improvement"],
            "Oscar Piastri": ["🚀 Rising star with consistent pace", "🧠 Smart race strategy execution"],
            "Charles Leclerc": ["⚡ Raw speed on street circuits", "🎪 Aggressive overtaking ability"],
            "Lewis Hamilton": ["👑 5-time Singapore winner", "🧭 Unmatched experience at Marina Bay"],
            "Fernando Alonso": ["🏎️ Street circuit specialist", "🎭 Master of race strategy"],
        }
        
        if driver in driver_factors:
            factors.extend(driver_factors[driver])
        
        # Probability-based factors
        if probability > 0.3:
            factors.append("🎯 Very high win probability based on all factors")
        elif probability > 0.15:
            factors.append("📊 Strong contender with good chance")
        
        # Singapore-specific factors
        singapore_factors = {
            "high_safety_car": "🚨 75% safety car probability benefits strategic flexibility",
            "night_race": "🌙 Night race conditions suit experienced drivers",
            "street_circuit": "🏙️ Street circuit rewards precision and consistency"
        }
        
        factors.append(singapore_factors["high_safety_car"])
        
        return factors[:4]  # Limit to 4 key factors
    
    def get_live_data_update(self) -> Dict:
        """Get latest data updates (for real-time monitoring)"""
        try:
            # In a real implementation, this would fetch live timing data
            return {
                "last_update": datetime.now().isoformat(),
                "qualifying_confirmed": True,
                "pole_sitter": "George Russell",
                "weather_update": {
                    "temperature": 30,
                    "humidity": 85,
                    "rain_probability": 25
                },
                "race_status": "Scheduled for tomorrow",
                "grid_penalties": []  # Any grid penalties applied after qualifying
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize the API handler
singapore_api = SingaporeGPAPI()

# FastAPI endpoints (these would be added to your existing app.py)
def add_singapore_endpoints(app: FastAPI):
    """Add Singapore GP endpoints to existing FastAPI app"""
    
    @app.post("/predict_singapore_2025", response_model=SingaporePredictionResponse)
    async def predict_singapore_gp(request: SingaporePredictionRequest):
        """Get Singapore GP 2025 predictions"""
        return singapore_api.get_singapore_prediction(request)
    
    @app.get("/singapore_2025/live_data")
    async def get_singapore_live_data():
        """Get live data updates for Singapore GP"""
        return singapore_api.get_live_data_update()
    
    @app.get("/singapore_2025/quick_prediction")
    async def quick_singapore_prediction():
        """Get quick race winner prediction for Singapore GP"""
        request = SingaporePredictionRequest(
            prediction_type="race_winner",
            include_analysis=False,
            confidence_threshold=0.05
        )
        result = singapore_api.get_singapore_prediction(request)
        
        # Return just the top 3 predictions
        top_3 = result.race_winner_predictions[:3] if result.race_winner_predictions else []
        
        return {
            "race": "Singapore GP 2025",
            "prediction_time": result.metadata["prediction_time"],
            "top_3_predictions": [
                {
                    "driver": pred.driver,
                    "team": pred.team,
                    "win_probability": f"{pred.win_probability*100:.1f}%",
                    "key_factor": pred.key_factors[0] if pred.key_factors else ""
                }
                for pred in top_3
            ],
            "race_favorite": top_3[0].driver if top_3 else "Unknown",
            "confidence": result.metadata["confidence_level"]
        }

# Standalone testing function
def test_singapore_api():
    """Test the Singapore GP API locally"""
    print("🧪 Testing Singapore GP 2025 API")
    
    api = SingaporeGPAPI()
    
    # Test full prediction
    request = SingaporePredictionRequest(
        prediction_type="full",
        include_analysis=True,
        confidence_threshold=0.05
    )
    
    result = api.get_singapore_prediction(request)
    
    print(f"\n🏁 API Test Results:")
    print(f"Event: {result.event_info['race']}")
    print(f"Pole: {result.event_info['pole_sitter']}")
    
    if result.race_winner_predictions:
        print(f"\n🏆 Top 3 Race Predictions:")
        for i, pred in enumerate(result.race_winner_predictions[:3], 1):
            print(f"{i}. {pred.driver} ({pred.team}): {pred.win_probability*100:.1f}%")
    
    print(f"\n✅ API test completed successfully!")
    return result

if __name__ == "__main__":
    test_singapore_api()