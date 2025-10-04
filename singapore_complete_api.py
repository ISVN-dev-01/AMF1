"""
Singapore GP 2025 - Simplified Real-time Prediction API
Standalone version without external F1 data dependencies
"""

from typing import Dict, List, Optional
import json
import numpy as np
from datetime import datetime

# Import our predictor
from singapore_gp_2025_predictor import SingaporeGP2025Predictor

class SimplifiedSingaporeAPI:
    def __init__(self):
        self.predictor = SingaporeGP2025Predictor()
        
    def get_complete_prediction(self) -> Dict:
        """Get complete Singapore GP 2025 prediction"""
        results = self.predictor.generate_comprehensive_prediction()
        
        # Format for API response
        api_response = {
            "race_info": {
                "name": "Singapore Grand Prix 2025",
                "date": "2025-10-05",
                "circuit": "Marina Bay Street Circuit",
                "location": "Singapore",
                "pole_sitter": results["event_info"]["pole_sitter"],
                "race_distance": "61 laps (308.828 km)"
            },
            "qualifying_analysis": {
                "pole_time": "1:29.525",
                "pole_driver": "George Russell",
                "average_gap_to_pole": f"{np.mean(list(results['stage_1_analysis']['relative_performance'].values())):.3f}s",
                "fastest_sectors": {
                    "sector_1": "Mercedes (Russell)",
                    "sector_2": "McLaren (Norris)", 
                    "sector_3": "Red Bull (Verstappen)"
                }
            },
            "race_predictions": {
                "methodology": "2-Stage ML Pipeline (Qualifying → Race Winner)",
                "confidence_level": "HIGH",
                "top_5_predictions": []
            },
            "weather_forecast": {
                "temperature": "30°C",
                "humidity": "85%",
                "rain_probability": "25%",
                "conditions": "Hot and humid, typical Singapore night race"
            },
            "circuit_factors": {
                "safety_car_probability": "75%",
                "overtaking_difficulty": "Very High",
                "grid_position_importance": "Critical",
                "key_challenges": [
                    "Narrow street circuit with limited overtaking",
                    "High safety car probability",
                    "Physical demands in hot, humid conditions",
                    "Night race under artificial lighting"
                ]
            }
        }
        
        # Add top 5 race predictions
        stage2 = results["stage_2_prediction"]
        top_predictions = sorted(stage2["win_probabilities"].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        for i, (driver, prob) in enumerate(top_predictions, 1):
            # Get driver team
            team = next((r["team"] for r in self.predictor.qualifying_data["q3_results"] 
                        if r["driver"] == driver), "Unknown")
            
            # Get grid position
            grid_pos = results["stage_1_analysis"]["grid_positions"].get(driver, 99)
            
            prediction_entry = {
                "position": i,
                "driver": driver,
                "team": team,
                "win_probability": f"{prob*100:.1f}%",
                "grid_position": grid_pos,
                "key_strengths": self._get_driver_strengths(driver),
                "singapore_record": self._get_singapore_record(driver)
            }
            
            api_response["race_predictions"]["top_5_predictions"].append(prediction_entry)
        
        # Add prediction summary
        favorite = top_predictions[0][0]
        api_response["prediction_summary"] = {
            "race_favorite": favorite,
            "win_probability": f"{top_predictions[0][1]*100:.1f}%",
            "confidence": "High - Based on pole position and Singapore expertise",
            "key_insight": f"{favorite} benefits from pole position at Marina Bay where overtaking is extremely difficult",
            "alternative_scenarios": [
                "Safety car deployment could shuffle the order",
                "Weather changes (rain) would favor experienced drivers",
                "Strategic pit windows critical for position changes"
            ]
        }
        
        # Add metadata
        api_response["metadata"] = {
            "prediction_time": datetime.now().isoformat(),
            "model_version": "AMF1-Singapore-2025-v1.0",
            "data_sources": [
                "2025 Singapore GP Qualifying Results",
                "Historical Singapore GP Performance",
                "Current Season Driver/Team Form",
                "Circuit-Specific Racing Factors"
            ],
            "accuracy_notes": "Predictions based on actual qualifying results from Oct 4, 2025"
        }
        
        return api_response
    
    def _get_driver_strengths(self, driver: str) -> List[str]:
        """Get key strengths for each driver"""
        strengths = {
            "George Russell": ["Exceptional qualifier", "Strong race management", "Mercedes pace"],
            "Lando Norris": ["Excellent 2025 form", "Strong McLaren package", "Consistent performer"],
            "Max Verstappen": ["Championship experience", "Adaptable to any circuit", "Race craft"],
            "Oscar Piastri": ["Rising talent", "Smart strategy execution", "Strong under pressure"],
            "Charles Leclerc": ["Raw speed", "Aggressive overtaker", "Street circuit ability"],
            "Lewis Hamilton": ["Singapore specialist (5 wins)", "Unmatched experience", "Strategic mastery"],
            "Fernando Alonso": ["Street circuit expert", "Strategic genius", "Consistency"],
            "Carlos Sainz": ["Solid race pace", "Good tyre management", "Reliable points scorer"],
            "Sergio Pérez": ["Good in traffic", "Strategic flexibility", "Experience"],
            "Lance Stroll": ["Improving form", "Good in difficult conditions", "Team player"]
        }
        return strengths.get(driver, ["Solid performer", "Team contributor"])
    
    def _get_singapore_record(self, driver: str) -> str:
        """Get Singapore GP historical record"""
        records = {
            "Lewis Hamilton": "5 wins (2009, 2014, 2017, 2018, 2019) - Most successful",
            "Max Verstappen": "1 win (2022) - Strong recent form",
            "Fernando Alonso": "Multiple podiums - Street circuit specialist",
            "George Russell": "2nd place (2024) - Improving with Mercedes",
            "Lando Norris": "Consistent points scorer - First podium in 2024",
            "Charles Leclerc": "Podium finisher - Fast but needs reliability",
            "Carlos Sainz": "Regular points scorer - Consistent performer",
            "Oscar Piastri": "Rookie season strong showing - Rising star",
            "Sergio Pérez": "Podium capability - Street circuit experience",
            "Lance Stroll": "Points scorer - Solid night race record"
        }
        return records.get(driver, "Solid performer at Marina Bay")

def main():
    """Run the complete Singapore GP prediction and display results"""
    print("🏎️ SINGAPORE GP 2025 - COMPLETE PREDICTION ANALYSIS")
    print("=" * 70)
    
    api = SimplifiedSingaporeAPI()
    results = api.get_complete_prediction()
    
    # Display race info
    print(f"\n📍 {results['race_info']['name']}")
    print(f"📅 Date: {results['race_info']['date']}")
    print(f"🏁 Pole: {results['race_info']['pole_sitter']}")
    print(f"🌡️  Weather: {results['weather_forecast']['temperature']}, {results['weather_forecast']['humidity']} humidity")
    
    # Display predictions
    print(f"\n🏆 RACE WINNER PREDICTIONS:")
    print("-" * 50)
    for pred in results["race_predictions"]["top_5_predictions"]:
        emoji = "🥇" if pred["position"] == 1 else "🥈" if pred["position"] == 2 else "🥉" if pred["position"] == 3 else "🏁"
        print(f"{emoji} {pred['position']}. {pred['driver']} ({pred['team']})")
        print(f"   Win Probability: {pred['win_probability']}")
        print(f"   Grid: P{pred['grid_position']} | {pred['singapore_record']}")
        print()
    
    # Display key insights
    print("📊 KEY INSIGHTS:")
    print("-" * 30)
    summary = results["prediction_summary"]
    print(f"🎯 Race Favorite: {summary['race_favorite']} ({summary['win_probability']})")
    print(f"💡 Key Insight: {summary['key_insight']}")
    
    print(f"\n🔧 Circuit Factors:")
    for factor in results["circuit_factors"]["key_challenges"]:
        print(f"   • {factor}")
    
    print(f"\n🎲 Alternative Scenarios:")
    for scenario in summary["alternative_scenarios"]:
        print(f"   • {scenario}")
    
    # Save complete results
    with open("singapore_2025_complete_prediction.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Complete analysis saved to 'singapore_2025_complete_prediction.json'")
    print(f"🕒 Generated at: {results['metadata']['prediction_time']}")
    
    return results

if __name__ == "__main__":
    main()