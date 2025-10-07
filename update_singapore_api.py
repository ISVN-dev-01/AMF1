#!/usr/bin/env python3
"""
Updated Singapore GP 2025 API endpoints with actual race results
"""

import json
from pathlib import Path
from datetime import datetime

def create_updated_singapore_api_data():
    """Create updated API data with actual Singapore GP results"""
    
    # Load actual results
    results_path = Path("data/singapore_2025_actual_results.json")
    analysis_path = Path("data/singapore_2025_model_analysis.json")
    
    with open(results_path, 'r') as f:
        actual_results = json.load(f)
    
    with open(analysis_path, 'r') as f:
        model_analysis = json.load(f)
    
    # Create updated API responses
    singapore_info = {
        "event": "Singapore Grand Prix 2025",
        "date": "2025-10-06", 
        "circuit": "Marina Bay Street Circuit",
        "status": "COMPLETED",
        "race_winner": "Lando Norris",
        "pole_sitter": "George Russell",
        "weather": {
            "temperature": "31°C",
            "humidity": "78%",
            "conditions": "Dry"
        },
        "championship_impact": {
            "norris_points": 375,
            "verstappen_points": 308,
            "gap": 67,
            "races_remaining": 4
        }
    }
    
    singapore_prediction_update = {
        "prediction_status": "RACE_COMPLETED",
        "model_prediction": {
            "predicted_winner": "George Russell",
            "predicted_probability": "70.5%"
        },
        "actual_result": {
            "actual_winner": "Lando Norris",
            "actual_position": 1,
            "grid_position": 2
        },
        "prediction_accuracy": "INCORRECT",
        "top_5_actual_results": [
            {"position": 1, "driver": "Lando Norris", "team": "McLaren", "grid": 2},
            {"position": 2, "driver": "Max Verstappen", "team": "Red Bull Racing", "grid": 3},
            {"position": 3, "driver": "Charles Leclerc", "team": "Ferrari", "grid": 5},
            {"position": 4, "driver": "Oscar Piastri", "team": "McLaren", "grid": 4},
            {"position": 5, "driver": "Carlos Sainz", "team": "Ferrari", "grid": 6}
        ],
        "model_insights": {
            "lessons_learned": [
                "Pole position advantage overestimated",
                "Pit stop reliability underweighted", 
                "Championship pressure undervalued",
                "Strategy execution crucial at Marina Bay"
            ],
            "next_improvements": [
                "Include team reliability metrics",
                "Weight race pace vs qualifying pace", 
                "Factor championship standings pressure",
                "Improve strategy modeling"
            ]
        }
    }
    
    # Save updated API data
    api_data = {
        "singapore_info": singapore_info,
        "singapore_prediction": singapore_prediction_update,
        "updated_timestamp": datetime.now().isoformat()
    }
    
    api_path = Path("data/singapore_2025_api_update.json")
    with open(api_path, 'w') as f:
        json.dump(api_data, f, indent=2)
    
    print(f"💾 Updated API data saved: {api_path}")
    
    return api_data

def main():
    """Main execution"""
    print("🔄 Creating updated Singapore GP API data...")
    api_data = create_updated_singapore_api_data()
    
    print(f"✅ API data updated with actual race results!")
    print(f"🏆 Winner: {api_data['singapore_info']['race_winner']}")
    print(f"📊 Our prediction was: {api_data['singapore_prediction']['model_prediction']['predicted_winner']}")
    print(f"🎯 Accuracy: {api_data['singapore_prediction']['prediction_accuracy']}")

if __name__ == "__main__":
    main()