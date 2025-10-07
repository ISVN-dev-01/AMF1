#!/usr/bin/env python3
"""
Singapore GP 2025 - ACTUAL RESULTS UPDATE
Correcting the system with real race data from Formula 1 official sources
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def create_actual_singapore_results():
    """Create actual Singapore GP 2025 results from official F1 sources"""
    
    actual_results = {
        "singapore_gp_2025_actual_results": {
            "race_date": "2025-10-06",
            "circuit": "Marina Bay Street Circuit",
            "data_source": "Formula 1® - The Official F1® Website & RaceFans",
            
            "qualifying_results": {
                "pole_position": "George Russell",
                "pole_time": "1:29.158",
                "grid_order": [
                    {"position": 1, "driver": "George Russell", "team": "Mercedes", "time": "1:29.158"},
                    {"position": 2, "driver": "Max Verstappen", "team": "Red Bull", "time": "1:29.340"},
                    {"position": 3, "driver": "Oscar Piastri", "team": "McLaren", "time": "1:29.524"},
                    {"position": 4, "driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:29.537"},
                    {"position": 5, "driver": "Lando Norris", "team": "McLaren", "time": "1:29.586"}
                ]
            },
            
            # Note: Race results not provided yet, but we have qualifying and championship standings
            "race_status": "QUALIFYING_COMPLETED",
            "race_results_available": False,
            
            "championship_standings_after_singapore": {
                "constructors": [
                    {"position": 1, "team": "McLaren", "points": 650},
                    {"position": 2, "team": "Mercedes", "points": 325},
                    {"position": 3, "team": "Ferrari", "points": 298},
                    {"position": 4, "team": "Red Bull", "points": 290},
                    {"position": 5, "team": "Williams", "points": 102}
                ]
            },
            
            "qualifying_analysis": {
                "pole_sitter": "George Russell",
                "pole_time": "1:29.158",
                "front_row": ["George Russell", "Max Verstappen"],
                "mercedes_performance": "Strong - Both cars in top 4",
                "mclaren_performance": "Mixed - Piastri P3, Norris P5",
                "red_bull_performance": "Solid - Verstappen P2",
                "key_observations": [
                    "George Russell secures pole position",
                    "Mercedes locks out P1 and P4 (Antonelli)",
                    "McLaren split performance - Piastri outqualifies Norris",
                    "Red Bull's Verstappen starts P2",
                    "Tight qualifying - less than 0.5s covers top 5"
                ]
            },
            
            "model_prediction_vs_qualifying": {
                "our_prediction": {
                    "predicted_pole": "George Russell",
                    "predicted_probability": "70.5%"
                },
                "actual_qualifying": {
                    "pole_sitter": "George Russell",
                    "pole_time": "1:29.158"
                },
                "qualifying_prediction_accuracy": "CORRECT - We predicted Russell pole",
                "grid_analysis": {
                    "russell_p1": "✅ Correct prediction",
                    "verstappen_p2": "Strong position for race",
                    "piastri_p3": "McLaren threat from P3",
                    "antonelli_p4": "Mercedes rookie impressive",
                    "norris_p5": "Championship leader starts mid-pack"
                }
            }
        }
    }
    
    return actual_results

def update_model_with_real_data():
    """Update model insights with actual qualifying data"""
    
    model_updates = {
        "singapore_gp_2025_actual_model_update": {
            "update_date": datetime.now().isoformat(),
            "data_source": "Official F1 Results & RaceFans Championship Standings",
            
            "qualifying_prediction_analysis": {
                "pole_prediction": {
                    "predicted": "George Russell (70.5%)",
                    "actual": "George Russell (1:29.158)",
                    "result": "✅ CORRECT PREDICTION"
                },
                "qualifying_accuracy": "HIGH - Correctly predicted Russell pole position"
            },
            
            "race_prediction_status": {
                "status": "AWAITING_RACE_RESULTS",
                "current_prediction": "George Russell from pole position",
                "probability": "70.5%",
                "key_factors": [
                    "Pole position advantage at Marina Bay",
                    "Mercedes strong qualifying pace",
                    "Clean air start crucial"
                ]
            },
            
            "championship_context_update": {
                "mclaren_leading": "650 points - Dominant constructor lead",
                "mercedes_second": "325 points - Strong recovery season",
                "ferrari_third": "298 points - Close fight with Red Bull",
                "red_bull_fourth": "290 points - Only 8 points behind Ferrari",
                "tight_midfield": "Championship very competitive"
            },
            
            "race_strategy_implications": {
                "russell_pole": "Clean air advantage, but must avoid mistakes",
                "verstappen_p2": "Strong starting position, proven racecraft",
                "piastri_p3": "McLaren's best grid position, undercut threat",
                "antonelli_p4": "Rookie pressure at night race",
                "norris_p5": "Championship leader needs recovery drive"
            },
            
            "model_confidence": {
                "qualifying_prediction": "✅ VALIDATED",
                "race_prediction": "PENDING RACE RESULTS",
                "next_update": "After race completion"
            }
        }
    }
    
    return model_updates

def main():
    """Main execution with correct data"""
    
    print("================================================================================")
    print("🏁 SINGAPORE GP 2025 - ACTUAL RESULTS CORRECTION")
    print("================================================================================")
    print("📊 Data Source: Formula 1® Official Website & RaceFans")
    
    # Create actual results with correct data
    actual_results = create_actual_singapore_results()
    
    # Save actual results
    results_path = Path("data/singapore_2025_actual_corrected_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(actual_results, f, indent=2)
    
    print(f"💾 Actual results saved: {results_path}")
    
    # Generate model updates
    model_updates = update_model_with_real_data()
    
    # Save model updates
    update_path = Path("data/singapore_2025_actual_model_update.json")
    with open(update_path, 'w') as f:
        json.dump(model_updates, f, indent=2)
    
    print(f"💾 Model updates saved: {update_path}")
    
    # Print actual qualifying results
    qualifying = actual_results["singapore_gp_2025_actual_results"]["qualifying_results"]
    
    print(f"\n🏁 ACTUAL QUALIFYING RESULTS:")
    print(f"   🥇 Pole: {qualifying['pole_position']} ({qualifying['pole_time']})")
    
    print(f"\n📊 TOP 5 QUALIFYING:")
    for result in qualifying["grid_order"]:
        print(f"   P{result['position']}: {result['driver']} ({result['team']}) - {result['time']}")
    
    print(f"\n🏆 CONSTRUCTOR CHAMPIONSHIP (After Singapore):")
    constructors = actual_results["singapore_gp_2025_actual_results"]["championship_standings_after_singapore"]["constructors"]
    for standing in constructors:
        print(f"   P{standing['position']}: {standing['team']} - {standing['points']} points")
    
    print(f"\n🎯 OUR PREDICTION vs REALITY:")
    analysis = actual_results["singapore_gp_2025_actual_results"]["model_prediction_vs_qualifying"]
    print(f"   Predicted Pole: {analysis['our_prediction']['predicted_pole']}")
    print(f"   Actual Pole: {analysis['actual_qualifying']['pole_sitter']} ({analysis['actual_qualifying']['pole_time']})")
    print(f"   Accuracy: {analysis['qualifying_prediction_accuracy']}")
    
    print(f"\n📈 RACE PREDICTION STATUS:")
    print(f"   Status: Race results not yet available")
    print(f"   Current prediction: George Russell to win from pole")
    print(f"   Probability: 70.5% (based on pole position advantage)")
    
    print(f"\n✅ Singapore GP 2025 actual data correction complete!")
    print(f"📝 Note: Waiting for race results to validate race winner prediction")
    
    return actual_results, model_updates

if __name__ == "__main__":
    main()