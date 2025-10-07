'use client'

import { useState, useEffect } from 'react'
import { CalendarIcon, MapPinIcon, TrophyIcon } from '@heroicons/react/24/outline'
import axios from 'axios'

interface SingaporeInfo {
  event: string
  date: string
  circuit: string
  status?: string
  qualifying_date?: string
  pole_sitter: string
  pole_time?: string
  race_winner?: string
  race_completed?: boolean
  weather: {
    temperature: string
    humidity: string
    rain_probability?: string
    conditions?: string
  }
  championship_impact?: {
    norris_points: number
    verstappen_points: number
    gap: number
    races_remaining: number
  }
}

interface ActualResult {
  position: number
  driver: string
  team: string
  grid: number
}

interface SingaporePrediction {
  position: number
  driver: string
  team: string
  win_probability: string
  grid_position: string
  key_strength: string
}

interface SingaporeData {
  race: string
  status?: string
  prediction_vs_actual?: {
    our_prediction: {
      winner: string
      probability: string
      reasoning: string
    }
    actual_result: {
      winner: string
      position: number
      grid_start: number
      team: string
    }
    prediction_accuracy: string
  }
  actual_podium?: ActualResult[]
  george_russell_result?: {
    predicted: string
    actual: string
    grid: number
  }
  model_lessons?: string[]
  // Legacy prediction fields
  top_3_predictions?: SingaporePrediction[]
  race_favorite?: string
  key_insight?: string
  safety_car_probability?: string
  pole_sitter?: string
  weather?: string
  prediction_time?: string
}

export default function SingaporeGP2025() {
  const [info, setInfo] = useState<SingaporeInfo | null>(null)
  const [predictions, setPredictions] = useState<SingaporeData | null>(null)

  useEffect(() => {
    fetchSingaporeInfo()
  }, [])

  const fetchSingaporeInfo = async () => {
    try {
      const response = await axios.get('http://localhost:8000/singapore_2025/info')
      setInfo(response.data)
    } catch (err) {
      console.error('Failed to fetch Singapore info:', err)
      // Use fallback data when API is not available
      setInfo({
        event: "Singapore Grand Prix 2025",
        date: "2025-10-05",
        circuit: "Marina Bay Street Circuit",
        qualifying_date: "2025-10-04",
        pole_sitter: "George Russell",
        pole_time: "1:29.525",
        weather: {
          temperature: "30°C",
          humidity: "85%",
          rain_probability: "25%"
        }
      })
    }
  }

  const fetchQuickPrediction = async () => {
    try {
      const response = await axios.get('http://localhost:8000/singapore_2025/quick_prediction')
      setPredictions(response.data)
    } catch (err) {
      console.error('Failed to fetch Singapore predictions:', err)
      setPredictions(null)
    }
  }

  const getDriverFlag = (driver: string) => {
    const flags: Record<string, string> = {
      'George Russell': '🇬🇧',
      'Lando Norris': '🇬🇧', 
      'Max Verstappen': '🇳🇱',
      'Oscar Piastri': '🇦🇺',
      'Charles Leclerc': '🇲🇨',
      'Lewis Hamilton': '🇬🇧',
      'Fernando Alonso': '🇪🇸',
      'Carlos Sainz': '🇪🇸',
      'Sergio Pérez': '🇲🇽',
      'Lance Stroll': '🇨🇦'
    }
    return flags[driver] || '🏁'
  }

  const getTeamColor = (team: string) => {
    const colors: Record<string, string> = {
      'Mercedes': 'bg-gray-600',
      'McLaren': 'bg-orange-500',
      'Red Bull': 'bg-blue-600',
      'Ferrari': 'bg-red-600',
      'Aston Martin': 'bg-green-600'
    }
    return colors[team] || 'bg-gray-500'
  }

  return (
    <div className="bg-gray-800/60 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2 flex items-center">
          🇸🇬 Singapore GP 2025
          <span className="ml-3 px-3 py-1 bg-red-600 text-white text-sm rounded-full">LIVE</span>
        </h2>
        <p className="text-gray-400">Real-time predictions for tomorrow&apos;s race at Marina Bay</p>
      </div>

      {/* Event Info */}
      {info && (
        <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="flex items-center text-gray-300 mb-2">
              <CalendarIcon className="w-5 h-5 mr-2" />
              Race Date
            </div>
            <div className="text-white font-medium">{info.date}</div>
          </div>
          
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="flex items-center text-gray-300 mb-2">
              <MapPinIcon className="w-5 h-5 mr-2" />
              Circuit
            </div>
            <div className="text-white font-medium">{info.circuit}</div>
          </div>
          
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="flex items-center text-gray-300 mb-2">
              <TrophyIcon className="w-5 h-5 mr-2" />
              Pole Position
            </div>
            <div className="text-white font-medium">
              {getDriverFlag(info.pole_sitter)} {info.pole_sitter}
              <div className="text-sm text-gray-400">{info.pole_time}</div>
            </div>
          </div>
        </div>
      )}

      {/* Weather Info */}
      {info && (
        <div className="mb-6 bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-white font-medium mb-3 flex items-center">
            🌡️ Race Day Weather
          </h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Temperature:</span>
              <div className="text-white font-medium">{info.weather.temperature}</div>
            </div>
            <div>
              <span className="text-gray-400">Humidity:</span>
              <div className="text-white font-medium">{info.weather.humidity}</div>
            </div>
            <div>
              <span className="text-gray-400">Rain Chance:</span>
              <div className="text-white font-medium">{info.weather.rain_probability}</div>
            </div>
          </div>
        </div>
      )}

      {/* Prediction Button */}
      <div className="text-center mb-6">
                <button
          onClick={fetchQuickPrediction}
          className="bg-red-600 hover:bg-red-700 text-white px-8 py-3 rounded-lg font-medium transition-all flex items-center mx-auto"
        >
          <TrophyIcon className="h-5 w-5 mr-2" />
          View Race Results
        </button>
      </div>



      {/* Race Results vs Predictions */}
      {predictions && (
        <div>
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            � Singapore GP 2025 - Results vs Predictions
          </h3>
          
          {/* Race Status */}
          {predictions.status === "RACE_COMPLETED" && (
            <div className="mb-4 bg-green-900/30 border border-green-500 rounded-lg p-4">
              <div className="text-green-300 text-sm mb-1">Race Completed</div>
              <div className="text-white text-lg font-semibold">
                🏆 Winner: {getDriverFlag(predictions.actual_podium?.[0]?.driver || "")} {predictions.actual_podium?.[0]?.driver} ({predictions.actual_podium?.[0]?.team})
              </div>
            </div>
          )}

          {/* Prediction vs Actual Comparison */}
          {predictions.prediction_vs_actual && (
            <div className="mb-6 bg-red-900/30 border border-red-500 rounded-lg p-4">
              <div className="text-red-300 text-sm mb-2">Our Prediction vs Reality</div>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="text-gray-400 text-xs">OUR PREDICTION</div>
                  <div className="text-white font-semibold">
                    🎯 {getDriverFlag(predictions.prediction_vs_actual.our_prediction.winner)} {predictions.prediction_vs_actual.our_prediction.winner}
                  </div>
                  <div className="text-sm text-gray-300">{predictions.prediction_vs_actual.our_prediction.probability} probability</div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">ACTUAL RESULT</div>
                  <div className="text-white font-semibold">
                    🏆 {getDriverFlag(predictions.prediction_vs_actual.actual_result.winner)} {predictions.prediction_vs_actual.actual_result.winner}
                  </div>
                  <div className="text-sm text-gray-300">Started P{predictions.prediction_vs_actual.actual_result.grid_start}</div>
                </div>
              </div>
              <div className="mt-3 text-center">
                <span className="px-3 py-1 bg-red-600 text-white text-sm rounded-full">
                  {predictions.prediction_vs_actual?.prediction_accuracy}
                </span>
              </div>
            </div>
          )}

          {/* Actual Podium Results */}
          {predictions.actual_podium && (
            <div className="space-y-3">
              <h4 className="text-lg font-semibold text-white">🏁 Final Podium</h4>
              {predictions.actual_podium.slice(0, 3).map((result, index) => {
                const isWinner = index === 0
                
                return (
                  <div 
                    key={result.driver}
                    className={`rounded-lg p-4 border transition-all ${
                      isWinner 
                        ? 'border-green-500 bg-green-900/20' 
                        : 'border-gray-600 bg-gray-700/50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div className={`text-2xl font-bold ${
                          isWinner ? 'text-green-400' : 'text-white'
                        }`}>
                          {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                        </div>
                        
                        <div className="flex items-center space-x-3">
                          <div className={`w-4 h-4 rounded-full ${getTeamColor(result.team)}`}></div>
                          <div>
                            <div className="text-white font-medium flex items-center space-x-2">
                              <span>{getDriverFlag(result.driver)}</span>
                              <span>{result.driver}</span>
                              {isWinner && <span className="text-green-400">⭐</span>}
                            </div>
                            <div className="text-gray-400 text-sm">{result.team} • Started P{result.grid}</div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className={`text-xl font-bold ${
                          isWinner ? 'text-green-400' : 'text-white'
                        }`}>
                          P{result.position}
                        </div>
                        <div className="text-gray-400 text-sm">finish</div>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* George Russell Analysis */}
          {predictions.george_russell_result && (
            <div className="mt-6 bg-gray-900/50 border border-gray-600 rounded-lg p-4">
              <h4 className="text-lg font-semibold text-white mb-2">🎯 Pole Sitter Analysis</h4>
              <div className="text-white">
                <span className="text-gray-400">Predicted:</span> {predictions.george_russell_result.predicted}
              </div>
              <div className="text-white">
                <span className="text-gray-400">Actual:</span> {predictions.george_russell_result.actual}
              </div>
            </div>
          )}

          {/* Model Lessons */}
          {predictions.model_lessons && (
            <div className="mt-6 bg-yellow-900/30 border border-yellow-500 rounded-lg p-4">
              <h4 className="text-lg font-semibold text-yellow-300 mb-2">📚 Model Lessons Learned</h4>
              <ul className="space-y-1">
                {predictions.model_lessons.map((lesson, index) => (
                  <li key={index} className="text-yellow-100 text-sm">• {lesson}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-4 text-center text-sm text-gray-400">
            Race completed: October 6, 2025
          </div>
        </div>
      )}
    </div>
  )
}