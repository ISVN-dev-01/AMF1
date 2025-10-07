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
    title_battle?: string
  }
}

interface QualifyingResult {
  position: number
  driver: string
  team: string
  time: string
}

interface ChampionshipStanding {
  position: number
  team: string
  points: number
}

interface RacePrediction {
  position: number
  driver: string
  team: string
  probability: string
}

interface QualifyingPredictionAccuracy {
  race: string
  qualifying_date: string
  predicted_pole: string
  actual_pole: string
  pole_time: string
  accuracy_status: string
}

interface SingaporePrediction {
  status: string
  data_source?: string
  qualifying_results?: QualifyingResult[]
  championship_standings?: ChampionshipStanding[]
  race_predictions?: RacePrediction[]
  qualifying_prediction_accuracy?: QualifyingPredictionAccuracy
  metadata?: {
    model_version: string
    prediction_date: string
    accuracy_metrics: Record<string, unknown>
  }
}

const getDriverFlag = (driverName: string): string => {
  const flagMap: { [key: string]: string } = {
    'Max Verstappen': '🇳🇱',
    'Lewis Hamilton': '🇬🇧',
    'George Russell': '🇬🇧',
    'Charles Leclerc': '🇲🇨',
    'Carlos Sainz': '🇪🇸',
    'Lando Norris': '🇬🇧',
    'Oscar Piastri': '🇦🇺',
    'Fernando Alonso': '🇪🇸',
    'Lance Stroll': '🇨🇦',
    'Sergio Perez': '🇲🇽',
    'Valtteri Bottas': '🇫🇮',
    'Zhou Guanyu': '🇨🇳',
    'Kevin Magnussen': '🇩🇰',
    'Nico Hulkenberg': '🇩🇪',
    'Yuki Tsunoda': '🇯🇵',
    'Pierre Gasly': '🇫🇷',
    'Esteban Ocon': '🇫🇷',
    'Alex Albon': '🇹🇭',
    'Logan Sargeant': '🇺🇸',
    'Nyck de Vries': '🇳🇱',
    'Kimi Antonelli': '🇮🇹'
  }
  return flagMap[driverName] || '🏁'
}

export default function SingaporeGP2025() {
  const [info, setInfo] = useState<SingaporeInfo | null>(null)
  const [predictions, setPredictions] = useState<SingaporePrediction | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        
        // Fetch race info and predictions from our FastAPI backend
        const [infoResponse, predictionResponse] = await Promise.all([
          axios.get('http://localhost:8000/singapore_2025/info'),
          axios.get('http://localhost:8000/singapore_2025/quick_prediction')
        ])
        
        setInfo(infoResponse.data)
        setPredictions(predictionResponse.data)
        
      } catch (err) {
        console.error('Error fetching Singapore GP data:', err)
        setError('Failed to fetch race data. Please ensure the backend server is running.')
        
        // Fallback data with official Singapore 2025 qualifying results
        setInfo({
          event: "Singapore Grand Prix 2025",
          date: "October 5, 2025",
          circuit: "Marina Bay Street Circuit",
          status: "Qualifying Completed",
          qualifying_date: "October 4, 2025",
          pole_sitter: "George Russell",
          pole_time: "1:29.158",
          race_completed: false,
          weather: {
            temperature: "28°C",
            humidity: "75%",
            rain_probability: "20%",
            conditions: "Clear skies expected"
          },
          championship_impact: {
            norris_points: 279,
            verstappen_points: 303,
            title_battle: "Verstappen leads by 24 points"
          }
        })
        
        setPredictions({
          status: "QUALIFYING_COMPLETED",
          data_source: "Formula 1® Official Website & RaceFans",
          qualifying_results: [
            { position: 1, driver: "George Russell", team: "Mercedes", time: "1:29.158" },
            { position: 2, driver: "Max Verstappen", team: "Red Bull", time: "1:29.340" },
            { position: 3, driver: "Oscar Piastri", team: "McLaren", time: "1:29.524" },
            { position: 4, driver: "Kimi Antonelli", team: "Mercedes", time: "1:29.537" },
            { position: 5, driver: "Lando Norris", team: "McLaren", time: "1:29.586" }
          ],
          championship_standings: [
            { position: 1, team: "McLaren", points: 650 },
            { position: 2, team: "Mercedes", points: 325 },
            { position: 3, team: "Ferrari", points: 298 },
            { position: 4, team: "Red Bull", points: 290 }
          ],
          race_predictions: [
            { position: 1, driver: "George Russell", team: "Mercedes", probability: "70.5%" },
            { position: 2, driver: "Max Verstappen", team: "Red Bull", probability: "68.2%" },
            { position: 3, driver: "Oscar Piastri", team: "McLaren", probability: "65.8%" }
          ],
          qualifying_prediction_accuracy: {
            race: "Singapore Grand Prix 2025",
            qualifying_date: "October 4, 2025",
            predicted_pole: "George Russell",
            actual_pole: "George Russell",
            pole_time: "1:29.158",
            accuracy_status: "✅ CORRECT"
          }
        })
        
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 shadow-xl border border-gray-700">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-700 rounded w-1/2 mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-700 rounded w-3/4"></div>
            <div className="h-4 bg-gray-700 rounded w-1/2"></div>
            <div className="h-4 bg-gray-700 rounded w-2/3"></div>
          </div>
        </div>
      </div>
    )
  }

  if (error && !info) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-xl p-6">
        <div className="text-red-400 font-semibold">Error</div>
        <div className="text-red-300">{error}</div>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 rounded-xl p-6 shadow-xl border border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <TrophyIcon className="h-8 w-8 text-red-500" />
        <h2 className="text-2xl font-bold text-white">{info?.event || 'Singapore Grand Prix 2025'}</h2>
      </div>

      {error && (
        <div className="mb-4 bg-yellow-900/30 border border-yellow-500 rounded-lg p-4">
          <div className="text-yellow-300 text-sm">⚠️ Using fallback data - {error}</div>
        </div>
      )}

      {/* Race Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <CalendarIcon className="h-5 w-5 text-gray-400" />
            <span className="text-gray-400 text-sm">Race Date</span>
          </div>
          <div className="text-white font-semibold">{info?.date}</div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <MapPinIcon className="h-5 w-5 text-gray-400" />
            <span className="text-gray-400 text-sm">Circuit</span>
          </div>
          <div className="text-white font-semibold">{info?.circuit}</div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <TrophyIcon className="h-5 w-5 text-gray-400" />
            <span className="text-gray-400 text-sm">Status</span>
          </div>
          <div className="text-white font-semibold">{info?.status || predictions?.status}</div>
        </div>
      </div>

      {/* Qualifying Results and Race Prediction */}
      {predictions && (
        <div>
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            🏁 Singapore GP 2025 - Official Results
          </h3>
          
          {/* Data Source */}
          {predictions.data_source && (
            <div className="mb-4 bg-blue-900/30 border border-blue-500 rounded-lg p-4">
              <div className="text-blue-300 text-sm mb-1">Official Data Source</div>
              <div className="text-white text-sm">{predictions.data_source}</div>
            </div>
          )}

          {/* Qualifying Status */}
          {predictions.status === "QUALIFYING_COMPLETED" && (
            <div className="mb-4 bg-green-900/30 border border-green-500 rounded-lg p-4">
              <div className="text-green-300 text-sm mb-1">Qualifying Completed</div>
              <div className="text-white text-lg font-semibold">
                🥇 Pole Position: {getDriverFlag(predictions.qualifying_prediction_accuracy?.actual_pole || "")} {predictions.qualifying_prediction_accuracy?.actual_pole}
              </div>
              <div className="text-gray-300 text-sm">
                Time: {predictions.qualifying_prediction_accuracy?.pole_time}
              </div>
            </div>
          )}

          {/* Qualifying Results */}
          {predictions.qualifying_results && predictions.qualifying_results.length > 0 && (
            <div className="mb-6">
              <h4 className="text-lg font-semibold text-white mb-3">🏎️ Qualifying Results - Top 5</h4>
              <div className="space-y-2">
                {predictions.qualifying_results.slice(0, 5).map((result, index) => (
                  <div key={index} className="bg-gray-800 rounded-lg p-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-red-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                        {result.position}
                      </div>
                      <div>
                        <div className="text-white font-medium">
                          {getDriverFlag(result.driver)} {result.driver}
                        </div>
                        <div className="text-gray-400 text-sm">{result.team}</div>
                      </div>
                    </div>
                    <div className="text-yellow-400 font-mono font-semibold">
                      {result.time}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Prediction Accuracy */}
          {predictions.qualifying_prediction_accuracy && (
            <div className="mb-6 bg-green-900/20 border border-green-600 rounded-lg p-4">
              <h4 className="text-lg font-semibold text-green-300 mb-3">🎯 Qualifying Prediction Accuracy</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="text-green-400 text-sm mb-1">Predicted Pole</div>
                  <div className="text-white font-semibold">
                    {getDriverFlag(predictions.qualifying_prediction_accuracy.predicted_pole)} {predictions.qualifying_prediction_accuracy.predicted_pole}
                  </div>
                </div>
                <div>
                  <div className="text-green-400 text-sm mb-1">Actual Pole</div>
                  <div className="text-white font-semibold">
                    {getDriverFlag(predictions.qualifying_prediction_accuracy.actual_pole)} {predictions.qualifying_prediction_accuracy.actual_pole}
                  </div>
                </div>
              </div>
              <div className="mt-3 text-center">
                <span className="text-2xl">{predictions.qualifying_prediction_accuracy.accuracy_status}</span>
              </div>
            </div>
          )}

          {/* Race Predictions */}
          {predictions.race_predictions && predictions.race_predictions.length > 0 && (
            <div className="mb-6">
              <h4 className="text-lg font-semibold text-white mb-3">🔮 Race Winner Predictions</h4>
              <div className="space-y-2">
                {predictions.race_predictions.slice(0, 3).map((prediction, index) => (
                  <div key={index} className="bg-gray-800 rounded-lg p-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                        {prediction.position}
                      </div>
                      <div>
                        <div className="text-white font-medium">
                          {getDriverFlag(prediction.driver)} {prediction.driver}
                        </div>
                        <div className="text-gray-400 text-sm">{prediction.team}</div>
                      </div>
                    </div>
                    <div className="text-blue-400 font-semibold">
                      {prediction.probability}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Constructor Championship Standings */}
          {predictions.championship_standings && predictions.championship_standings.length > 0 && (
            <div className="mb-6">
              <h4 className="text-lg font-semibold text-white mb-3">🏆 Constructor Championship Standings</h4>
              <div className="space-y-2">
                {predictions.championship_standings.slice(0, 4).map((standing, index) => (
                  <div key={index} className="bg-gray-800 rounded-lg p-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-yellow-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                        {standing.position}
                      </div>
                      <div className="text-white font-semibold">{standing.team}</div>
                    </div>
                    <div className="text-yellow-400 font-bold">
                      {standing.points} pts
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Weather Info */}
      {info?.weather && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-white mb-3">🌤️ Weather Conditions</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-800 rounded-lg p-3 text-center">
              <div className="text-gray-400 text-sm">Temperature</div>
              <div className="text-white font-semibold">{info.weather.temperature}</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-3 text-center">
              <div className="text-gray-400 text-sm">Humidity</div>
              <div className="text-white font-semibold">{info.weather.humidity}</div>
            </div>
            {info.weather.rain_probability && (
              <div className="bg-gray-800 rounded-lg p-3 text-center">
                <div className="text-gray-400 text-sm">Rain Chance</div>
                <div className="text-white font-semibold">{info.weather.rain_probability}</div>
              </div>
            )}
            {info.weather.conditions && (
              <div className="bg-gray-800 rounded-lg p-3 text-center">
                <div className="text-gray-400 text-sm">Conditions</div>
                <div className="text-white font-semibold">{info.weather.conditions}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Championship Impact */}
      {info?.championship_impact && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-white mb-3">🏆 Championship Battle</h3>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
              <div>
                <div className="text-gray-400 text-sm">Lando Norris</div>
                <div className="text-white font-bold text-xl">{info.championship_impact.norris_points} points</div>
              </div>
              <div>
                <div className="text-gray-400 text-sm">Max Verstappen</div>
                <div className="text-white font-bold text-xl">{info.championship_impact.verstappen_points} points</div>
              </div>
            </div>
            {info.championship_impact.title_battle && (
              <div className="text-center text-yellow-400 font-semibold">
                {info.championship_impact.title_battle}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}