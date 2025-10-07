'use client'

import { TrophyIcon, ClockIcon, ChartBarIcon } from '@heroicons/react/24/outline'

// F1 driver data for display
const F1_DRIVERS = {
  1: { name: "Max Verstappen", team: "Red Bull Racing", color: "bg-blue-600", flag: "🇳🇱" },
  11: { name: "Sergio Pérez", team: "Red Bull Racing", color: "bg-blue-600", flag: "🇲🇽" },
  16: { name: "Charles Leclerc", team: "Ferrari", color: "bg-red-600", flag: "🇲🇨" },
  55: { name: "Carlos Sainz", team: "Ferrari", color: "bg-red-600", flag: "🇪🇸" },
  44: { name: "Lewis Hamilton", team: "Mercedes", color: "bg-gray-600", flag: "🇬🇧" },
  63: { name: "George Russell", team: "Mercedes", color: "bg-gray-600", flag: "🇬🇧" },
  4: { name: "Lando Norris", team: "McLaren", color: "bg-orange-500", flag: "🇬🇧" },
  81: { name: "Oscar Piastri", team: "McLaren", color: "bg-orange-500", flag: "🇦🇺" },
  14: { name: "Fernando Alonso", team: "Aston Martin", color: "bg-green-600", flag: "🇪🇸" },
  18: { name: "Lance Stroll", team: "Aston Martin", color: "bg-green-600", flag: "🇨🇦" }
}

interface QualifyingPrediction {
  driver_id: number
  predicted_time: number
  probability_score: number
  grid_position_estimate: number
}

interface RaceWinnerPrediction {
  driver_id: number
  win_probability: number
  confidence_score: number
  ranking: number
}

interface PredictionData {
  qualifying_predictions?: QualifyingPrediction[]
  race_winner_predictions?: RaceWinnerPrediction[]
  error?: string
  metadata?: {
    prediction_time?: string
    model_version?: string
    confidence_level?: string
  }
}

interface PredictionResultsProps {
  predictions: PredictionData | null
  loading: boolean
}

export default function PredictionResults({ predictions, loading }: PredictionResultsProps) {
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(3)
    return `${minutes}:${secs.padStart(6, '0')}`
  }

  const getDriverInfo = (driverId: number) => {
    return F1_DRIVERS[driverId as keyof typeof F1_DRIVERS] || {
      name: `Driver ${driverId}`,
      team: "Unknown Team",
      color: "bg-gray-500",
      flag: "🏁"
    }
  }

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-500 mx-auto mb-4"></div>
        <h3 className="text-white font-medium mb-2">Processing Predictions...</h3>
        <p className="text-gray-400 text-sm">Our ML models are analyzing the data</p>
      </div>
    )
  }

  if (!predictions) {
    return (
      <div className="text-center py-12">
        <ChartBarIcon className="w-16 h-16 text-gray-500 mx-auto mb-4" />
        <h3 className="text-white font-medium mb-2">Ready for Predictions</h3>
        <p className="text-gray-400 text-sm">Configure drivers and conditions, then hit &quot;Get Predictions&quot;</p>
      </div>
    )
  }

  if (predictions.error) {
    return (
      <div className="text-center py-12">
        <div className="bg-red-900/50 border border-red-600 rounded-lg p-6">
          <div className="text-red-400 text-4xl mb-4">⚠️</div>
          <h3 className="text-white font-medium mb-2">Prediction Failed</h3>
          <p className="text-red-300 text-sm">{predictions.error}</p>
          <div className="mt-4 text-xs text-gray-400">
            Make sure the F1 ML API is running on localhost:8000
          </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
        <ChartBarIcon className="w-8 h-8 mr-3" />
        Results
      </h2>

      {/* Metadata */}
      {predictions.metadata && (
        <div className="mb-6 bg-gray-700/30 rounded-lg p-4 border border-gray-600">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            {predictions.metadata.model_version && (
              <div>
                <span className="text-gray-400">Model Version:</span>
                <div className="text-white font-medium">{predictions.metadata.model_version}</div>
              </div>
            )}
            {predictions.metadata.confidence_level && (
              <div>
                <span className="text-gray-400">Confidence:</span>
                <div className="text-white font-medium capitalize">{predictions.metadata.confidence_level}</div>
              </div>
            )}
            {predictions.metadata.prediction_time && (
              <div>
                <span className="text-gray-400">Generated:</span>
                <div className="text-white font-medium">
                  {new Date(predictions.metadata.prediction_time).toLocaleTimeString()}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Qualifying Results */}
      {predictions.qualifying_predictions && predictions.qualifying_predictions.length > 0 && (
        <div className="mb-8">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <ClockIcon className="w-6 h-6 mr-2" />
            🏁 Qualifying Predictions
          </h3>
          
          <div className="space-y-3">
            {predictions.qualifying_predictions
              .sort((a, b) => a.predicted_time - b.predicted_time)
              .map((pred, index) => {
                const driver = getDriverInfo(pred.driver_id)
                const isPoleSitter = index === 0
                
                return (
                  <div 
                    key={pred.driver_id}
                    className={`bg-gray-700/50 rounded-lg p-4 border transition-all ${
                      isPoleSitter 
                        ? 'border-yellow-500 bg-yellow-900/20' 
                        : 'border-gray-600 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div className={`text-2xl font-bold ${
                          isPoleSitter ? 'text-yellow-400' : 'text-white'
                        }`}>
                          P{index + 1}
                        </div>
                        
                        <div className="flex items-center space-x-3">
                          <div className={`w-4 h-4 rounded-full ${driver.color}`}></div>
                          <div>
                            <div className="text-white font-medium flex items-center space-x-2">
                              <span>{driver.flag}</span>
                              <span>{driver.name}</span>
                              {isPoleSitter && <span className="text-yellow-400">👑</span>}
                            </div>
                            <div className="text-gray-400 text-sm">{driver.team}</div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className={`text-xl font-mono ${
                          isPoleSitter ? 'text-yellow-400' : 'text-white'
                        }`}>
                          {formatTime(pred.predicted_time)}
                        </div>
                        <div className="text-gray-400 text-sm">
                          {(pred.probability_score * 100).toFixed(1)}% confidence
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {/* Race Winner Results */}
      {predictions.race_winner_predictions && predictions.race_winner_predictions.length > 0 && (
        <div>
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <TrophyIcon className="w-6 h-6 mr-2" />
            🏆 Race Winner Predictions
          </h3>
          
          <div className="space-y-3">
            {predictions.race_winner_predictions
              .sort((a, b) => b.win_probability - a.win_probability)
              .map((pred, index) => {
                const driver = getDriverInfo(pred.driver_id)
                const winPercentage = (pred.win_probability * 100)
                const isTopFavorite = index === 0
                
                return (
                  <div 
                    key={pred.driver_id}
                    className={`bg-gray-700/50 rounded-lg p-4 border transition-all ${
                      isTopFavorite 
                        ? 'border-green-500 bg-green-900/20' 
                        : 'border-gray-600 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-4">
                        <div className={`text-lg font-bold ${
                          isTopFavorite ? 'text-green-400' : 'text-white'
                        }`}>
                          #{index + 1}
                        </div>
                        
                        <div className="flex items-center space-x-3">
                          <div className={`w-4 h-4 rounded-full ${driver.color}`}></div>
                          <div>
                            <div className="text-white font-medium flex items-center space-x-2">
                              <span>{driver.flag}</span>
                              <span>{driver.name}</span>
                              {isTopFavorite && <span className="text-green-400">⭐</span>}
                            </div>
                            <div className="text-gray-400 text-sm">{driver.team}</div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className={`text-xl font-bold ${
                          isTopFavorite ? 'text-green-400' : 'text-white'
                        }`}>
                          {winPercentage.toFixed(1)}%
                        </div>
                        <div className="text-gray-400 text-sm">
                          {(pred.confidence_score * 100).toFixed(0)}% confidence
                        </div>
                      </div>
                    </div>
                    
                    {/* Win Probability Bar */}
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          isTopFavorite ? 'bg-green-500' : 'bg-blue-500'
                        }`}
                        style={{ width: `${winPercentage}%` }}
                      ></div>
                    </div>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!predictions.qualifying_predictions?.length && !predictions.race_winner_predictions?.length && !predictions.error && (
        <div className="text-center py-12">
          <div className="text-gray-500 text-4xl mb-4">🤔</div>
          <h3 className="text-white font-medium mb-2">No Predictions Available</h3>
          <p className="text-gray-400 text-sm">The API returned empty results. Try adjusting your parameters.</p>
        </div>
      )}
    </div>
  )
}