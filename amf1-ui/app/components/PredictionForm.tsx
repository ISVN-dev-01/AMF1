'use client'

import { useState } from 'react'

import { PlusIcon, MinusIcon, PlayIcon } from '@heroicons/react/24/outline'
import axios from 'axios'

// Real F1 driver data for 2024 season
const F1_DRIVERS = {
  1: { name: "Max Verstappen", team: "Red Bull Racing", color: "bg-blue-600" },
  11: { name: "Sergio Pérez", team: "Red Bull Racing", color: "bg-blue-600" },
  16: { name: "Charles Leclerc", team: "Ferrari", color: "bg-red-600" },
  55: { name: "Carlos Sainz", team: "Ferrari", color: "bg-red-600" },
  44: { name: "Lewis Hamilton", team: "Mercedes", color: "bg-gray-600" },
  63: { name: "George Russell", team: "Mercedes", color: "bg-gray-600" },
  4: { name: "Lando Norris", team: "McLaren", color: "bg-orange-500" },
  81: { name: "Oscar Piastri", team: "McLaren", color: "bg-orange-500" },
  14: { name: "Fernando Alonso", team: "Aston Martin", color: "bg-green-600" },
  18: { name: "Lance Stroll", team: "Aston Martin", color: "bg-green-600" }
}



interface Driver {
  driver_id: number
  weather_condition: string
  temperature: number
  humidity: number
  rain_probability: number
}

interface PredictionFormProps {
  onPredictions: (data: unknown) => void
  loading: boolean
  setLoading: (loading: boolean) => void
}

export default function PredictionForm({ onPredictions, loading, setLoading }: PredictionFormProps) {
  const [drivers, setDrivers] = useState<Driver[]>([
    { driver_id: 44, weather_condition: 'dry', temperature: 25, humidity: 50, rain_probability: 0 },
    { driver_id: 1, weather_condition: 'dry', temperature: 25, humidity: 50, rain_probability: 0 }
  ])
  const [predictionType, setPredictionType] = useState<'quali' | 'race' | 'full'>('full')

  const addDriver = () => {
    const availableDrivers = Object.keys(F1_DRIVERS).map(Number).filter(
      id => !drivers.some(d => d.driver_id === id)
    )
    
    if (availableDrivers.length > 0) {
      setDrivers([...drivers, {
        driver_id: availableDrivers[0],
        weather_condition: 'dry',
        temperature: 25,
        humidity: 50,
        rain_probability: 0
      }])
    }
  }

  const removeDriver = (index: number) => {
    if (drivers.length > 1) {
      setDrivers(drivers.filter((_, i) => i !== index))
    }
  }

  const updateDriver = (index: number, field: keyof Driver, value: string | number) => {
    const updatedDrivers = [...drivers]
    updatedDrivers[index] = { ...updatedDrivers[index], [field]: value }
    setDrivers(updatedDrivers)
  }

  const handleSubmit = async () => {
    setLoading(true)
    
    try {
      const endpoint = predictionType === 'quali' ? 'predict_quali' : 
                     predictionType === 'race' ? 'predict_race' : 'predict_full'
      
      const response = await axios.post(`http://localhost:8000/${endpoint}`, {
        drivers: drivers
      })
      
      onPredictions(response.data)
    } catch (error: unknown) {
      console.error('Prediction failed:', error)
      onPredictions({
        error: error instanceof Error ? error.message : 'Failed to get predictions'
      })
    } finally {
      setLoading(false)
    }
  }

  const generateRandomData = () => {
    const weather_conditions = ['dry', 'wet', 'mixed', 'cloudy']
    const randomDrivers = drivers.map(driver => ({
      ...driver,
      weather_condition: weather_conditions[Math.floor(Math.random() * weather_conditions.length)],
      temperature: Math.round(Math.random() * 15 + 15), // 15-30°C
      humidity: Math.round(Math.random() * 50 + 30), // 30-80%
      rain_probability: Math.round(Math.random() * 100)
    }))
    setDrivers(randomDrivers)
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-white">🏁 Prediction Setup</h2>
        <button
          onClick={generateRandomData}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
        >
          🎲 Random Data
        </button>
      </div>

      {/* Prediction Type Selection */}
      <div className="mb-6">
        <label className="block text-white font-medium mb-3">Prediction Type</label>
        <div className="grid grid-cols-3 gap-2">
          {[
            { key: 'quali', label: '🏁 Qualifying', desc: 'Lap times' },
            { key: 'race', label: '🏆 Race Winner', desc: 'Win probability' },
            { key: 'full', label: '🎯 Full Analysis', desc: 'Both predictions' }
          ].map((type) => (
            <button
              key={type.key}
              onClick={() => setPredictionType(type.key as 'quali' | 'race' | 'full')}
              className={`p-3 rounded-lg text-center transition-colors ${
                predictionType === type.key
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <div className="font-medium text-sm">{type.label}</div>
              <div className="text-xs opacity-75">{type.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Drivers */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <label className="text-white font-medium">Drivers ({drivers.length})</label>
          <button
            onClick={addDriver}
            disabled={drivers.length >= 10}
            className="flex items-center space-x-2 px-3 py-1 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          >
            <PlusIcon className="w-4 h-4" />
            <span>Add Driver</span>
          </button>
        </div>

        <div className="space-y-4 max-h-80 overflow-y-auto">
          {drivers.map((driver, index) => {
            const driverInfo = F1_DRIVERS[driver.driver_id as keyof typeof F1_DRIVERS]
            return (
              <div key={index} className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${driverInfo?.color || 'bg-gray-500'}`}></div>
                    <select
                      value={driver.driver_id}
                      onChange={(e) => updateDriver(index, 'driver_id', Number(e.target.value))}
                      className="bg-gray-800 text-white rounded px-3 py-1 text-sm"
                    >
                      {Object.entries(F1_DRIVERS).map(([id, info]) => (
                        <option key={id} value={id}>
                          {info.name} ({info.team})
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {drivers.length > 1 && (
                    <button
                      onClick={() => removeDriver(index)}
                      className="text-red-400 hover:text-red-300 transition-colors"
                    >
                      <MinusIcon className="w-4 h-4" />
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-gray-300 text-xs mb-1">Weather</label>
                    <select
                      value={driver.weather_condition}
                      onChange={(e) => updateDriver(index, 'weather_condition', e.target.value)}
                      className="w-full bg-gray-800 text-white rounded px-2 py-1 text-sm"
                    >
                      <option value="dry">☀️ Dry</option>
                      <option value="wet">🌧️ Wet</option>
                      <option value="mixed">⛅ Mixed</option>
                      <option value="cloudy">☁️ Cloudy</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-xs mb-1">Temperature (°C)</label>
                    <input
                      type="range"
                      min="10"
                      max="40"
                      value={driver.temperature}
                      onChange={(e) => updateDriver(index, 'temperature', Number(e.target.value))}
                      className="w-full"
                    />
                    <div className="text-center text-xs text-gray-400">{driver.temperature}°C</div>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-xs mb-1">Humidity (%)</label>
                    <input
                      type="range"
                      min="20"
                      max="90"
                      value={driver.humidity}
                      onChange={(e) => updateDriver(index, 'humidity', Number(e.target.value))}
                      className="w-full"
                    />
                    <div className="text-center text-xs text-gray-400">{driver.humidity}%</div>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-xs mb-1">Rain Chance (%)</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={driver.rain_probability}
                      onChange={(e) => updateDriver(index, 'rain_probability', Number(e.target.value))}
                      className="w-full"
                    />
                    <div className="text-center text-xs text-gray-400">{driver.rain_probability}%</div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        disabled={loading || drivers.length === 0}
        className="w-full flex items-center justify-center space-x-3 bg-gradient-to-r from-red-600 to-red-700 text-white py-4 rounded-lg hover:from-red-700 hover:to-red-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium"
      >
        {loading ? (
          <>
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
            <span>Making Predictions...</span>
          </>
        ) : (
          <>
            <PlayIcon className="w-5 h-5" />
            <span>Get Predictions</span>
          </>
        )}
      </button>
    </div>
  )
}