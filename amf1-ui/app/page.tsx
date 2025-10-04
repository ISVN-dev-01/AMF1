'use client'

import { useState } from 'react'
import Header from './components/Header'
import PredictionForm from './components/PredictionForm'
import PredictionResults from './components/PredictionResults'
import SingaporeGP2025 from './components/SingaporeGP2025'

export default function Home() {
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-red-900 to-gray-900">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            🏎️ <span className="bg-gradient-to-r from-red-500 to-orange-500 bg-clip-text text-transparent">AMF1 Predictor</span>
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Harness the power of machine learning to predict qualifying times and race winners. 
            Get data-driven insights for Formula 1 races with our advanced AI models.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <div className="text-4xl mb-4">⏱️</div>
            <h3 className="text-xl font-semibold text-white mb-2">Qualifying Predictions</h3>
            <p className="text-gray-400">Predict lap times and grid positions based on driver performance, weather conditions, and track characteristics.</p>
          </div>
          
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <div className="text-4xl mb-4">🏆</div>
            <h3 className="text-xl font-semibold text-white mb-2">Race Winner Analysis</h3>
            <p className="text-gray-400">Calculate win probabilities for each driver using advanced ML algorithms and historical race data.</p>
          </div>
          
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <div className="text-4xl mb-4">🌦️</div>
            <h3 className="text-xl font-semibold text-white mb-2">Weather Integration</h3>
            <p className="text-gray-400">Factor in real-time weather conditions including temperature, humidity, and track conditions.</p>
          </div>
        </div>

        {/* Singapore GP 2025 - Special Event */}
        <div className="mb-12">
          <SingaporeGP2025 />
        </div>

        {/* Main Interface */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <div className="bg-gray-800/60 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <PredictionForm 
              onPredictionComplete={setPredictions}
              onLoadingChange={setLoading}
            />
          </div>

          {/* Results Panel */}
          <div className="bg-gray-800/60 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <PredictionResults 
              predictions={predictions}
              loading={loading}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-16 pt-8 border-t border-gray-700">
          <p className="text-gray-400 text-sm">
            Powered by advanced machine learning • Real F1 data • Made with ❤️ for motorsport fans
          </p>
        </div>
      </main>
    </div>
  )
}
