'use client'

import { useState } from 'react'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <header className="bg-black/50 backdrop-blur-sm border-b border-gray-800">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="text-2xl">🏎️</div>
            <div>
              <h1 className="text-white font-bold text-xl">F1 Predictor</h1>
              <p className="text-gray-400 text-xs">ML Powered</p>
            </div>
          </div>

          <nav className="hidden md:flex items-center space-x-8">
            <a href="#predictions" className="text-gray-300 hover:text-white transition-colors">
              Predictions
            </a>
            <a href="#about" className="text-gray-300 hover:text-white transition-colors">
              About
            </a>
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>API Connected</span>
            </div>
          </nav>

          <button
            className="md:hidden text-gray-300 hover:text-white"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? (
              <XMarkIcon className="w-6 h-6" />
            ) : (
              <Bars3Icon className="w-6 h-6" />
            )}
          </button>
        </div>

        {isMenuOpen && (
          <div className="md:hidden border-t border-gray-800 py-4">
            <nav className="flex flex-col space-y-4">
              <a 
                href="#predictions" 
                className="text-gray-300 hover:text-white transition-colors"
                onClick={() => setIsMenuOpen(false)}
              >
                Predictions
              </a>
              <a 
                href="#about" 
                className="text-gray-300 hover:text-white transition-colors"
                onClick={() => setIsMenuOpen(false)}
              >
                About
              </a>
            </nav>
          </div>
        )}
      </div>
    </header>
  )
}