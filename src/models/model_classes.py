#!/usr/bin/env python3
"""
Model definitions for F1 ML pipeline
Pickleable ensemble classes for production deployment
"""

import numpy as np

class Stage1Ensemble:
    """Ensemble wrapper for Stage-1 qualifying time prediction"""
    
    def __init__(self, models):
        self.models = models
        
    def predict(self, X):
        predictions = []
        for name, model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        # Average ensemble predictions
        return np.mean(predictions, axis=0)
    
    def get_feature_importance(self):
        # Average feature importance across models
        importances = []
        for name, model in self.models:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
        if importances:
            return np.mean(importances, axis=0)
        return None

class Stage2Ensemble:
    """Ensemble wrapper for Stage-2 race winner prediction"""
    
    def __init__(self, models):
        self.models = models
        
    def predict_proba(self, X):
        probabilities = []
        for name, model in self.models:
            prob = model.predict_proba(X)
            probabilities.append(prob)
        # Average ensemble probabilities
        return np.mean(probabilities, axis=0)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)