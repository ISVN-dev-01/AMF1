#!/usr/bin/env python3
"""
PHASE 11.1: Metrics and Monitoring System
Production monitoring for F1 ML models with Prometheus metrics
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from fastapi import Response
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1ModelMetrics:
    """Prometheus metrics for F1 ML models"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Model performance metrics
        self.prediction_requests = Counter(
            'f1_prediction_requests_total',
            'Total number of prediction requests',
            ['endpoint', 'status'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'f1_prediction_duration_seconds',
            'Time spent on predictions',
            ['endpoint', 'stage'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'f1_model_accuracy',
            'Model accuracy per race',
            ['stage', 'race_id', 'metric_type'],
            registry=self.registry
        )
        
        self.calibration_score = Gauge(
            'f1_model_calibration_brier_score',
            'Brier calibration score per race',
            ['stage', 'race_id'],
            registry=self.registry
        )
        
        # Feature drift metrics
        self.feature_drift = Gauge(
            'f1_feature_drift_kl_divergence',
            'KL divergence of features vs training data',
            ['feature_name', 'race_id'],
            registry=self.registry
        )
        
        # Infrastructure metrics
        self.error_rate = Gauge(
            'f1_api_error_rate',
            'API error rate percentage',
            ['time_window'],
            registry=self.registry
        )
        
        self.model_health = Gauge(
            'f1_model_health_score',
            'Overall model health score (0-1)',
            ['stage'],
            registry=self.registry
        )
        
        # Retraining metrics
        self.last_retrain_timestamp = Gauge(
            'f1_last_retrain_timestamp',
            'Timestamp of last model retraining',
            ['stage'],
            registry=self.registry
        )
        
        self.retrain_trigger = Counter(
            'f1_retrain_triggers_total',
            'Number of retrain triggers',
            ['trigger_type', 'stage'],
            registry=self.registry
        )
        
    def record_prediction_request(self, endpoint: str, status: str):
        """Record a prediction request"""
        self.prediction_requests.labels(endpoint=endpoint, status=status).inc()
    
    def record_prediction_latency(self, endpoint: str, stage: str, duration: float):
        """Record prediction latency"""
        self.prediction_latency.labels(endpoint=endpoint, stage=stage).observe(duration)
    
    def update_model_accuracy(self, stage: str, race_id: str, metric_type: str, accuracy: float):
        """Update model accuracy metrics"""
        self.model_accuracy.labels(stage=stage, race_id=race_id, metric_type=metric_type).set(accuracy)
    
    def update_calibration_score(self, stage: str, race_id: str, brier_score: float):
        """Update calibration Brier score"""
        self.calibration_score.labels(stage=stage, race_id=race_id).set(brier_score)
    
    def update_feature_drift(self, feature_name: str, race_id: str, kl_divergence: float):
        """Update feature drift KL divergence"""
        self.feature_drift.labels(feature_name=feature_name, race_id=race_id).set(kl_divergence)
    
    def update_error_rate(self, time_window: str, error_rate: float):
        """Update API error rate"""
        self.error_rate.labels(time_window=time_window).set(error_rate)
    
    def update_model_health(self, stage: str, health_score: float):
        """Update overall model health score"""
        self.model_health.labels(stage=stage).set(health_score)
    
    def record_retrain_trigger(self, trigger_type: str, stage: str):
        """Record retrain trigger event"""
        self.retrain_trigger.labels(trigger_type=trigger_type, stage=stage).inc()
        self.last_retrain_timestamp.labels(stage=stage).set(time.time())
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics format"""
        return generate_latest(self.registry).decode('utf-8')

class F1MetricsDatabase:
    """SQLite database for storing F1 model metrics and results"""
    
    def __init__(self, db_path: str = "data/metrics/f1_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS race_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT NOT NULL,
                    race_date DATE NOT NULL,
                    driver_id INTEGER NOT NULL,
                    actual_quali_time REAL,
                    actual_race_winner INTEGER,
                    predicted_quali_time REAL,
                    predicted_win_probability REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT NOT NULL,
                    race_date DATE NOT NULL,
                    stage TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_drift (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT NOT NULL,
                    race_date DATE NOT NULL,
                    feature_name TEXT NOT NULL,
                    kl_divergence REAL NOT NULL,
                    drift_threshold REAL DEFAULT 0.1,
                    is_drifted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retrain_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_type TEXT NOT NULL,
                    trigger_reason TEXT,
                    model_version_old TEXT,
                    model_version_new TEXT,
                    performance_improvement REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("✅ Metrics database initialized")
    
    def store_race_results(self, race_results: List[Dict]):
        """Store actual race results for later evaluation"""
        with sqlite3.connect(self.db_path) as conn:
            for result in race_results:
                conn.execute("""
                    INSERT INTO race_results 
                    (race_id, race_date, driver_id, actual_quali_time, actual_race_winner,
                     predicted_quali_time, predicted_win_probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['race_id'], result['race_date'], result['driver_id'],
                    result.get('actual_quali_time'), result.get('actual_race_winner'),
                    result.get('predicted_quali_time'), result.get('predicted_win_probability')
                ))
            conn.commit()
            logger.info(f"✅ Stored {len(race_results)} race results")
    
    def store_performance_metrics(self, race_id: str, race_date: str, metrics: Dict):
        """Store model performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            for stage, stage_metrics in metrics.items():
                for metric_name, metric_value in stage_metrics.items():
                    conn.execute("""
                        INSERT INTO model_performance 
                        (race_id, race_date, stage, metric_name, metric_value)
                        VALUES (?, ?, ?, ?, ?)
                    """, (race_id, race_date, stage, metric_name, metric_value))
            conn.commit()
            logger.info(f"✅ Stored performance metrics for race {race_id}")
    
    def store_feature_drift(self, race_id: str, race_date: str, drift_metrics: Dict):
        """Store feature drift measurements"""
        with sqlite3.connect(self.db_path) as conn:
            for feature_name, kl_div in drift_metrics.items():
                is_drifted = kl_div > 0.1  # Threshold for significant drift
                conn.execute("""
                    INSERT INTO feature_drift 
                    (race_id, race_date, feature_name, kl_divergence, is_drifted)
                    VALUES (?, ?, ?, ?, ?)
                """, (race_id, race_date, feature_name, kl_div, is_drifted))
            conn.commit()
            logger.info(f"✅ Stored feature drift for race {race_id}")
    
    def get_recent_performance(self, days: int = 30) -> pd.DataFrame:
        """Get recent model performance data"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM model_performance 
                WHERE race_date >= date('now', '-{} days')
                ORDER BY race_date DESC
            """.format(days)
            return pd.read_sql_query(query, conn)
    
    def get_drift_summary(self, days: int = 30) -> pd.DataFrame:
        """Get feature drift summary"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT feature_name, AVG(kl_divergence) as avg_drift,
                       MAX(kl_divergence) as max_drift,
                       COUNT(CASE WHEN is_drifted THEN 1 END) as drift_count,
                       COUNT(*) as total_measurements
                FROM feature_drift 
                WHERE race_date >= date('now', '-{} days')
                GROUP BY feature_name
                ORDER BY avg_drift DESC
            """.format(days)
            return pd.read_sql_query(query, conn)

class F1ModelMonitor:
    """Main monitoring class for F1 ML models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.metrics = F1ModelMetrics()
        self.db = F1MetricsDatabase()
        self.training_data_stats = self.load_training_statistics()
    
    def load_training_statistics(self) -> Optional[Dict]:
        """Load training data statistics for drift detection"""
        try:
            # Load training feature statistics
            training_stats_file = self.models_dir / "training_statistics.json"
            if training_stats_file.exists():
                with open(training_stats_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Training statistics not found, computing from data...")
                return self.compute_training_statistics()
        except Exception as e:
            logger.error(f"Failed to load training statistics: {e}")
            return None
    
    def compute_training_statistics(self) -> Dict:
        """Compute statistics from training data"""
        try:
            import joblib
            
            # Load training data
            data = pd.read_parquet('data/features/complete_features.parquet')
            
            # Load feature metadata
            metadata = joblib.load(self.models_dir / 'feature_metadata.pkl')
            feature_cols = metadata['feature_columns']
            
            # Compute statistics
            stats = {}
            for col in feature_cols:
                if col in data.columns:
                    col_data = data[col].dropna()
                    stats[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q25': float(col_data.quantile(0.25)),
                        'q75': float(col_data.quantile(0.75))
                    }
            
            # Save statistics
            stats_file = self.models_dir / "training_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"✅ Computed training statistics for {len(stats)} features")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute training statistics: {e}")
            return {}
    
    def compute_model_accuracy(self, race_id: str, predictions: List[Dict], actuals: List[Dict]) -> Dict:
        """Compute model accuracy metrics"""
        
        accuracy_metrics = {
            'stage1': {},
            'stage2': {}
        }
        
        try:
            # Stage-1: Qualifying time accuracy
            pred_times = [p.get('predicted_quali_time', 0) for p in predictions]
            actual_times = [a.get('actual_quali_time', 0) for a in actuals]
            
            if pred_times and actual_times:
                mae = np.mean(np.abs(np.array(pred_times) - np.array(actual_times)))
                rmse = np.sqrt(np.mean((np.array(pred_times) - np.array(actual_times)) ** 2))
                
                accuracy_metrics['stage1'] = {
                    'mae': mae,
                    'rmse': rmse,
                    'race_id': race_id
                }
            
            # Stage-2: Race winner accuracy
            pred_winners = [1 if p.get('predicted_win_probability', 0) > 0.5 else 0 for p in predictions]
            actual_winners = [a.get('actual_race_winner', 0) for a in actuals]
            
            if pred_winners and actual_winners:
                accuracy = np.mean(np.array(pred_winners) == np.array(actual_winners))
                
                # Top-1 and Top-3 accuracy
                sorted_preds = sorted(predictions, key=lambda x: x.get('predicted_win_probability', 0), reverse=True)
                actual_winner_id = next((a['driver_id'] for a in actuals if a.get('actual_race_winner')), None)
                
                top1_accuracy = 1 if actual_winner_id and sorted_preds[0]['driver_id'] == actual_winner_id else 0
                top3_accuracy = 1 if actual_winner_id and actual_winner_id in [p['driver_id'] for p in sorted_preds[:3]] else 0
                
                # Brier score for calibration
                brier_score = np.mean([
                    (p.get('predicted_win_probability', 0) - a.get('actual_race_winner', 0)) ** 2
                    for p, a in zip(predictions, actuals)
                ])
                
                accuracy_metrics['stage2'] = {
                    'accuracy': accuracy,
                    'top1_accuracy': top1_accuracy,
                    'top3_accuracy': top3_accuracy,
                    'brier_score': brier_score,
                    'race_id': race_id
                }
            
            return accuracy_metrics
            
        except Exception as e:
            logger.error(f"Failed to compute accuracy metrics: {e}")
            return accuracy_metrics
    
    def compute_feature_drift(self, current_features: pd.DataFrame, race_id: str) -> Dict:
        """Compute KL divergence for feature drift detection"""
        
        if not self.training_data_stats:
            logger.warning("No training statistics available for drift detection")
            return {}
        
        drift_metrics = {}
        
        try:
            for feature_name in current_features.columns:
                if feature_name in self.training_data_stats:
                    current_data = current_features[feature_name].dropna()
                    training_stats = self.training_data_stats[feature_name]
                    
                    if len(current_data) > 0:
                        # Simple KL divergence approximation using statistical moments
                        current_mean = current_data.mean()
                        current_std = current_data.std()
                        
                        training_mean = training_stats['mean']
                        training_std = training_stats['std']
                        
                        # Approximate KL divergence for normal distributions
                        if current_std > 0 and training_std > 0:
                            kl_div = np.log(current_std / training_std) + \
                                    (training_std**2 + (training_mean - current_mean)**2) / (2 * current_std**2) - 0.5
                            drift_metrics[feature_name] = max(0, kl_div)  # Ensure non-negative
                        else:
                            drift_metrics[feature_name] = 0.0
            
            logger.info(f"✅ Computed drift for {len(drift_metrics)} features")
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Failed to compute feature drift: {e}")
            return {}
    
    def update_all_metrics(self, race_id: str, race_date: str, predictions: List[Dict], 
                          actuals: List[Dict], current_features: pd.DataFrame):
        """Update all monitoring metrics"""
        
        try:
            # Compute and store accuracy metrics
            accuracy_metrics = self.compute_model_accuracy(race_id, predictions, actuals)
            self.db.store_performance_metrics(race_id, race_date, accuracy_metrics)
            
            # Update Prometheus metrics
            for stage, metrics in accuracy_metrics.items():
                for metric_name, value in metrics.items():
                    if metric_name != 'race_id':
                        self.metrics.update_model_accuracy(stage, race_id, metric_name, value)
                        
                        # Update calibration score specifically
                        if metric_name == 'brier_score':
                            self.metrics.update_calibration_score(stage, race_id, value)
            
            # Compute and store feature drift
            drift_metrics = self.compute_feature_drift(current_features, race_id)
            if drift_metrics:
                self.db.store_feature_drift(race_id, race_date, drift_metrics)
                
                # Update Prometheus drift metrics
                for feature_name, kl_div in drift_metrics.items():
                    self.metrics.update_feature_drift(feature_name, race_id, kl_div)
            
            # Compute overall model health score
            stage1_health = 1.0 - min(accuracy_metrics.get('stage1', {}).get('mae', 1.0) / 10.0, 1.0)
            stage2_health = accuracy_metrics.get('stage2', {}).get('accuracy', 0.0)
            
            self.metrics.update_model_health('stage1', stage1_health)
            self.metrics.update_model_health('stage2', stage2_health)
            
            logger.info(f"✅ Updated all metrics for race {race_id}")
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def check_retrain_triggers(self) -> Dict:
        """Check if retraining should be triggered"""
        
        triggers = {
            'should_retrain': False,
            'reasons': [],
            'trigger_type': None
        }
        
        try:
            # Get recent performance
            recent_perf = self.db.get_recent_performance(days=30)
            
            if len(recent_perf) < 3:
                logger.info("Insufficient data for retrain trigger evaluation")
                return triggers
            
            # Performance-based triggers
            stage2_accuracy = recent_perf[
                (recent_perf['stage'] == 'stage2') & 
                (recent_perf['metric_name'] == 'top1_accuracy')
            ]['metric_value']
            
            if len(stage2_accuracy) >= 3:
                recent_avg = stage2_accuracy.tail(3).mean()
                if recent_avg < 0.3:  # Threshold for poor performance
                    triggers['should_retrain'] = True
                    triggers['reasons'].append(f"Stage-2 Top-1 accuracy dropped to {recent_avg:.3f}")
                    triggers['trigger_type'] = 'performance'
                    self.metrics.record_retrain_trigger('performance_degradation', 'stage2')
            
            # Drift-based triggers
            drift_summary = self.db.get_drift_summary(days=14)
            high_drift_features = drift_summary[drift_summary['avg_drift'] > 0.2]
            
            if len(high_drift_features) > 3:
                triggers['should_retrain'] = True
                triggers['reasons'].append(f"High drift detected in {len(high_drift_features)} features")
                if not triggers['trigger_type']:
                    triggers['trigger_type'] = 'drift'
                    self.metrics.record_retrain_trigger('feature_drift', 'both')
            
            # Time-based triggers (weekly)
            last_retrain = self.get_last_retrain_date()
            if last_retrain and (datetime.now() - last_retrain).days > 7:
                triggers['should_retrain'] = True
                triggers['reasons'].append("Weekly retrain schedule")
                if not triggers['trigger_type']:
                    triggers['trigger_type'] = 'scheduled'
                    self.metrics.record_retrain_trigger('scheduled_weekly', 'both')
            
            if triggers['should_retrain']:
                logger.info(f"🔄 Retrain triggered: {triggers['reasons']}")
            
            return triggers
            
        except Exception as e:
            logger.error(f"Failed to check retrain triggers: {e}")
            return triggers
    
    def get_last_retrain_date(self) -> Optional[datetime]:
        """Get the date of the last retraining"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                result = conn.execute(
                    "SELECT MAX(created_at) FROM retrain_events"
                ).fetchone()
                
                if result[0]:
                    return datetime.fromisoformat(result[0])
                return None
                
        except Exception as e:
            logger.error(f"Failed to get last retrain date: {e}")
            return None

# Global metrics instance
f1_monitor = F1ModelMonitor()

def get_metrics_for_prometheus() -> str:
    """Get metrics in Prometheus format"""
    return f1_monitor.metrics.generate_metrics()

def update_race_metrics(race_id: str, predictions: List[Dict], actuals: List[Dict], features: pd.DataFrame):
    """Update metrics after race completion"""
    race_date = datetime.now().strftime('%Y-%m-%d')
    f1_monitor.update_all_metrics(race_id, race_date, predictions, actuals, features)

def check_retrain_needed() -> Dict:
    """Check if retraining is needed"""
    return f1_monitor.check_retrain_triggers()

if __name__ == "__main__":
    print("🔄 PHASE 11.1: F1 Model Monitoring System")
    print("=" * 60)
    
    # Initialize monitoring system
    monitor = F1ModelMonitor()
    
    # Example usage
    print("✅ Monitoring system initialized")
    print(f"   Database: {monitor.db.db_path}")
    print(f"   Training stats: {len(monitor.training_data_stats)} features")
    
    # Generate sample metrics
    sample_predictions = [
        {'driver_id': 1, 'predicted_quali_time': 75.5, 'predicted_win_probability': 0.8},
        {'driver_id': 2, 'predicted_quali_time': 75.8, 'predicted_win_probability': 0.2}
    ]
    
    sample_actuals = [
        {'driver_id': 1, 'actual_quali_time': 75.3, 'actual_race_winner': 1},
        {'driver_id': 2, 'actual_quali_time': 76.1, 'actual_race_winner': 0}
    ]
    
    # Test metrics update
    sample_features = pd.DataFrame({
        'temperature': [25.0, 26.0],
        'humidity': [60.0, 65.0],
        'wind_speed': [10.0, 12.0]
    })
    
    update_race_metrics('test_race_001', sample_predictions, sample_actuals, sample_features)
    
    # Check retrain triggers
    triggers = check_retrain_needed()
    print(f"   Retrain needed: {triggers['should_retrain']}")
    if triggers['reasons']:
        print(f"   Reasons: {triggers['reasons']}")
    
    print("✅ PHASE 11.1 MONITORING SYSTEM READY!")