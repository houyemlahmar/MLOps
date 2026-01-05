"""
Model Monitoring & Drift Detection
Tracks prediction distribution, data drift, and model performance over time
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import joblib


class ModelMonitor:
    """Monitor model predictions and detect data drift"""
    
    def __init__(self, reference_data_path: str, model_path: str, 
                 selected_features_path: str, log_dir: str = "logs/monitoring"):
        """
        Initialize monitor with reference data (training data statistics)
        
        Args:
            reference_data_path: Path to training/reference dataset
            model_path: Path to trained model
            selected_features_path: Path to selected features JSON
            log_dir: Directory to store monitoring logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and features
        self.model = joblib.load(model_path)
        with open(selected_features_path, 'r') as f:
            self.features = json.load(f)
        
        # Load and compute reference statistics
        ref_data = pd.read_csv(reference_data_path)
        self.reference_stats = self._compute_statistics(ref_data[self.features])
        
        # Initialize prediction log
        self.predictions_log = []
        
    def _compute_statistics(self, data: pd.DataFrame) -> Dict:
        """Compute statistical properties of data"""
        stats_dict = {}
        for col in data.columns:
            stats_dict[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'q25': float(data[col].quantile(0.25)),
                'q75': float(data[col].quantile(0.75))
            }
        return stats_dict
    
    def log_prediction(self, features: Dict, prediction: int, 
                      probability: float, request_id: str = None):
        """Log a single prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id or f"req_{datetime.now().timestamp()}",
            'features': features,
            'prediction': int(prediction),
            'probability': float(probability),
        }
        self.predictions_log.append(log_entry)
        
        # Save to file every 10 predictions
        if len(self.predictions_log) % 10 == 0:
            self._save_logs()
    
    def _save_logs(self):
        """Save prediction logs to file"""
        log_file = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            for entry in self.predictions_log:
                f.write(json.dumps(entry) + '\n')
        self.predictions_log = []
    
    def detect_data_drift(self, new_data: pd.DataFrame, 
                         significance_level: float = 0.05) -> Dict:
        """
        Detect data drift using Kolmogorov-Smirnov test
        
        Args:
            new_data: New incoming data
            significance_level: P-value threshold for drift detection
            
        Returns:
            Dictionary with drift status for each feature
        """
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'overall_drift': False
        }
        
        drifted_features = []
        
        for feature in self.features:
            if feature not in new_data.columns:
                continue
                
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                self._generate_reference_sample(feature, len(new_data)),
                new_data[feature].values
            )
            
            is_drifted = p_value < significance_level
            if is_drifted:
                drifted_features.append(feature)
            
            drift_report['features'][feature] = {
                'ks_statistic': float(statistic),
                'p_value': float(p_value),
                'is_drifted': bool(is_drifted),
                'current_mean': float(new_data[feature].mean()),
                'reference_mean': self.reference_stats[feature]['mean'],
                'mean_difference': float(new_data[feature].mean() - 
                                       self.reference_stats[feature]['mean'])
            }
        
        drift_report['overall_drift'] = len(drifted_features) > 0
        drift_report['drifted_features'] = drifted_features
        drift_report['drift_percentage'] = len(drifted_features) / len(self.features) * 100
        
        # Save drift report
        self._save_drift_report(drift_report)
        
        return drift_report
    
    def _generate_reference_sample(self, feature: str, size: int) -> np.ndarray:
        """Generate sample from reference distribution"""
        stats = self.reference_stats[feature]
        return np.random.normal(stats['mean'], stats['std'], size)
    
    def _save_drift_report(self, report: Dict):
        """Save drift detection report"""
        report_file = self.log_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def get_prediction_distribution(self, days: int = 7) -> Dict:
        """
        Analyze prediction distribution over time
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics about prediction distribution
        """
        # Load recent logs
        recent_logs = self._load_recent_logs(days)
        
        if not recent_logs:
            return {'error': 'No prediction logs found'}
        
        predictions = [log['prediction'] for log in recent_logs]
        probabilities = [log['probability'] for log in recent_logs]
        
        return {
            'total_predictions': len(predictions),
            'positive_predictions': sum(predictions),
            'negative_predictions': len(predictions) - sum(predictions),
            'positive_rate': sum(predictions) / len(predictions) if predictions else 0,
            'avg_probability': np.mean(probabilities),
            'probability_std': np.std(probabilities),
            'confidence_distribution': {
                'high_confidence': sum(1 for p in probabilities if p > 0.8 or p < 0.2),
                'medium_confidence': sum(1 for p in probabilities if 0.4 <= p <= 0.8),
                'low_confidence': sum(1 for p in probabilities if 0.2 < p < 0.4)
            }
        }
    
    def _load_recent_logs(self, days: int) -> List[Dict]:
        """Load prediction logs from recent days"""
        logs = []
        for log_file in self.log_dir.glob("predictions_*.jsonl"):
            with open(log_file, 'r') as f:
                for line in f:
                    logs.append(json.loads(line))
        return logs


def generate_monitoring_report(monitor: ModelMonitor, new_data: pd.DataFrame) -> Dict:
    """
    Generate comprehensive monitoring report
    
    Args:
        monitor: ModelMonitor instance
        new_data: New data to analyze
        
    Returns:
        Complete monitoring report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_samples': len(new_data),
    }
    
    # Drift detection
    drift_report = monitor.detect_data_drift(new_data)
    report['drift_detection'] = drift_report
    
    # Prediction distribution
    pred_dist = monitor.get_prediction_distribution(days=7)
    report['prediction_distribution'] = pred_dist
    
    # Feature statistics comparison
    current_stats = monitor._compute_statistics(new_data[monitor.features])
    feature_comparison = {}
    
    for feature in monitor.features:
        if feature in current_stats:
            feature_comparison[feature] = {
                'reference_mean': monitor.reference_stats[feature]['mean'],
                'current_mean': current_stats[feature]['mean'],
                'mean_shift_percent': (
                    (current_stats[feature]['mean'] - monitor.reference_stats[feature]['mean']) /
                    monitor.reference_stats[feature]['mean'] * 100
                ),
                'reference_std': monitor.reference_stats[feature]['std'],
                'current_std': current_stats[feature]['std']
            }
    
    report['feature_comparison'] = feature_comparison
    
    # Alert generation
    alerts = []
    if drift_report['overall_drift']:
        alerts.append({
            'severity': 'WARNING',
            'message': f"Data drift detected in {len(drift_report['drifted_features'])} features",
            'features': drift_report['drifted_features']
        })
    
    if pred_dist.get('total_predictions', 0) > 0:
        positive_rate = pred_dist['positive_rate']
        if positive_rate > 0.7 or positive_rate < 0.3:
            alerts.append({
                'severity': 'INFO',
                'message': f"Unusual prediction rate: {positive_rate:.2%}",
                'positive_rate': positive_rate
            })
    
    report['alerts'] = alerts
    
    return report


if __name__ == "__main__":
    # Example usage
    print("🔍 Model Monitoring System")
    print("=" * 50)
    
    # Initialize monitor
    monitor = ModelMonitor(
        reference_data_path="data/processed/diabetes_processed.csv",
        model_path="models/final/model.pkl",
        selected_features_path="src/selected_features.json"
    )
    
    print("✅ Monitor initialized with reference data")
    
    # Load some test data
    test_data = pd.read_csv("data/processed/diabetes_processed.csv").sample(100)
    
    # Generate monitoring report
    report = generate_monitoring_report(monitor, test_data)
    
    print(f"\n📊 Monitoring Report ({report['timestamp']})")
    print(f"   Samples analyzed: {report['data_samples']}")
    print(f"   Drift detected: {report['drift_detection']['overall_drift']}")
    
    if report['drift_detection']['overall_drift']:
        print(f"   Drifted features: {', '.join(report['drift_detection']['drifted_features'])}")
    
    print(f"\n📈 Prediction Distribution:")
    if 'total_predictions' in report['prediction_distribution']:
        pd_stats = report['prediction_distribution']
        print(f"   Total predictions: {pd_stats['total_predictions']}")
        print(f"   Positive rate: {pd_stats['positive_rate']:.2%}")
    
    print(f"\n⚠️  Alerts: {len(report['alerts'])}")
    for alert in report['alerts']:
        print(f"   [{alert['severity']}] {alert['message']}")
    
    # Save report
    report_file = Path("logs/monitoring") / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_file}")
