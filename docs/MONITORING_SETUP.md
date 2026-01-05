# 📊 Grafana/Prometheus Monitoring Setup

## Quick Start

### 1. Start All Services
```bash
docker-compose up -d
```

This will start:
- **API**: http://localhost:5002
- **MLflow**: http://localhost:5050
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### 2. Access Grafana Dashboard

1. Open http://localhost:3000
2. Login with:
   - Username: `admin`
   - Password: `admin`
3. Navigate to **Dashboards** → **Diabetes Prediction - Model Monitoring**

### 3. Generate Some Predictions

Make predictions to see metrics:

```bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "bmi": 28.5,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 140,
    "hypertension": 1,
    "heart_disease": 0,
    "gender": 1,
    "smoking_history": 2
  }'
```

Or use the Web UI at http://localhost:5002

### 4. View Metrics

**Prometheus Raw Metrics**: http://localhost:5002/metrics

**Prometheus UI**: http://localhost:9090
- Query: `rate(prediction_requests_total[5m])`
- Query: `prediction_probability`
- Query: `prediction_duration_seconds`

**Grafana Dashboard**: http://localhost:3000
- Real-time request rate
- Prediction distribution
- Model confidence scores
- Data drift detection

## 📈 Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter | Total prediction requests by endpoint and prediction class |
| `prediction_duration_seconds` | Histogram | Prediction latency distribution |
| `prediction_probability` | Gauge | Last prediction probability score |
| `data_drift_detected` | Gauge | Data drift detection status (0=no drift, 1=drift) |
| `model_info` | Gauge | Model metadata (type, version) |

## 🎯 Dashboard Panels

1. **Requests per Second** - Real-time request rate gauge
2. **Prediction Request Rate** - Time series of requests
3. **Positive Prediction Rate** - Percentage of positive predictions
4. **Predictions by Class** - Distribution of predictions (0 vs 1)
5. **Prediction Latency** - Average response time
6. **Model Confidence Score** - Average probability scores
7. **Data Drift Status** - Drift detection alerts

## 🔧 Configuration

### Prometheus Configuration
File: `prometheus.yml`
- Scrape interval: 15s
- Metrics endpoint: http://api:5002/metrics

### Grafana Datasource
File: `grafana/datasources.yml`
- Automatically provisions Prometheus datasource
- No manual configuration needed

### Dashboard
File: `grafana-dashboard.json`
- Auto-imported on Grafana startup
- Customizable via Grafana UI

## 🚀 Production Tips

### 1. Secure Grafana
Change default password:
```yaml
environment:
  - GF_SECURITY_ADMIN_PASSWORD=your_secure_password
```

### 2. Persist Data
Volumes are already configured:
- `prometheus-data`: Prometheus time-series data
- `grafana-data`: Grafana dashboards and settings

### 3. Add Alerts

Create `alert_rules.yml`:
```yaml
groups:
  - name: model_monitoring
    interval: 1m
    rules:
      - alert: HighPredictionRate
        expr: rate(prediction_requests_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction request rate"
      
      - alert: DataDriftDetected
        expr: data_drift_detected > 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Data drift detected"
```

### 4. Set Retention Period

Update `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  
storage:
  tsdb:
    retention.time: 30d
    retention.size: 10GB
```

## 📊 Monitoring Workflow

1. **Real-time Monitoring**
   - Watch Grafana dashboard for live metrics
   - Set up alerts for anomalies

2. **Data Drift Detection**
   - Run `python src/monitor.py` periodically
   - Check drift reports in `logs/monitoring/`
   - Update `data_drift_detected` metric

3. **Performance Tracking**
   - Monitor latency trends
   - Identify bottlenecks
   - Optimize model/API as needed

4. **Model Performance**
   - Track prediction distribution
   - Monitor confidence scores
   - Detect prediction drift

## 🛠️ Troubleshooting

### Prometheus Not Scraping
```bash
# Check API metrics endpoint
curl http://localhost:5002/metrics

# Check Prometheus targets
# Visit http://localhost:9090/targets
```

### Grafana Dashboard Empty
```bash
# Restart Grafana to reload dashboard
docker-compose restart grafana

# Check datasource connection
# Visit Settings → Data Sources
```

### No Metrics Showing
```bash
# Make some predictions first
curl -X POST http://localhost:5002/predict -H "Content-Type: application/json" -d '...'

# Wait 15-30 seconds for Prometheus scrape
# Refresh Grafana dashboard
```

## 📚 Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Flask + Prometheus](https://github.com/prometheus/client_python)
- [MLOps Monitoring Best Practices](https://ml-ops.org/content/monitoring)

---

**Happy Monitoring! 📊**
