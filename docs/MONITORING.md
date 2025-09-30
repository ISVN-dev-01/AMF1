# 📊 AMF1 Monitoring Guide

*Comprehensive MLOps monitoring and alerting setup for AMF1*

---

## 📋 **Overview**

This guide covers the complete monitoring and observability setup for the AMF1 Formula 1 prediction system. The monitoring stack provides real-time insights into model performance, system health, data quality, and business metrics.

### **Monitoring Stack**
- **Metrics Collection**: Prometheus
- **Visualization**: Grafana
- **Alerting**: Prometheus Alertmanager + Slack/PagerDuty
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger (optional)
- **Application Monitoring**: Custom Python metrics

---

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AMF1 API      │───▶│   Prometheus    │───▶│    Grafana      │
│  (FastAPI)      │    │   (Metrics)     │    │ (Dashboards)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Application    │    │  Alertmanager   │    │   Slack/Email   │
│     Logs        │    │   (Alerts)      │    │ (Notifications) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🚀 **Quick Setup**

### **1. Docker Compose Setup**

Create `monitoring/docker-compose.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: amf1-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alert-rules.yml:/etc/prometheus/alert-rules.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--web.route-prefix=/'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: amf1-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=amf1admin

  alertmanager:
    image: prom/alertmanager:latest
    container_name: amf1-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  grafana-storage:
```

### **2. Start Monitoring Stack**

```bash
cd monitoring
docker-compose up -d

# Check services
docker-compose ps

# Access interfaces
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/amf1admin)
# Alertmanager: http://localhost:9093
```

---

## ⚙️ **Configuration Files**

### **Prometheus Configuration (`prometheus.yml`)**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert-rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # AMF1 API Metrics
  - job_name: 'amf1-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # Prometheus Self-Monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### **Alert Rules (`alert-rules.yml`)**

```yaml
groups:
  - name: amf1_model_performance
    rules:
      # Stage-1 Model Performance
      - alert: Stage1ModelDegradation
        expr: amf1_model_mae{model="stage1"} > 0.40
        for: 5m
        labels:
          severity: warning
          model: stage1
        annotations:
          summary: "Stage-1 model performance degradation detected"
          description: "MAE increased to {{ $value }}s (threshold: 0.40s)"

      - alert: Stage1ModelCritical
        expr: amf1_model_mae{model="stage1"} > 0.50
        for: 2m
        labels:
          severity: critical
          model: stage1
        annotations:
          summary: "Stage-1 model critical performance degradation"
          description: "MAE critically high at {{ $value }}s (threshold: 0.50s)"

      # Stage-2 Model Performance
      - alert: Stage2ModelDegradation
        expr: amf1_model_brier_score{model="stage2"} > 0.16
        for: 5m
        labels:
          severity: warning
          model: stage2
        annotations:
          summary: "Stage-2 model performance degradation detected"
          description: "Brier score increased to {{ $value }} (threshold: 0.16)"

  - name: amf1_api_health
    rules:
      # API Latency
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, amf1_prediction_latency_seconds_bucket) > 0.2
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High API prediction latency"
          description: "95th percentile latency is {{ $value }}s (threshold: 0.2s)"

      # API Error Rate
      - alert: HighErrorRate
        expr: rate(amf1_predictions_errors_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 1%)"

      # Model Loading Issues
      - alert: ModelNotLoaded
        expr: amf1_model_loaded{model=~"stage1|stage2"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model not loaded"
          description: "{{ $labels.model }} model is not loaded"

  - name: amf1_data_quality
    rules:
      # Data Freshness
      - alert: StaleData
        expr: (time() - amf1_data_last_update_timestamp) > 86400
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Stale training data detected"
          description: "Data not updated for {{ $value | humanizeDuration }}"

      # Data Quality Issues
      - alert: HighMissingDataRate
        expr: amf1_data_missing_rate > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High missing data rate"
          description: "Missing data rate is {{ $value | humanizePercentage }} (threshold: 5%)"
```

### **Alertmanager Configuration (`alertmanager.yml`)**

```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@amf1.com'

templates:
  - '/etc/alertmanager/templates/*.tmpl'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#amf1-alerts'

  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#amf1-critical'
        title: '🚨 AMF1 Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'

  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#amf1-alerts'
        title: '⚠️ AMF1 Warning Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

---

## 📊 **Grafana Dashboards**

### **1. Model Performance Dashboard**

Create `grafana/dashboards/amf1-model-performance.json`:

```json
{
  "dashboard": {
    "title": "AMF1 Model Performance",
    "panels": [
      {
        "title": "Stage-1 MAE Trend",
        "type": "stat",
        "targets": [
          {
            "expr": "amf1_model_mae{model=\"stage1\"}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 0.35},
                {"color": "red", "value": 0.45}
              ]
            }
          }
        }
      },
      {
        "title": "Stage-2 Brier Score",
        "type": "stat",
        "targets": [
          {
            "expr": "amf1_model_brier_score{model=\"stage2\"}"
          }
        ]
      },
      {
        "title": "Prediction Volume",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(amf1_predictions_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### **2. API Performance Dashboard**

Key panels to include:
- **Request Rate**: `rate(amf1_requests_total[5m])`
- **Response Time**: `histogram_quantile(0.95, amf1_request_duration_seconds_bucket)`
- **Error Rate**: `rate(amf1_requests_errors_total[5m]) / rate(amf1_requests_total[5m])`
- **Active Connections**: `amf1_active_connections`

### **3. System Health Dashboard**

Key panels:
- **CPU Usage**: `100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`
- **Memory Usage**: `(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100`
- **Disk Usage**: `100 - ((node_filesystem_avail_bytes * 100) / node_filesystem_size_bytes)`
- **Network I/O**: `rate(node_network_receive_bytes_total[5m])`, `rate(node_network_transmit_bytes_total[5m])`

---

## 🔧 **Custom Metrics Implementation**

### **Python Metrics Collection**

Add to your FastAPI application (`src/serve/app.py`):

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import prometheus_client
import time

# Define metrics
PREDICTIONS_TOTAL = Counter(
    'amf1_predictions_total',
    'Total number of predictions made',
    ['model', 'status']
)

PREDICTION_LATENCY = Histogram(
    'amf1_prediction_latency_seconds',
    'Time spent processing predictions',
    ['model']
)

MODEL_PERFORMANCE = Gauge(
    'amf1_model_mae',
    'Current model MAE performance',
    ['model']
)

BRIER_SCORE = Gauge(
    'amf1_model_brier_score',
    'Current model Brier score',
    ['model']
)

DATA_QUALITY = Gauge(
    'amf1_data_missing_rate',
    'Rate of missing data in features'
)

MODEL_LOADED = Gauge(
    'amf1_model_loaded',
    'Whether model is loaded (1) or not (0)',
    ['model']
)

# Middleware to track metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Track response time
    duration = time.time() - start_time
    if '/predict/' in str(request.url):
        model = 'stage1' if 'qualifying' in str(request.url) else 'stage2'
        PREDICTION_LATENCY.labels(model=model).observe(duration)
        
        # Track prediction counts
        status = 'success' if response.status_code == 200 else 'error'
        PREDICTIONS_TOTAL.labels(model=model, status=status).inc()
    
    return response

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(
        generate_latest(prometheus_client.REGISTRY),
        media_type="text/plain"
    )

# Update model performance metrics periodically
async def update_performance_metrics():
    while True:
        try:
            # Update with actual performance calculations
            MODEL_PERFORMANCE.labels(model='stage1').set(0.314)  # Current MAE
            MODEL_PERFORMANCE.labels(model='stage2').set(0.789)  # Current accuracy
            BRIER_SCORE.labels(model='stage2').set(0.142)  # Current Brier score
            
            # Check if models are loaded
            stage1_loaded = 1 if stage1_model is not None else 0
            stage2_loaded = 1 if stage2_model is not None else 0
            MODEL_LOADED.labels(model='stage1').set(stage1_loaded)
            MODEL_LOADED.labels(model='stage2').set(stage2_loaded)
            
            await asyncio.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            await asyncio.sleep(60)
```

### **Data Quality Monitoring**

```python
def monitor_data_quality(df):
    """Monitor data quality and update metrics"""
    missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    DATA_QUALITY.set(missing_rate)
    
    # Log data quality issues
    if missing_rate > 0.05:
        logger.warning(f"High missing data rate: {missing_rate:.2%}")
    
    return missing_rate
```

---

## 🚨 **Alerting Setup**

### **Slack Integration**

1. **Create Slack App**: Go to api.slack.com/apps
2. **Add Incoming Webhooks**: Enable and create webhook URLs
3. **Configure Channels**:
   - `#amf1-alerts`: General alerts and warnings
   - `#amf1-critical`: Critical alerts requiring immediate attention
   - `#amf1-data`: Data quality and pipeline alerts

### **PagerDuty Integration**

1. **Create Service**: In PagerDuty, create AMF1 service
2. **Get Integration Key**: Copy the service integration key
3. **Configure Escalation**: Set up on-call schedules and escalation policies

### **Email Notifications**

Configure SMTP settings in Alertmanager:

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'amf1-alerts@yourdomain.com'
  smtp_auth_username: 'amf1-alerts@yourdomain.com'
  smtp_auth_password: 'your-app-password'
```

---

## 📈 **Performance Baselines**

### **Model Performance Baselines**

| Metric | Stage-1 (Qualifying) | Stage-2 (Race Winner) |
|--------|---------------------|------------------------|
| **Excellent** | MAE < 0.30s | Brier < 0.12 |
| **Good** | MAE < 0.35s | Brier < 0.15 |
| **Acceptable** | MAE < 0.40s | Brier < 0.18 |
| **Warning** | MAE > 0.40s | Brier > 0.18 |
| **Critical** | MAE > 0.50s | Brier > 0.25 |

### **API Performance Baselines**

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| **Response Time (p95)** | <100ms | >200ms | >500ms |
| **Error Rate** | <0.1% | >1% | >5% |
| **Throughput** | >1000 RPS | <500 RPS | <100 RPS |
| **Uptime** | >99.9% | <99.5% | <99% |

---

## 🔍 **Log Analysis**

### **Log Aggregation Setup**

#### **ELK Stack Configuration**

```yaml
# docker-compose.elk.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"

  kibana:
    image: docker.elastic.co/kibana/kibana:7.14.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
```

#### **Logstash Configuration (`logstash.conf`)**

```ruby
input {
  file {
    path => "/var/log/amf1/*.log"
    start_position => "beginning"
  }
}

filter {
  if [message] =~ /ERROR|CRITICAL/ {
    mutate {
      add_tag => ["error"]
    }
  }
  
  grok {
    match => { 
      "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{LOGLEVEL:level} - %{GREEDYDATA:msg}" 
    }
  }
  
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => "elasticsearch:9200"
    index => "amf1-logs-%{+YYYY.MM.dd}"
  }
}
```

### **Key Log Patterns to Monitor**

1. **Model Loading Failures**:
   ```
   ERROR - Failed to load Stage-1 model: FileNotFoundError
   ```

2. **Prediction Errors**:
   ```
   ERROR - Prediction failed for driver: hamilton, circuit: silverstone
   ```

3. **Data Quality Issues**:
   ```
   WARNING - High missing data rate: 8.5% in feature pipeline
   ```

4. **Performance Degradation**:
   ```
   WARNING - Stage-1 MAE increased to 0.42s (threshold: 0.40s)
   ```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **1. High Memory Usage**
```bash
# Check memory usage
docker stats amf1-api

# Scale down if needed
docker-compose scale amf1-api=2
```

#### **2. Model Loading Issues**
```bash
# Check model files
ls -la models/
docker exec -it amf1-api ls -la /opt/amf1/models/

# Check logs
docker logs amf1-api | grep -i "model"
```

#### **3. Database Connection Issues**
```bash
# Test database connectivity
docker exec -it amf1-api python -c "import pandas as pd; print('DB OK')"
```

### **Performance Tuning**

#### **1. API Optimization**
```python
# Add connection pooling
app.state.pool = create_connection_pool()

# Enable response caching
@lru_cache(maxsize=1000)
def cached_prediction(driver, circuit, weather):
    return make_prediction(driver, circuit, weather)
```

#### **2. Model Optimization**
```python
# Model quantization for faster inference
import onnxruntime as ort

# Convert to ONNX for faster inference
model_onnx = convert_to_onnx(lightgbm_model)
session = ort.InferenceSession(model_onnx)
```

---

## 📚 **Monitoring Playbooks**

### **Incident Response Playbook**

#### **P0 - Critical Issues (Response: 15 minutes)**
1. **Model Not Loading**
   - Check model file integrity
   - Verify file permissions
   - Rollback to previous model version
   - Notify team via Slack + PagerDuty

2. **API Down**
   - Check container health
   - Verify resource availability
   - Scale up if resource constrained
   - Implement circuit breaker

#### **P1 - High Severity (Response: 1 hour)**
1. **Performance Degradation**
   - Analyze recent data changes
   - Check for concept drift
   - Consider model retraining
   - Implement temporary workarounds

### **Weekly Health Check**

```bash
#!/bin/bash
# weekly_health_check.sh

echo "🏥 AMF1 Weekly Health Check"
echo "=========================="

# Check API health
curl -s http://localhost:8000/health | jq '.status'

# Check Prometheus metrics
echo "📊 Model Performance:"
curl -s 'http://localhost:9090/api/v1/query?query=amf1_model_mae{model="stage1"}' | jq '.data.result[0].value[1]'

# Check disk space
echo "💾 Disk Usage:"
df -h | grep -E "/$|/var"

# Check recent errors
echo "🚨 Recent Errors:"
docker logs amf1-api --since="7d" | grep -i error | wc -l

echo "✅ Health check completed"
```

---

## 📊 **Business Metrics**

### **Key Performance Indicators**

1. **Model Accuracy Metrics**
   - Stage-1 MAE trending
   - Stage-2 calibration score
   - Prediction confidence distribution

2. **Usage Metrics**
   - Daily prediction volume
   - API endpoint usage patterns
   - User engagement metrics

3. **Operational Metrics**
   - System uptime
   - Deployment frequency
   - Time to recovery (MTTR)

### **Monthly Reporting**

Create automated monthly reports:

```python
def generate_monthly_report():
    """Generate monthly performance report"""
    report = {
        'model_performance': {
            'stage1_mae': get_avg_mae_last_month(),
            'stage2_brier': get_avg_brier_last_month(),
            'improvement_trend': calculate_trend()
        },
        'api_metrics': {
            'total_predictions': get_prediction_count(),
            'avg_response_time': get_avg_response_time(),
            'uptime_percentage': calculate_uptime()
        },
        'incidents': {
            'p0_incidents': count_p0_incidents(),
            'p1_incidents': count_p1_incidents(),
            'mttr': calculate_mttr()
        }
    }
    
    send_report_to_stakeholders(report)
```

---

## 🔗 **Related Documentation**

- [Model Card](MODEL_CARD.md) - Model specifications and performance
- [Retraining Runbook](../runbooks/retrain.md) - Operational procedures  
- [API Documentation](API.md) - Complete API reference
- [README](../README.md) - System overview and setup

---

*Last Updated: September 30, 2025*
*Monitoring Guide Version: 1.0.0*