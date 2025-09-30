# 🔄 AMF1 Model Retraining Runbook

*Operational procedures for AMF1 model maintenance and retraining*

---

## 📋 **Overview**

This runbook provides step-by-step procedures for maintaining, monitoring, and retraining the AMF1 Formula 1 prediction models. Follow these procedures to ensure optimal model performance and reliability in production.

### **Key Responsibilities**
- **ML Engineers**: Execute retraining procedures, monitor performance
- **DevOps Engineers**: Manage infrastructure, deployment pipelines
- **Data Engineers**: Ensure data quality and availability
- **F1 Domain Experts**: Validate model outputs and provide context

---

## ⏰ **Retraining Schedule**

### **Regular Retraining Cadence**

| Schedule Type | Frequency | Trigger | Duration | Responsibility |
|---------------|-----------|---------|----------|----------------|
| **Incremental** | After every 3 races | Auto (via cron) | 2-4 hours | Automated |
| **Full Retrain** | End of season | Manual trigger | 8-12 hours | ML Engineer |
| **Emergency** | Performance degradation | Alert-based | 1-2 hours | On-call Engineer |
| **Regulation** | Rule changes | Manual trigger | 4-6 hours | Domain Expert + ML |

### **Seasonal Retraining Calendar 2025**

```
Season Events:
├── Pre-Season Testing (Feb): Full retrain with regulation updates
├── Mid-Season Break (Aug): Performance review and incremental update  
├── Season End (Dec): Complete model refresh and validation
└── Off-Season (Jan): Architecture improvements and feature engineering
```

---

## 🚨 **Performance Monitoring & Alerts**

### **Key Performance Indicators (KPIs)**

#### **Stage-1 (Qualifying) Thresholds**
```python
Performance Alerts:
├── MAE > 0.40s (29% degradation) → WARNING
├── MAE > 0.50s (61% degradation) → CRITICAL  
├── RMSE > 0.55s (28% degradation) → WARNING
├── Top-1 Accuracy < 75% (9pt drop) → WARNING
└── Prediction Latency > 200ms → WARNING
```

#### **Stage-2 (Race Winner) Thresholds**
```python
Performance Alerts:
├── Brier Score > 0.16 (14% degradation) → WARNING
├── Brier Score > 0.20 (43% degradation) → CRITICAL
├── Log-Loss > 1.20 (22% degradation) → WARNING  
├── Calibration Error > 0.08 (60% increase) → WARNING
└── AUC-ROC < 0.85 (7% drop) → WARNING
```

### **Data Quality Monitoring**
```python
Data Quality Alerts:
├── Missing Data > 5% → WARNING
├── Missing Data > 10% → CRITICAL
├── Feature Drift > 2σ → WARNING
├── Target Distribution Shift > 1.5σ → WARNING
└── API Error Rate > 1% → WARNING
```

### **Alert Channels**
- **Slack**: `#amf1-alerts` for all alerts
- **PagerDuty**: Critical alerts only (24/7 on-call)
- **Email**: Weekly performance summaries
- **Grafana**: Real-time dashboards and visualizations

---

## 🔧 **Retraining Procedures**

### **1. Incremental Retraining (Every 3 Races)**

#### **Prerequisites**
- [ ] New race data available in `data/raw/`
- [ ] Data quality checks passed
- [ ] Model performance within acceptable thresholds
- [ ] Compute resources available (8GB RAM, 4 vCPUs)

#### **Execution Steps**

```bash
# 1. Set up environment
cd /opt/amf1
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/opt/amf1/src

# 2. Validate new data
python src/data_collection/validate_new_data.py --races=3

# 3. Update feature pipeline  
python src/features/incremental_features.py --update_latest

# 4. Incremental model training
python scripts/incremental_retrain.py \
    --model_type=both \
    --validation_method=holdout \
    --performance_threshold=0.05

# 5. Model validation
python src/evaluation/validate_new_model.py \
    --compare_baseline=true \
    --min_improvement=0.02

# 6. Deploy if validation passes
python scripts/deploy_model.py \
    --deployment_type=canary \
    --rollback_threshold=0.1
```

#### **Expected Outcomes**
- Training Time: 2-4 hours
- Performance Improvement: 1-3%
- Model Size: <500MB per stage
- API Downtime: <5 minutes (canary deployment)

#### **Rollback Procedures**
```bash
# If new model underperforms
python scripts/rollback_model.py \
    --version=previous \
    --reason="performance_degradation"

# Monitor for 24 hours before considering stable
python scripts/monitor_rollback.py --duration=24h
```

### **2. Full Season Retraining (Annual)**

#### **Prerequisites**
- [ ] Complete season data collected and validated
- [ ] Infrastructure scaled up (32GB RAM, 16 vCPUs)
- [ ] Regulation changes documented and integrated
- [ ] Historical performance benchmarks established
- [ ] Stakeholder approval for downtime window

#### **Execution Steps (8-12 hours)**

```bash
# Phase 1: Data Preparation (2-3 hours)
python src/data_collection/full_season_collection.py --season=2024
python src/data_collection/comprehensive_cleaning.py
python src/features/full_feature_engineering.py --historical_lookback=5

# Phase 2: Model Training (4-6 hours)
python scripts/full_retrain.py \
    --hyperparameter_tuning=true \
    --cross_validation=5fold \
    --ensemble_optimization=true \
    --feature_selection=true

# Phase 3: Comprehensive Validation (2-3 hours)
python src/evaluation/comprehensive_evaluation.py \
    --baseline_comparison=true \
    --fairness_analysis=true \
    --robustness_testing=true

# Phase 4: Deployment Preparation
python scripts/prepare_production_models.py
python scripts/update_model_metadata.py
python docs/update_model_card.py
```

#### **Validation Checklist**
- [ ] Performance exceeds baseline by >5% (Stage-1 MAE)
- [ ] Calibration error <0.06 (Stage-2)
- [ ] Fairness metrics within acceptable bounds
- [ ] Load testing passed (1000+ RPS)
- [ ] Documentation updated
- [ ] Stakeholder sign-off received

### **3. Emergency Retraining (Performance Degradation)**

#### **Trigger Conditions**
- Critical performance alerts for 2+ consecutive races
- Data quality issues affecting >15% of predictions
- Security vulnerabilities in model dependencies
- Major regulation changes mid-season

#### **Emergency Response (1-2 hours)**

```bash
# 1. Immediate Assessment
python scripts/emergency_diagnosis.py \
    --performance_check=true \
    --data_quality_check=true \
    --system_health_check=true

# 2. Quick Fix Options
# Option A: Rollback to previous stable version
python scripts/emergency_rollback.py --version=last_stable

# Option B: Rapid retraining with reduced dataset
python scripts/emergency_retrain.py \
    --quick_mode=true \
    --validation_minimal=true \
    --deployment_immediate=true

# 3. Communication
python scripts/send_incident_notification.py \
    --severity=high \
    --stakeholders=all
```

#### **Post-Emergency Actions**
1. **Root Cause Analysis**: Complete within 48 hours
2. **Documentation Update**: Document incident and resolution
3. **Process Improvement**: Update procedures based on learnings
4. **Follow-up Monitoring**: Extended monitoring for 1 week

---

## 📊 **Data Management**

### **Data Collection Workflows**

#### **Automated Data Ingestion**
```bash
# Cron job runs every Sunday at 02:00 UTC
0 2 * * 0 /opt/amf1/scripts/weekly_data_collection.py

# Daily incremental updates
0 6 * * * /opt/amf1/scripts/daily_data_update.py
```

#### **Data Validation Pipeline**
```python
Validation Checks:
├── Schema Validation: Column types, names, constraints
├── Completeness: Missing value thresholds per column
├── Consistency: Cross-table relationships and referential integrity
├── Timeliness: Data freshness and update frequencies  
├── Accuracy: Outlier detection and range validation
└── Uniqueness: Duplicate detection and handling
```

### **Data Quality Issues & Resolutions**

| Issue Type | Detection | Resolution | Prevention |
|------------|-----------|------------|------------|
| **Missing Race Data** | Ergast API timeout | Manual data entry + validation | Multiple API sources |
| **Weather Data Gaps** | FastF1 telemetry incomplete | Historical weather interpolation | Weather API backup |
| **Timing Anomalies** | Statistical outlier detection | Domain expert review | Automated flagging |
| **Schema Changes** | Pipeline validation failure | Schema migration scripts | API version pinning |

---

## 🚀 **Deployment Procedures**

### **Model Deployment Pipeline**

#### **Canary Deployment (Default)**
```bash
# Stage 1: Deploy to 5% of traffic
python scripts/canary_deploy.py \
    --traffic_percentage=5 \
    --monitoring_duration=2h \
    --success_criteria="mae_improvement>0.01"

# Stage 2: Increase to 25% if successful  
python scripts/canary_scale.py --traffic_percentage=25

# Stage 3: Full deployment
python scripts/canary_complete.py --full_traffic=true
```

#### **Blue-Green Deployment (Critical Updates)**
```bash
# Prepare green environment
python scripts/prepare_green_environment.py

# Deploy new model to green
python scripts/deploy_to_green.py --model_version=latest

# Switch traffic (zero downtime)
python scripts/switch_to_green.py --validation_checks=all
```

### **Deployment Checklist**

#### **Pre-Deployment**
- [ ] Model performance validated against benchmarks
- [ ] Integration tests passed (API, database, monitoring)
- [ ] Load testing completed (target: 1000 RPS)
- [ ] Security scanning completed (no critical vulnerabilities)
- [ ] Documentation updated (model card, API docs)
- [ ] Rollback plan prepared and tested
- [ ] Stakeholder notification sent

#### **During Deployment**
- [ ] Monitor key metrics continuously
- [ ] Check error rates and response times
- [ ] Validate prediction quality in real-time
- [ ] Confirm monitoring and alerting functional
- [ ] Document any issues or anomalies

#### **Post-Deployment**
- [ ] 24-hour monitoring period completed
- [ ] Performance metrics stable and improved
- [ ] No critical alerts or issues
- [ ] User feedback collected and reviewed
- [ ] Deployment summary report created

---

## 🔍 **Troubleshooting Guide**

### **Common Issues & Solutions**

#### **Training Failures**

| Error | Symptoms | Diagnosis | Solution |
|-------|----------|-----------|----------|
| **Memory Error** | OOM during training | `htop`, memory profiling | Reduce batch size, enable chunking |  
| **Data Corruption** | Training accuracy drops | Data validation checks | Re-download affected data |
| **Feature Drift** | Model performance degrades | Feature distribution analysis | Retrain with recent data |
| **Convergence Issues** | Loss plateaus early | Learning curves analysis | Adjust learning rate, add regularization |

#### **Deployment Issues**

| Issue | Symptoms | Diagnosis | Solution |
|-------|----------|-----------|----------|
| **API Latency** | Response times >200ms | APM monitoring | Model optimization, caching |
| **Memory Leaks** | Gradual memory increase | Memory profiling | Restart services, fix leaks |
| **Prediction Errors** | High error rates | Error logs analysis | Model rollback, data validation |
| **Load Balancer Issues** | Uneven traffic distribution | Infrastructure monitoring | Load balancer config |

### **Diagnostic Commands**

```bash
# Check model performance
python scripts/diagnose_performance.py --days=7 --verbose

# Validate data pipeline
python scripts/diagnose_data_pipeline.py --full_check

# Test API health
python scripts/diagnose_api.py --load_test --duration=5m

# Check infrastructure
python scripts/diagnose_infrastructure.py --components=all
```

### **Log Analysis**

#### **Key Log Locations**
```
/var/log/amf1/
├── training.log          # Model training logs
├── api.log              # API request/response logs  
├── data_pipeline.log    # Data collection and processing
├── monitoring.log       # Performance metrics and alerts
└── deployment.log       # Deployment and infrastructure
```

#### **Log Analysis Tools**
```bash
# Find training errors
grep -i "error\|exception\|failed" /var/log/amf1/training.log | tail -20

# Check API performance
awk '{print $9}' /var/log/amf1/api.log | sort -n | tail -10

# Monitor data quality issues  
grep "data_quality_check: FAILED" /var/log/amf1/data_pipeline.log
```

---

## 📞 **Escalation Procedures**

### **On-Call Rotation**
- **Primary**: ML Engineer (24/7)
- **Secondary**: DevOps Engineer (business hours)
- **Escalation**: Senior ML Engineer (critical issues only)

### **Escalation Matrix**

| Severity | Response Time | Escalation Path | Communication |
|----------|---------------|-----------------|---------------|
| **P0 - Critical** | 15 minutes | Immediate to Senior ML Engineer | Slack + PagerDuty + Email |
| **P1 - High** | 1 hour | After 2 hours to Team Lead | Slack + Email |
| **P2 - Medium** | 4 hours | After 8 hours to Manager | Slack |
| **P3 - Low** | 24 hours | Weekly review | Email summary |

### **Communication Templates**

#### **Critical Issue Notification**
```
Subject: [P0] AMF1 Critical Issue - Model Performance Degradation

Issue: Stage-1 MAE increased to 0.52s (67% degradation)
Impact: Production API serving degraded predictions
Timeline: Started at 14:30 UTC, ongoing
Actions Taken: 
1. Rolled back to previous model version
2. Investigating root cause in latest data
ETA for Resolution: 2 hours
Next Update: 30 minutes

Contact: [On-call Engineer] for questions
```

---

## 📈 **Performance Optimization**

### **Model Optimization Techniques**

#### **Training Acceleration**
```python
Optimization Strategies:
├── Early Stopping: Monitor validation loss, stop when plateauing
├── Learning Rate Scheduling: Reduce LR on plateau  
├── Feature Selection: Remove low-importance features (<0.01)
├── Hyperparameter Tuning: Bayesian optimization (Optuna)
├── Distributed Training: Multi-GPU for large datasets
└── Model Pruning: Remove redundant ensemble components
```

#### **Inference Optimization**
```python
Inference Acceleration:
├── Model Quantization: INT8 quantization for 2x speedup
├── ONNX Conversion: Cross-platform optimized runtime
├── Batch Prediction: Process multiple requests together
├── Feature Caching: Cache computed features for 30 minutes
├── Result Caching: Cache predictions for identical inputs
└── Load Balancing: Distribute requests across instances
```

### **Infrastructure Scaling**

#### **Auto-Scaling Configuration**
```yaml
Auto-Scaling Rules:
├── CPU Utilization > 70%: Add instance
├── Memory Usage > 80%: Add instance  
├── Request Latency > 150ms: Add instance
├── Error Rate > 0.5%: Add instance
├── Min Instances: 2
└── Max Instances: 10
```

---

## 📚 **Documentation Maintenance**

### **Documentation Update Schedule**
- **Model Card**: Updated with each major retrain
- **API Documentation**: Updated with each API change
- **Runbook**: Reviewed monthly, updated quarterly
- **Performance Reports**: Generated weekly, archived monthly

### **Version Control**
```bash
# Documentation versioning
git tag -a docs-v1.2.0 -m "Updated for Q4 2025 retraining"
git push origin docs-v1.2.0

# Model versioning  
git tag -a model-v2.1.0 -m "Post-season retraining December 2025"
git push origin model-v2.1.0
```

---

## ✅ **Checklist Templates**

### **Pre-Retraining Checklist**
- [ ] Data quality validation passed
- [ ] Compute resources allocated
- [ ] Baseline performance documented
- [ ] Rollback plan prepared
- [ ] Stakeholder notification sent
- [ ] Infrastructure health check passed
- [ ] Backup of current model created

### **Post-Retraining Checklist**  
- [ ] Model performance validated
- [ ] A/B testing completed
- [ ] Documentation updated
- [ ] Monitoring alerts configured
- [ ] Team training completed (if needed)
- [ ] Performance summary report created
- [ ] Post-mortem scheduled (if issues occurred)

---

*Last Updated: September 30, 2025*
*Runbook Version: 1.0.0*