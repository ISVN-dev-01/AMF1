#!/bin/bash

# PHASE 11.4: Cron-based Retraining Scheduler
# Alternative to Airflow for simpler deployment environments

set -e

# Configuration
PROJECT_DIR="/Users/vishale/Documents/AMF!-MLmodel/AMF1"
RETRAIN_SCRIPT="$PROJECT_DIR/src/retraining/train_full_pipeline.py"
MONITORING_DB="$PROJECT_DIR/data/monitoring/metrics.db"
LOG_DIR="$PROJECT_DIR/logs/retraining"
EMAIL_RECIPIENTS="f1-ml-team@company.com"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/scheduler.log"
}

# Function to send email notifications
send_notification() {
    local subject="$1"
    local body="$2"
    
    # Use mail command if available, otherwise log
    if command -v mail &> /dev/null; then
        echo "$body" | mail -s "$subject" "$EMAIL_RECIPIENTS"
        log "📧 Email notification sent: $subject"
    else
        log "📧 Email not configured, logging notification: $subject"
        log "Body: $body"
    fi
}

# Function to check if retraining should be triggered
check_retrain_triggers() {
    log "🔍 Checking retraining triggers..."
    
    # Check if monitoring database exists
    if [ ! -f "$MONITORING_DB" ]; then
        log "⚠️  Monitoring database not found, skipping trigger check"
        return 1
    fi
    
    # Use Python to check triggers
    python3 << EOF
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import sys

try:
    conn = sqlite3.connect('$MONITORING_DB')
    
    # Check recent performance metrics
    recent_cutoff = datetime.now() - timedelta(days=7)
    
    query = """
    SELECT avg(stage1_accuracy) as avg_stage1_acc,
           avg(stage2_accuracy) as avg_stage2_acc,
           avg(feature_drift_score) as avg_drift_score,
           count(*) as total_predictions
    FROM race_results
    WHERE timestamp > ?
    """
    
    df = pd.read_sql_query(query, conn, params=(recent_cutoff.isoformat(),))
    conn.close()
    
    if len(df) == 0:
        print("No recent data")
        sys.exit(1)
    
    row = df.iloc[0]
    
    # Define thresholds
    STAGE1_THRESHOLD = 0.85
    STAGE2_THRESHOLD = 0.90
    DRIFT_THRESHOLD = 0.15
    MIN_PREDICTIONS = 50
    
    triggers = []
    
    if row['avg_stage1_acc'] < STAGE1_THRESHOLD:
        triggers.append(f"Stage-1 accuracy: {row['avg_stage1_acc']:.3f}")
    
    if row['avg_stage2_acc'] < STAGE2_THRESHOLD:
        triggers.append(f"Stage-2 accuracy: {row['avg_stage2_acc']:.3f}")
    
    if row['avg_drift_score'] > DRIFT_THRESHOLD:
        triggers.append(f"Feature drift: {row['avg_drift_score']:.3f}")
    
    should_retrain = len(triggers) > 0 and row['total_predictions'] >= MIN_PREDICTIONS
    
    print(f"Triggers: {len(triggers)}")
    print(f"Predictions: {row['total_predictions']}")
    
    for trigger in triggers:
        print(f"Trigger: {trigger}")
    
    sys.exit(0 if should_retrain else 1)
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(2)
EOF
    
    return $?
}

# Function to run retraining pipeline
run_retraining() {
    local trigger_type="$1"
    
    log "🚀 Starting retraining pipeline (trigger: $trigger_type)..."
    
    cd "$PROJECT_DIR"
    
    # Run retraining with logging
    if python3 "$RETRAIN_SCRIPT" --trigger "$trigger_type" --base-dir "$PROJECT_DIR" \
       > "$LOG_DIR/retrain_$(date +%Y%m%d_%H%M%S).log" 2>&1; then
        
        log "✅ Retraining pipeline completed successfully"
        return 0
    else
        log "❌ Retraining pipeline failed"
        return 1
    fi
}

# Function to restart API service
restart_api_service() {
    log "🔄 Checking if API service restart is needed..."
    
    cd "$PROJECT_DIR"
    
    # Check if new model was deployed
    if [ -f "models/feature_metadata.pkl" ]; then
        # Check deployment status from latest model registry
        registry_file=$(ls models/model_registry_*.json 2>/dev/null | tail -1)
        
        if [ -n "$registry_file" ]; then
            deployment_status=$(python3 -c "import json; print(json.load(open('$registry_file'))['deployment_status'])" 2>/dev/null || echo "unknown")
            
            if [ "$deployment_status" = "deployed" ]; then
                log "📦 New model deployed, restarting API service..."
                
                # Try Docker Compose restart
                if [ -f "docker-compose.yml" ]; then
                    docker-compose restart f1-ml-api 2>/dev/null || log "⚠️  Docker Compose restart failed"
                fi
                
                # Try systemd restart
                if command -v systemctl &> /dev/null; then
                    sudo systemctl restart f1-ml-api 2>/dev/null || log "⚠️  Systemd restart failed"
                fi
                
                log "✅ API service restart attempted"
            else
                log "⏸️  No deployment required, skipping restart"
            fi
        fi
    fi
}

# Function to cleanup old logs and backups
cleanup_old_files() {
    log "🧹 Cleaning up old files..."
    
    # Remove logs older than 30 days
    find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Remove model backups older than 60 days
    find "$PROJECT_DIR/models/backups" -type d -mtime +60 -exec rm -rf {} \; 2>/dev/null || true
    
    log "✅ Cleanup completed"
}

# Main function
main() {
    local mode="${1:-check}"
    
    log "🤖 F1 ML Retraining Scheduler started (mode: $mode)"
    
    case "$mode" in
        "check")
            # Check triggers and run if needed
            if check_retrain_triggers; then
                log "🚨 Retraining triggers detected, starting pipeline..."
                
                if run_retraining "performance"; then
                    restart_api_service
                    
                    send_notification \
                        "F1 ML Model Retrained Successfully" \
                        "The F1 ML model has been successfully retrained based on performance triggers. Check the logs for details."
                else
                    send_notification \
                        "F1 ML Model Retraining Failed" \
                        "The F1 ML model retraining pipeline failed. Please check the logs and investigate."
                fi
            else
                log "✅ No retraining triggers detected"
            fi
            ;;
            
        "force")
            # Force retraining regardless of triggers
            log "🔧 Forcing retraining pipeline..."
            
            if run_retraining "manual"; then
                restart_api_service
                
                send_notification \
                    "F1 ML Model Manual Retrain Completed" \
                    "The F1 ML model has been manually retrained. Check the logs for details."
            else
                send_notification \
                    "F1 ML Model Manual Retrain Failed" \
                    "The manual F1 ML model retraining failed. Please check the logs and investigate."
            fi
            ;;
            
        "weekly")
            # Weekly scheduled retrain
            log "📅 Running weekly scheduled retraining..."
            
            if run_retraining "scheduled"; then
                restart_api_service
                
                send_notification \
                    "F1 ML Model Weekly Retrain Completed" \
                    "The weekly F1 ML model retraining has completed successfully."
            else
                send_notification \
                    "F1 ML Model Weekly Retrain Failed" \
                    "The weekly F1 ML model retraining failed. Please investigate."
            fi
            ;;
            
        "cleanup")
            # Cleanup old files
            cleanup_old_files
            ;;
            
        *)
            echo "Usage: $0 {check|force|weekly|cleanup}"
            echo ""
            echo "Modes:"
            echo "  check   - Check triggers and retrain if needed (default)"
            echo "  force   - Force retraining regardless of triggers"
            echo "  weekly  - Run weekly scheduled retraining"
            echo "  cleanup - Clean up old logs and backups"
            exit 1
            ;;
    esac
    
    log "🏁 Scheduler completed"
}

# Install cron jobs function
install_cron_jobs() {
    echo "📅 Installing cron jobs for F1 ML retraining..."
    
    # Create cron entries
    cron_entries="
# F1 ML Model Retraining Schedule
# Check for performance-based triggers every 6 hours
0 */6 * * * $PWD/configs/cron/retrain_scheduler.sh check

# Weekly scheduled retraining (Sundays at 2 AM)
0 2 * * 0 $PWD/configs/cron/retrain_scheduler.sh weekly

# Daily cleanup (3 AM)
0 3 * * * $PWD/configs/cron/retrain_scheduler.sh cleanup
"
    
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$cron_entries") | crontab -
    
    echo "✅ Cron jobs installed successfully"
    echo ""
    echo "Installed schedules:"
    echo "- Performance check: Every 6 hours"
    echo "- Weekly retrain: Sundays at 2 AM"
    echo "- Cleanup: Daily at 3 AM"
    echo ""
    echo "To view: crontab -l"
    echo "To remove: crontab -r"
}

# If script is called with 'install', set up cron jobs
if [ "$1" = "install" ]; then
    install_cron_jobs
else
    main "$@"
fi