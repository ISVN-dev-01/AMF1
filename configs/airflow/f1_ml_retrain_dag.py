"""
PHASE 11.4: Airflow DAG for F1 ML Model Retraining
Automated scheduling and orchestration of model retraining pipeline
"""

from datetime import datetime, timedelta
from pathlib import Path
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.hooks.base import BaseHook

# Default arguments for the DAG
default_args = {
    'owner': 'f1-ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['f1-ml-alerts@company.com']
}

# Create the DAG
dag = DAG(
    'f1_ml_model_retraining',
    default_args=default_args,
    description='F1 ML Model Automated Retraining Pipeline',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'f1', 'retraining', 'production']
)

# Configuration
PROJECT_BASE_DIR = Variable.get("F1_ML_PROJECT_DIR", default_var="/opt/f1-ml")
MONITORING_DB_PATH = f"{PROJECT_BASE_DIR}/data/monitoring/metrics.db"
RETRAIN_SCRIPT_PATH = f"{PROJECT_BASE_DIR}/src/retraining/train_full_pipeline.py"

def check_retrain_triggers(**context):
    """Check if retraining should be triggered based on monitoring metrics"""
    
    import sqlite3
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("🔍 Checking retraining triggers...")
    
    try:
        # Connect to monitoring database
        conn = sqlite3.connect(MONITORING_DB_PATH)
        
        # Check recent performance metrics
        recent_cutoff = datetime.now() - timedelta(days=7)
        
        # Query recent performance data
        performance_query = """
        SELECT avg(stage1_accuracy) as avg_stage1_acc,
               avg(stage2_accuracy) as avg_stage2_acc,
               avg(feature_drift_score) as avg_drift_score,
               count(*) as total_predictions
        FROM race_results
        WHERE timestamp > ?
        """
        
        performance_df = pd.read_sql_query(
            performance_query, 
            conn, 
            params=(recent_cutoff.isoformat(),)
        )
        
        conn.close()
        
        if len(performance_df) == 0:
            print("⚠️  No recent performance data found")
            return False
        
        row = performance_df.iloc[0]
        
        # Define trigger thresholds
        STAGE1_ACCURACY_THRESHOLD = 0.85
        STAGE2_ACCURACY_THRESHOLD = 0.90
        DRIFT_SCORE_THRESHOLD = 0.15
        MIN_PREDICTIONS_THRESHOLD = 50
        
        # Check triggers
        triggers = []
        
        if row['avg_stage1_acc'] < STAGE1_ACCURACY_THRESHOLD:
            triggers.append(f"Stage-1 accuracy below threshold: {row['avg_stage1_acc']:.3f}")
        
        if row['avg_stage2_acc'] < STAGE2_ACCURACY_THRESHOLD:
            triggers.append(f"Stage-2 accuracy below threshold: {row['avg_stage2_acc']:.3f}")
        
        if row['avg_drift_score'] > DRIFT_SCORE_THRESHOLD:
            triggers.append(f"Feature drift above threshold: {row['avg_drift_score']:.3f}")
        
        if row['total_predictions'] < MIN_PREDICTIONS_THRESHOLD:
            triggers.append(f"Insufficient predictions for reliable metrics: {row['total_predictions']}")
        
        # Log results
        print(f"📊 Performance Metrics (last 7 days):")
        print(f"   Stage-1 Accuracy: {row['avg_stage1_acc']:.3f}")
        print(f"   Stage-2 Accuracy: {row['avg_stage2_acc']:.3f}")
        print(f"   Feature Drift Score: {row['avg_drift_score']:.3f}")
        print(f"   Total Predictions: {row['total_predictions']}")
        
        should_retrain = len(triggers) > 0 and row['total_predictions'] >= MIN_PREDICTIONS_THRESHOLD
        
        if should_retrain:
            print("🚨 RETRAINING TRIGGERED!")
            for trigger in triggers:
                print(f"   - {trigger}")
        else:
            print("✅ No retraining triggers detected")
        
        # Store trigger information for downstream tasks
        context['task_instance'].xcom_push(key='retrain_triggers', value=triggers)
        context['task_instance'].xcom_push(key='performance_metrics', value=row.to_dict())
        
        return should_retrain
        
    except Exception as e:
        print(f"❌ Error checking retrain triggers: {e}")
        # Default to retraining on error to be safe
        return True

def run_retrain_pipeline(**context):
    """Execute the retraining pipeline"""
    
    import subprocess
    
    print("🚀 Starting retraining pipeline...")
    
    try:
        # Get trigger information from previous task
        triggers = context['task_instance'].xcom_pull(key='retrain_triggers', task_ids='check_triggers')
        trigger_type = 'scheduled' if not triggers else 'performance'
        
        # Run retraining script
        cmd = [
            'python3', 
            RETRAIN_SCRIPT_PATH,
            '--trigger', trigger_type,
            '--base-dir', PROJECT_BASE_DIR
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_BASE_DIR
        )
        
        if result.returncode == 0:
            print("✅ Retraining pipeline completed successfully")
            print("STDOUT:")
            print(result.stdout)
            
            # Parse results for downstream tasks
            lines = result.stdout.split('\n')
            retrain_id = None
            new_version = None
            
            for line in lines:
                if 'Retrain ID:' in line:
                    retrain_id = line.split(':')[-1].strip()
                elif 'New Model Version:' in line:
                    new_version = line.split(':')[-1].strip()
            
            context['task_instance'].xcom_push(key='retrain_id', value=retrain_id)
            context['task_instance'].xcom_push(key='new_model_version', value=new_version)
            
            return True
            
        else:
            print("❌ Retraining pipeline failed")
            print("STDERR:")
            print(result.stderr)
            raise Exception(f"Retraining failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error running retraining pipeline: {e}")
        raise

def validate_new_model(**context):
    """Validate the newly trained model"""
    
    print("🔍 Validating new model...")
    
    try:
        retrain_id = context['task_instance'].xcom_pull(key='retrain_id', task_ids='run_retraining')
        new_version = context['task_instance'].xcom_pull(key='new_model_version', task_ids='run_retraining')
        
        if not retrain_id or not new_version:
            raise Exception("Missing retraining information from previous task")
        
        # Check if model files exist
        model_files = [
            f"{PROJECT_BASE_DIR}/models/stage1_lgb_ensemble.pkl",
            f"{PROJECT_BASE_DIR}/models/stage2_ensemble.pkl",
            f"{PROJECT_BASE_DIR}/models/preprocessor.pkl",
            f"{PROJECT_BASE_DIR}/models/feature_metadata.pkl"
        ]
        
        missing_files = []
        for model_file in model_files:
            if not Path(model_file).exists():
                missing_files.append(model_file)
        
        if missing_files:
            raise Exception(f"Missing model files: {missing_files}")
        
        # Load and validate model registry
        registry_file = f"{PROJECT_BASE_DIR}/models/model_registry_{new_version}.json"
        if not Path(registry_file).exists():
            raise Exception(f"Model registry file not found: {registry_file}")
        
        import json
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        # Validate registry content
        required_fields = ['model_version', 'retrain_id', 'training_timestamp', 'deployment_status']
        missing_fields = [field for field in required_fields if field not in registry]
        
        if missing_fields:
            raise Exception(f"Missing registry fields: {missing_fields}")
        
        print(f"✅ Model validation successful")
        print(f"   Model Version: {new_version}")
        print(f"   Deployment Status: {registry['deployment_status']}")
        
        # Store validation results
        context['task_instance'].xcom_push(key='validation_success', value=True)
        context['task_instance'].xcom_push(key='deployment_status', value=registry['deployment_status'])
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        context['task_instance'].xcom_push(key='validation_success', value=False)
        raise

def send_completion_notification(**context):
    """Send notification about retraining completion"""
    
    print("📧 Preparing completion notification...")
    
    try:
        # Gather information from previous tasks
        retrain_id = context['task_instance'].xcom_pull(key='retrain_id', task_ids='run_retraining')
        new_version = context['task_instance'].xcom_pull(key='new_model_version', task_ids='run_retraining')
        deployment_status = context['task_instance'].xcom_pull(key='deployment_status', task_ids='validate_model')
        triggers = context['task_instance'].xcom_pull(key='retrain_triggers', task_ids='check_triggers')
        performance_metrics = context['task_instance'].xcom_pull(key='performance_metrics', task_ids='check_triggers')
        
        # Create notification content
        subject = f"F1 ML Model Retraining Completed - {new_version}"
        
        body = f"""
F1 ML Model Retraining Pipeline Completed Successfully

Retrain Details:
- Retrain ID: {retrain_id}
- New Model Version: {new_version}
- Deployment Status: {deployment_status}
- Trigger Type: {'Performance-based' if triggers else 'Scheduled'}

Performance Metrics (last 7 days):
- Stage-1 Accuracy: {performance_metrics.get('avg_stage1_acc', 'N/A'):.3f}
- Stage-2 Accuracy: {performance_metrics.get('avg_stage2_acc', 'N/A'):.3f}
- Feature Drift Score: {performance_metrics.get('avg_drift_score', 'N/A'):.3f}
- Total Predictions: {performance_metrics.get('total_predictions', 'N/A')}

Triggers Detected:
{chr(10).join(f"- {trigger}" for trigger in triggers) if triggers else "- None (scheduled retrain)"}

Next Steps:
{'- New model deployed to production' if deployment_status == 'deployed' else '- Manual review required for deployment'}
- Monitor new model performance
- Update monitoring dashboards

Pipeline executed at: {context['ds']}
"""
        
        print("✅ Notification prepared")
        print(f"Subject: {subject}")
        
        # Store notification content for email operator
        context['task_instance'].xcom_push(key='notification_subject', value=subject)
        context['task_instance'].xcom_push(key='notification_body', value=body)
        
        return True
        
    except Exception as e:
        print(f"❌ Error preparing notification: {e}")
        raise

# Define tasks

# Task 1: Check if retraining should be triggered
check_triggers_task = PythonOperator(
    task_id='check_triggers',
    python_callable=check_retrain_triggers,
    dag=dag,
    doc_md="""
    ## Check Retraining Triggers
    
    Analyzes recent model performance metrics to determine if retraining should be triggered:
    - Stage-1 and Stage-2 accuracy thresholds
    - Feature drift detection
    - Minimum prediction volume requirements
    """
)

# Task 2: Wait for new data availability (optional sensor)
wait_for_data = FileSensor(
    task_id='wait_for_data',
    filepath=f'{PROJECT_BASE_DIR}/data/features/complete_features.parquet',
    poke_interval=300,  # Check every 5 minutes
    timeout=1800,       # Timeout after 30 minutes
    dag=dag,
    doc_md="Wait for feature data to be available before starting retraining"
)

# Task 3: Run the retraining pipeline
run_retraining_task = PythonOperator(
    task_id='run_retraining',
    python_callable=run_retrain_pipeline,
    dag=dag,
    doc_md="""
    ## Execute Retraining Pipeline
    
    Runs the complete automated retraining pipeline:
    - Data ingestion and feature engineering
    - Model training and evaluation
    - Performance comparison and deployment decision
    """
)

# Task 4: Validate the new model
validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_new_model,
    dag=dag,
    doc_md="""
    ## Model Validation
    
    Validates the newly trained model:
    - Checks model file existence
    - Validates model registry
    - Confirms deployment readiness
    """
)

# Task 5: Restart API service (if model was deployed)
restart_api_service = BashOperator(
    task_id='restart_api_service',
    bash_command=f"""
    cd {PROJECT_BASE_DIR}
    if [ "$(python3 -c "import json; print(json.load(open('models/model_registry_$(ls models/model_registry_*.json | tail -1 | cut -d'_' -f3 | cut -d'.' -f1).json'))['deployment_status'])")" = "deployed" ]; then
        echo "New model deployed, restarting API service..."
        docker-compose restart f1-ml-api || echo "API service restart failed or not containerized"
    else
        echo "No deployment required, skipping API restart"
    fi
    """,
    dag=dag,
    doc_md="Restart the API service if a new model was deployed"
)

# Task 6: Prepare notification
prepare_notification_task = PythonOperator(
    task_id='prepare_notification',
    python_callable=send_completion_notification,
    dag=dag,
    doc_md="Prepare notification content about retraining completion"
)

# Task 7: Send email notification
send_email_task = EmailOperator(
    task_id='send_notification_email',
    to=['f1-ml-team@company.com'],
    subject="{{ task_instance.xcom_pull(key='notification_subject', task_ids='prepare_notification') }}",
    html_content="{{ task_instance.xcom_pull(key='notification_body', task_ids='prepare_notification') }}",
    dag=dag,
    doc_md="Send email notification about retraining completion"
)

# Task 8: Update monitoring dashboards
update_dashboards = BashOperator(
    task_id='update_dashboards',
    bash_command=f"""
    echo "Updating Grafana dashboards with new model version..."
    # In a real implementation, this would use Grafana API to update dashboards
    # For now, just log the completion
    echo "Dashboard update completed"
    """,
    dag=dag,
    doc_md="Update monitoring dashboards with new model information"
)

# Define task dependencies
check_triggers_task >> wait_for_data >> run_retraining_task
run_retraining_task >> validate_model_task
validate_model_task >> [restart_api_service, prepare_notification_task]
prepare_notification_task >> send_email_task
[restart_api_service, send_email_task] >> update_dashboards

# Add task documentation
dag.doc_md = """
# F1 ML Model Retraining DAG

This DAG orchestrates the automated retraining of F1 ML models based on performance monitoring.

## Schedule
- **Frequency**: Weekly (every Sunday)
- **Trigger Types**: Scheduled, Performance-based, Feature drift

## Pipeline Steps
1. **Check Triggers**: Analyze performance metrics to determine if retraining is needed
2. **Wait for Data**: Ensure latest feature data is available
3. **Run Retraining**: Execute the complete retraining pipeline
4. **Validate Model**: Verify new model integrity and readiness
5. **Restart Services**: Update production services if model was deployed
6. **Send Notifications**: Alert team about retraining completion
7. **Update Dashboards**: Refresh monitoring visualizations

## Monitoring
- Email notifications on success/failure
- XCom data sharing between tasks
- Comprehensive logging throughout pipeline

## Configuration
Configure the following Airflow Variables:
- `F1_ML_PROJECT_DIR`: Base directory for the F1 ML project
"""