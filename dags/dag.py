from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os

# Add utils directory to path
sys.path.append('/opt/airflow/utils')

# Import preprocessing modules
from bronze_preprocess import ingest_raw_data
from silver_preprocess import process_to_silver
from gold_preprocess import prepare_for_analytics

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'cardio_detection_pipeline',
    default_args=default_args,
    description='Cardiovascular Disease Detection Data Pipeline',
    schedule_interval=timedelta(days=1),  # Daily run
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['cardio', 'medallion', 'xgboost'],
)

# Define tasks
start = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

# Bronze layer - Raw data processing
bronze_task = PythonOperator(
    task_id='bronze_process_raw_data',
    python_callable=ingest_raw_data,
    op_kwargs={'file_path': '/opt/airflow/cardio_train.csv'},
    dag=dag,
)

# Silver layer - Data cleaning
silver_task = PythonOperator(
    task_id='silver_clean_data',
    python_callable=process_to_silver,
    op_kwargs={'bronze_path': '{{ task_instance.xcom_pull(task_ids="bronze_process_raw_data") }}'},
    dag=dag,
)

# Gold layer - Feature engineering and model preparation
gold_task = PythonOperator(
    task_id='gold_prepare_model_data',
    python_callable=prepare_for_analytics,
    op_kwargs={'silver_path': '{{ task_instance.xcom_pull(task_ids="silver_clean_data") }}'},
    dag=dag,
)

end = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Define task dependencies
start >> bronze_task >> silver_task >> gold_task >> end