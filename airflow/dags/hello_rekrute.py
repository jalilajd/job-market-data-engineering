import sys
sys.path.insert(0, "/opt/airflow/project")

from airflow.operators.python import PythonOperator
from datetime import datetime
from docker.types import Mount
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from src.load_postgres import load_data
with DAG(
    dag_id="rekrute_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rekrute", "spark"],
) as dag:

    process_data = DockerOperator(
        task_id="process_data",
        image="airflow-spark",
        command=["/opt/spark/bin/spark-submit",
                 "/opt/spark-apps/src/spark_processing.py"],
        docker_url="unix://var/run/docker.sock",
        network_mode="airflow_default",
        auto_remove="never",

        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="C:/Users/dell/PycharmProjects/job_market_data_engineering/src",
                target="/opt/spark-apps/src",
                type="bind",
            ),
            Mount(
                source="C:/Users/dell/PycharmProjects/job_market_data_engineering/data",
                target="/opt/spark-apps/data",
                type="bind",
            ),
        ],
    )

    load_postgres = PythonOperator(
        task_id="load_postgres",
        python_callable=load_data,
    )

    process_data >> load_postgres


