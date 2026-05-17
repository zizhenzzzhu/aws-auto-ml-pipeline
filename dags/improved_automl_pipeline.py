"""Airflow DAG for the improved automated ML pipeline."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


DEFAULT_ARGS = {"owner": "ml-platform", "retries": 1}


with DAG(
    dag_id="improved_automl_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    params={
        "repo_root": "/opt/airflow/dags/repo",
        "query_file": "queries/training.sql",
        "raw_uri": "s3://bucket/pipeline/raw/",
        "etl_uri": "s3://bucket/pipeline/etl/",
        "feature_uri": "s3://bucket/pipeline/features/",
        "processed_uri": "s3://bucket/pipeline/processed/",
        "model_dir": "/opt/airflow/dags/repo/outputs/model",
        "metrics_dir": "/opt/airflow/dags/repo/outputs/metrics",
        "label_column": "label",
        "feature_log_path": "/opt/airflow/dags/repo/outputs/feature_log.json",
        "model_validation_path": "/opt/airflow/dags/repo/outputs/model_validation.json",
        "model_package_group": "your-model-package-group",
        "model_artifact_s3_uri": "s3://bucket/model/model.tar.gz",
        "inference_image_uri": "account-id.dkr.ecr.region.amazonaws.com/image:tag",
    },
) as dag:
    pull_data = BashOperator(
        task_id="pull_data",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/data_pull/run_data_pull.py "
            "--query-file {{ params.query_file }} "
            "--output-uri {{ params.raw_uri }}"
        ),
    )

    validate_schema = BashOperator(
        task_id="validate_schema",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/validation/schema_validation_spark.py "
            "--input-uri {{ params.raw_uri }} "
            "--expected-column feature_1 "
            "--expected-column feature_2 "
            "--expected-column {{ params.label_column }} "
            "--freshness-column event_date"
        ),
    )

    etl = BashOperator(
        task_id="etl",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/etl/etl_spark.py "
            "--input-uri {{ params.raw_uri }} "
            "--output-uri {{ params.etl_uri }} "
            "--drop-duplicate"
        ),
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/validation/data_validation_spark.py "
            "--input-uri {{ params.etl_uri }} "
            "--label-column {{ params.label_column }} "
            "--mostly-non-null-check feature_1:0.95"
        ),
    )

    feature_engineer = BashOperator(
        task_id="feature_engineer",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/feature_engineering/auto_feature_engineering.py "
            "--input-uri {{ params.etl_uri }} "
            "--output-uri {{ params.feature_uri }} "
            "--label-column {{ params.label_column }} "
            "--feature-log-path {{ params.feature_log_path }}"
        ),
    )

    feature_select = BashOperator(
        task_id="feature_select",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/feature_engineering/feature_select.py "
            "--feature-log-path {{ params.feature_log_path }}"
        ),
    )

    train_autopilot = BashOperator(
        task_id="train_autopilot",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/training/train_pytorch.py "
            "--train-uri {{ params.processed_uri }} "
            "--feature-log-path {{ params.feature_log_path }} "
            "--label-column {{ params.label_column }} "
            "--model-dir {{ params.model_dir }}"
        ),
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/preprocess/preprocess_spark.py "
            "--input-uri {{ params.feature_uri }} "
            "--output-uri {{ params.processed_uri }} "
            "--feature-log-path {{ params.feature_log_path }} "
            "--label-column {{ params.label_column }} "
            "--drop-null"
        ),
    )

    validate_model = BashOperator(
        task_id="validate_model",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/validation/model_validation.py "
            "--test-uri {{ params.processed_uri }} "
            "--model-dir {{ params.model_dir }} "
            "--report-path {{ params.model_validation_path }}"
        ),
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python stages/deployment/register_model.py "
            "--model-package-group {{ params.model_package_group }} "
            "--model-artifact-s3-uri {{ params.model_artifact_s3_uri }} "
            "--inference-image-uri {{ params.inference_image_uri }} "
            "--metrics-file {{ params.metrics_dir }}/metrics.json "
            "--validation-report {{ params.model_validation_path }}"
        ),
    )

    pull_data >> validate_schema >> etl >> validate_data >> feature_engineer >> feature_select
    feature_select >> preprocess >> train_autopilot >> validate_model >> register_model
