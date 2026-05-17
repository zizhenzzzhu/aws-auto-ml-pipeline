"""Airflow DAG for the improved automated ML pipeline."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


DEFAULT_ARGS = {"owner": "ml-platform", "retries": 1}


def cfg(name: str) -> str:
    return "{{ dag_run.conf.get('" + name + "', params." + name + ") if dag_run else params." + name + " }}"


def qcfg(name: str) -> str:
    return '"' + cfg(name) + '"'


def cli_list_cfg(name: str, flag: str) -> str:
    values = "dag_run.conf.get('" + name + "', params." + name + ") if dag_run else params." + name
    return "{% set values = " + values + " %}{% for value in values %} " + flag + ' "{{ value }}"{% endfor %}'


with DAG(
    dag_id="improved_automl_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    params={
        "repo_root": "/opt/airflow/repo",
        "query_file": "queries/training.sql",
        "source_type": "s3_csv",
        "s3_input_uri": "s3://bucket/input/training.csv",
        "csv_header": "true",
        "csv_infer_schema": "true",
        "raw_uri": "s3://bucket/pipeline/raw/",
        "etl_uri": "s3://bucket/pipeline/etl/",
        "feature_uri": "s3://bucket/pipeline/features/",
        "processed_uri": "s3://bucket/pipeline/processed/",
        "model_dir": "/opt/airflow/repo/outputs/model",
        "metrics_dir": "/opt/airflow/repo/outputs/metrics",
        "label_column": "label",
        "feature_log_path": "/opt/airflow/repo/outputs/feature_log.json",
        "model_validation_path": "/opt/airflow/repo/outputs/model_validation.json",
        "model_package_group": "your-model-package-group",
        "model_artifact_s3_uri": "s3://bucket/model/model.tar.gz",
        "inference_image_uri": "account-id.dkr.ecr.region.amazonaws.com/image:tag",
        "previous_run_uri": "s3://bucket/pipeline/baseline/",
        "champion_auc": "0.75",
        "expected_columns": ["feature_1", "feature_2", "label"],
        "freshness_column": "event_date",
        "mostly_non_null_checks": ["feature_1:0.95"],
        "range_checks": [],
        "set_checks": [],
        "drift_columns": ["feature_1"],
    },
) as dag:
    pull_data = BashOperator(
        task_id="pull_data",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/data_pull/run_data_pull.py "
            f"--source-type {qcfg('source_type')} "
            f"--query-file {qcfg('query_file')} "
            f"--output-uri {qcfg('raw_uri')} "
            f"--s3-input-uri {qcfg('s3_input_uri')} "
            f"--csv-header {qcfg('csv_header')} "
            f"--csv-infer-schema {qcfg('csv_infer_schema')}"
        ),
    )

    validate_schema = BashOperator(
        task_id="validate_schema",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/validation/schema_validation_spark.py "
            f"--input-uri {qcfg('raw_uri')} "
            f"{cli_list_cfg('expected_columns', '--expected-column')} "
            f"--freshness-column {qcfg('freshness_column')}"
        ),
    )

    etl = BashOperator(
        task_id="etl",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/etl/etl_spark.py "
            f"--input-uri {qcfg('raw_uri')} "
            f"--output-uri {qcfg('etl_uri')} "
            "--drop-duplicate"
        ),
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/validation/data_validation_spark.py "
            f"--input-uri {qcfg('etl_uri')} "
            f"--label-column {qcfg('label_column')} "
            f"{cli_list_cfg('mostly_non_null_checks', '--mostly-non-null-check')} "
            f"{cli_list_cfg('range_checks', '--range-check')} "
            f"{cli_list_cfg('set_checks', '--set-check')} "
            f"--previous-run-uri {qcfg('previous_run_uri')} "
            f"{cli_list_cfg('drift_columns', '--drift-column')}"
        ),
    )

    feature_engineer = BashOperator(
        task_id="feature_engineer",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/feature_engineering/auto_feature_engineering.py "
            f"--input-uri {qcfg('etl_uri')} "
            f"--output-uri {qcfg('feature_uri')} "
            f"--label-column {qcfg('label_column')} "
            f"--feature-log-path {qcfg('feature_log_path')}"
        ),
    )

    feature_select = BashOperator(
        task_id="feature_select",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/feature_engineering/feature_select.py "
            f"--feature-log-path {qcfg('feature_log_path')} "
            "--min-selected-features 1"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/training/train_pytorch.py "
            f"--train-uri {qcfg('processed_uri')} "
            f"--feature-log-path {qcfg('feature_log_path')} "
            f"--label-column {qcfg('label_column')} "
            f"--model-dir {qcfg('model_dir')}"
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/evaluation/evaluate_pytorch.py "
            f"--eval-uri {qcfg('processed_uri')} "
            f"--model-dir {qcfg('model_dir')} "
            f"--metrics-dir {qcfg('metrics_dir')}"
        ),
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/preprocess/preprocess_spark.py "
            f"--input-uri {qcfg('feature_uri')} "
            f"--output-uri {qcfg('processed_uri')} "
            f"--feature-log-path {qcfg('feature_log_path')} "
            f"--label-column {qcfg('label_column')} "
            "--drop-null"
        ),
    )

    validate_model = BashOperator(
        task_id="validate_model",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/validation/model_validation.py "
            f"--test-uri {qcfg('processed_uri')} "
            f"--model-dir {qcfg('model_dir')} "
            f"--champion-auc {qcfg('champion_auc')} "
            f"--report-path {qcfg('model_validation_path')}"
        ),
    )

    package_model = BashOperator(
        task_id="package_model",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/deployment/package_model.py "
            f"--model-dir {qcfg('model_dir')} "
            f"--model-artifact-s3-uri {qcfg('model_artifact_s3_uri')}"
        ),
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            f"cd {qcfg('repo_root')} && "
            "python stages/deployment/register_model.py "
            f"--model-package-group {qcfg('model_package_group')} "
            f"--model-artifact-s3-uri {qcfg('model_artifact_s3_uri')} "
            f"--inference-image-uri {qcfg('inference_image_uri')} "
            f"--metrics-file \"{cfg('metrics_dir')}/metrics.json\" "
            f"--validation-report {qcfg('model_validation_path')}"
        ),
    )

    pull_data >> validate_schema >> etl >> validate_data >> feature_engineer >> feature_select
    feature_select >> preprocess >> train_model >> evaluate_model >> validate_model >> package_model >> register_model
