import pytest

from scripts.bootstrap_ml_pipeline import dag_conf, ecr_image_uri, validate_config, validate_data_source
from scripts.bootstrap_ml_pipeline import resolve_ecr_repository_name, sanitize_ecr_repository_name


def sample_config():
    return {
        "project": {"pipeline_name": "pipeline", "repo_root": "/repo"},
        "aws": {"region": "us-west-2", "s3_bucket": "real-bucket"},
        "airflow": {"dag_id": "improved_automl_pipeline"},
        "data": {
            "source_type": "s3_csv",
            "query_file": "queries/training.sql",
            "s3_input_uri": "s3://real-bucket/input/training.csv",
            "csv_header": True,
            "csv_infer_schema": True,
            "raw_uri": "s3://real-bucket/raw/",
            "etl_uri": "s3://real-bucket/etl/",
            "feature_uri": "s3://real-bucket/features/",
            "processed_uri": "s3://real-bucket/processed/",
            "previous_run_uri": "s3://real-bucket/baseline/",
            "label_column": "label",
            "freshness_column": "event_date",
            "expected_columns": ["feature_1", "label"],
            "mostly_non_null_checks": ["feature_1:0.95"],
            "range_checks": [],
            "set_checks": [],
            "drift_columns": ["feature_1"],
        },
        "model": {
            "model_dir": "/repo/outputs/model",
            "metrics_dir": "/repo/outputs/metrics",
            "feature_log_path": "/repo/outputs/feature_log.json",
            "validation_report_path": "/repo/outputs/model_validation.json",
            "model_package_group": "model-group",
            "model_artifact_s3_uri": "s3://real-bucket/model/model.tar.gz",
            "champion_auc": 0.75,
        },
        "ecr": {
            "account_id": "111111111111",
            "repository_name": "automl-inference",
            "create_new_repository_per_model": False,
            "tag": "latest",
            "dockerfile": "containers/inference/Dockerfile.inference",
            "build_context": ".",
        },
    }


def test_validate_config_accepts_complete_config():
    validate_config(sample_config())
    validate_data_source(sample_config())


def test_validate_config_requires_mwaa_environment_name_for_mwaa_provider():
    config = sample_config()
    config["airflow"]["provider"] = "mwaa"

    with pytest.raises(ValueError, match="airflow.mwaa_environment_name"):
        validate_config(config)


def test_validate_config_accepts_mwaa_provider_with_environment_name():
    config = sample_config()
    config["airflow"]["provider"] = "mwaa"
    config["airflow"]["mwaa_environment_name"] = "automl-mwaa"

    validate_config(config)


def test_ecr_image_uri_uses_account_region_repository_and_tag():
    assert (
        ecr_image_uri(sample_config())
        == "111111111111.dkr.ecr.us-west-2.amazonaws.com/automl-inference:latest"
    )


def test_dag_conf_flattens_nested_config():
    conf = dag_conf(sample_config(), "image-uri")
    assert conf["source_type"] == "s3_csv"
    assert conf["s3_input_uri"] == "s3://real-bucket/input/training.csv"
    assert conf["query_file"] == "queries/training.sql"
    assert conf["inference_image_uri"] == "image-uri"
    assert conf["expected_columns"] == ["feature_1", "label"]


def test_sanitize_ecr_repository_name_handles_spaces():
    assert sanitize_ecr_repository_name("AutoML Inference / TikTok 3DS") == "automl-inference-tiktok-3ds"


def test_resolve_ecr_repository_name_can_create_per_model_repository():
    config = sample_config()
    config["ecr"]["create_new_repository_per_model"] = True
    repository_name = resolve_ecr_repository_name(config)
    assert repository_name.startswith("automl-inference-pipeline-")
    assert " " not in repository_name
