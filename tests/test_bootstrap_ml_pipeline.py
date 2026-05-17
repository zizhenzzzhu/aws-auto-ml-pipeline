from scripts.bootstrap_ml_pipeline import dag_conf, ecr_image_uri, validate_config


def sample_config():
    return {
        "project": {"pipeline_name": "pipeline", "repo_root": "/repo"},
        "aws": {"region": "us-west-2", "s3_bucket": "real-bucket"},
        "airflow": {"dag_id": "improved_automl_pipeline"},
        "data": {
            "query_file": "queries/training.sql",
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
            "tag": "latest",
            "dockerfile": "containers/inference/Dockerfile.inference",
            "build_context": ".",
        },
    }


def test_validate_config_accepts_complete_config():
    validate_config(sample_config())


def test_ecr_image_uri_uses_account_region_repository_and_tag():
    assert (
        ecr_image_uri(sample_config())
        == "111111111111.dkr.ecr.us-west-2.amazonaws.com/automl-inference:latest"
    )


def test_dag_conf_flattens_nested_config():
    conf = dag_conf(sample_config(), "image-uri")
    assert conf["query_file"] == "queries/training.sql"
    assert conf["inference_image_uri"] == "image-uri"
    assert conf["expected_columns"] == ["feature_1", "label"]
