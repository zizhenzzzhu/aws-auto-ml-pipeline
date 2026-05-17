# AWS Auto ML Pipeline

This repository contains a sanitized scaffold for an AWS-based automated model
development pipeline. The scripts keep SQL, platform hosts, user names,
passwords, account IDs, bucket names, and secret names out of source control.

## Folder Layout

```text
common/                     Shared config, logging, and metric helpers
infrastructure/iam/          Least-privilege IAM policy generation
infrastructure/glue/         AWS Glue ETL job submission helper
monitoring/cloudwatch/       CloudWatch logs, alarms, and dashboard setup
containers/                  ECR-ready training and inference images
dags/                        Airflow orchestration DAG
deployment/                  ECS and EKS deployment templates
tests/                       Unit tests for local logic and DAG source checks
stages/data_pull/            SQL data extraction into Parquet
stages/validation/           Schema, data, and model validation gates
stages/etl/                  Post-pull cleanup and ETL
stages/feature_engineering/  Auto column typing, feature creation, selection
stages/preprocess/           PySpark cleaning and feature projection
stages/training/             PyTorch model training
stages/evaluation/           PyTorch model evaluation and metric publishing
stages/deployment/           SageMaker Model Registry registration
orchestration/               Local stage runner
queries/                     Local SQL files, ignored by git
```

IAM controls who can do what at each stage. CloudWatch observes stage logs,
metrics, alarms, and dashboards.

## Data Pull Stage

Use `data-pull.py` with `mode=cloud` for AWS execution against Databricks SQL.
Connection details are loaded from AWS Secrets Manager through
`DATA_PLATFORM_SECRET_NAME`; local SQL files are ignored by git under
`queries/*.sql`.

Required environment variables:

```powershell
$env:DATA_PLATFORM_CATALOG = "catalog_name"
$env:DATA_PLATFORM_SCHEMA = "schema_name"
$env:DATA_PLATFORM_SECRET_NAME = "secret_name_in_aws"
$env:AWS_REGION = "us-west-2"
```

The AWS secret should contain these generic Databricks fields:

```json
{
  "server_hostname": "workspace-host",
  "http_path": "/sql/1.0/warehouses/warehouse-id",
  "access_token": "token-from-secrets-manager"
}
```

Generate an IAM policy template:

```powershell
python .\infrastructure\iam\generate_iam_policy.py --output-file .\outputs\pipeline-policy.json
```

Preview an AWS Glue ETL job submission:

```powershell
python .\infrastructure\glue\submit_glue_job.py --job-name automl-etl --input-uri s3://bucket/raw/ --output-uri s3://bucket/etl/ --dry-run
```

Preview CloudWatch resources without creating them:

```powershell
python .\monitoring\cloudwatch\setup_monitoring.py --dry-run
```

Run a Spark pull and write training-ready Parquet:

```powershell
python .\data-pull.py --mode cloud --engine spark --query-file .\queries\training.sql --output-uri s3://bucket/prefix/training/
```

Spark pulls require the Databricks JDBC driver to be available to the Spark
runtime. Pandas pulls use `databricks-sql-connector`.

For small validation checks, use pandas:

```powershell
python .\data-pull.py --mode cloud --engine pandas --query-file .\queries\training.sql
```

PySpark handles extraction and large-scale preparation. PyTorch training should
consume the generated Parquet from the training container or mounted data
channel, for example through the `create_torch_dataloader` helper in
`data_pull.py`.

```python
import data_pull

dataloader = data_pull.create_torch_dataloader(
    parquet_uri="/opt/ml/input/data/training",
    feature_columns=["feature_1", "feature_2"],
    label_column="label",
)
```

## Local Stage Flow

Run the stage scripts directly:

```powershell
python .\stages\data_pull\run_data_pull.py --query-file .\queries\training.sql --output-uri .\data\raw
python .\stages\validation\schema_validation_spark.py --input-uri .\data\raw --expected-column feature_1 --expected-column feature_2 --expected-column label --freshness-column event_date
python .\stages\etl\etl_spark.py --input-uri .\data\raw --output-uri .\data\etl --drop-duplicate
python .\stages\validation\data_validation_spark.py --input-uri .\data\etl --label-column label --mostly-non-null-check feature_1:0.95 --range-check age:0:120 --set-check category:A,B,C --previous-run-uri .\data\baseline --drift-column feature_1
python .\stages\feature_engineering\auto_feature_engineering.py --input-uri .\data\etl --output-uri .\data\features --label-column label --feature-log-path .\outputs\feature_log.json
python .\stages\feature_engineering\feature_select.py --feature-log-path .\outputs\feature_log.json
python .\stages\preprocess\preprocess_spark.py --input-uri .\data\features --output-uri .\data\processed --feature-log-path .\outputs\feature_log.json --label-column label --drop-null
python .\stages\training\train_pytorch.py --train-uri .\data\processed --feature-log-path .\outputs\feature_log.json --label-column label --model-dir .\outputs\model
python .\stages\evaluation\evaluate_pytorch.py --eval-uri .\data\processed --model-dir .\outputs\model --metrics-dir .\outputs\metrics
python .\stages\validation\model_validation.py --test-uri .\data\processed --model-dir .\outputs\model --champion-auc 0.75 --report-path .\outputs\model_validation.json
```

Or run the local orchestrator:

```powershell
python .\orchestration\run_local_pipeline.py --query-file .\queries\training.sql --label-column label --previous-run-uri .\data\baseline --drift-column feature_1 --champion-auc 0.75
```

Feature engineering follows this automatic flow:

```text
input DataFrame
  -> auto column classifier: numerical, categorical, datetime, text
  -> typed rules: scaling/outliers/bins/interactions, encodings, datetime cycles, text hash/length/sentiment
  -> optional time-aware features: lag and rolling windows when a time order is provided
  -> validation and feature selection: null ratio, constants, leakage-name patterns, high correlation
  -> final feature matrix plus feature_log.json
```

When feature engineering is enabled, `--feature-column` is optional for
preprocessing and training because the selected feature list is read from the
feature log.

Register a model package after model artifacts have been uploaded to S3:

```powershell
python .\stages\deployment\register_model.py --model-package-group your-model-group --model-artifact-s3-uri s3://bucket/model/model.tar.gz --inference-image-uri account-id.dkr.ecr.region.amazonaws.com/image:tag --metrics-file .\outputs\metrics\metrics.json --validation-report .\outputs\model_validation.json --dry-run
```

## Airflow

`dags/improved_automl_pipeline.py` defines the weekly DAG:

```text
pull_data -> validate_schema -> etl -> validate_data -> feature_engineer
-> feature_select -> preprocess -> train_model -> validate_model -> register_model
```

The DAG task name is `train_model` because the current implementation trains a
PyTorch model. A SageMaker Autopilot task should be added as a separate stage if
that becomes the chosen training backend.

## Tests and CI

The repository includes focused unit tests under `tests/` and a GitHub Actions
workflow in `.github/workflows/ci.yml`.

Run local checks:

```powershell
python -m compileall -q .
pytest -q
```

The CI workflow compiles Python, runs tests, and builds the training and
inference Docker images.

## ECR, ECS, and EKS

Build and push an ECR image:

```powershell
.\containers\build_push_ecr.ps1 -RepositoryName automl-training -AccountId 111111111111 -Region us-west-2 -Dockerfile containers/training/Dockerfile.train
```

Use `deployment/ecs/task_definition.template.json` for ECS/Fargate inference
and `deployment/eks/deployment.yaml` for EKS/Kubernetes inference. ECS is
simpler for managed services; EKS is better when the platform already runs
Kubernetes or needs custom scheduling.
