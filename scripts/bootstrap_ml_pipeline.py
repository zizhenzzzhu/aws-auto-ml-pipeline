"""Bootstrap an ML pipeline registration from ml_config.yaml."""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PLACEHOLDER_PREFIXES = ("your-", "account-id", "bucket", "example", "replace-me")


REQUIRED_FIELDS = (
    "project.pipeline_name",
    "project.repo_root",
    "aws.region",
    "aws.s3_bucket",
    "airflow.dag_id",
    "data.query_file",
    "data.source_type",
    "data.raw_uri",
    "data.etl_uri",
    "data.feature_uri",
    "data.processed_uri",
    "data.label_column",
    "data.expected_columns",
    "model.model_dir",
    "model.metrics_dir",
    "model.feature_log_path",
    "model.validation_report_path",
    "model.model_package_group",
    "model.model_artifact_s3_uri",
    "ecr.account_id",
    "ecr.repository_name",
    "ecr.tag",
    "ecr.dockerfile",
    "ecr.build_context",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate config, prepare ECR, register Airflow, and trigger the DAG.")
    parser.add_argument("--config", default="ml_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--skip-ecr", action="store_true")
    parser.add_argument("--skip-airflow-register", action="store_true")
    parser.add_argument("--skip-trigger", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install dependencies with: pip install -r requirements.txt") from exc

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return payload


def nested_get(config: dict[str, Any], dotted_path: str) -> Any:
    current: Any = config
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def validate_config(config: dict[str, Any]) -> None:
    missing = []
    placeholders = []

    for field in REQUIRED_FIELDS:
        value = nested_get(config, field)
        if value in (None, "", []):
            missing.append(field)
        elif isinstance(value, str) and value.lower().startswith(PLACEHOLDER_PREFIXES):
            placeholders.append(field)

    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
    if placeholders:
        raise ValueError(f"Replace placeholder config fields before running: {', '.join(placeholders)}")

    if airflow_provider(config) == "mwaa" and not nested_get(config, "airflow.mwaa_environment_name"):
        raise ValueError("airflow.mwaa_environment_name is required when airflow.provider is mwaa.")


def resolve_path(config_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def validate_sql_file(config: dict[str, Any], config_path: Path) -> None:
    query_file = resolve_path(config_path, nested_get(config, "data.query_file"))
    if not query_file.exists():
        raise FileNotFoundError(f"SQL query file does not exist: {query_file}")
    if query_file.suffix.lower() != ".sql":
        raise ValueError(f"Query file must be a .sql file: {query_file}")
    print("SQL file exists. Contents are ignored when data.source_type is s3_csv.")


def validate_data_source(config: dict[str, Any]) -> None:
    source_type = nested_get(config, "data.source_type")
    if source_type == "s3_csv":
        s3_input_uri = nested_get(config, "data.s3_input_uri")
        if not s3_input_uri:
            raise ValueError("data.s3_input_uri is required when data.source_type is s3_csv.")
        if isinstance(s3_input_uri, str) and "your-" in s3_input_uri.lower():
            raise ValueError("Replace placeholder data.s3_input_uri before running.")
    if source_type not in {"s3_csv", "databricks_sql", "snowflake_sql"}:
        raise ValueError("data.source_type must be one of: s3_csv, databricks_sql, snowflake_sql.")


def require_cli(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"Required CLI is not available on PATH: {name}")


def run_command(command: list[str], dry_run: bool = False, input_text: str | None = None) -> subprocess.CompletedProcess:
    printable = " ".join(command)
    if dry_run:
        print(f"[dry-run] {printable}")
        return subprocess.CompletedProcess(command, 0, "", "")
    print(printable)
    return subprocess.run(command, input=input_text, text=True, check=True, capture_output=True)


def airflow_provider(config: dict[str, Any]) -> str:
    return str(nested_get(config, "airflow.provider") or "cli").strip().lower()


def quote_airflow_arg(value: str) -> str:
    escaped = value.replace("'", "'\"'\"'")
    return f"'{escaped}'"


def validate_s3_bucket(config: dict[str, Any], dry_run: bool) -> None:
    if not dry_run:
        require_cli("aws")
    bucket = nested_get(config, "aws.s3_bucket")
    region = nested_get(config, "aws.region")
    run_command(["aws", "s3api", "head-bucket", "--bucket", bucket, "--region", region], dry_run=dry_run)


def ecr_image_uri(config: dict[str, Any]) -> str:
    account_id = nested_get(config, "ecr.account_id")
    region = nested_get(config, "aws.region")
    repository = nested_get(config, "ecr.repository_name")
    tag = nested_get(config, "ecr.tag")
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository}:{tag}"


def bool_config(config: dict[str, Any], dotted_path: str, default: bool = False) -> bool:
    value = nested_get(config, dotted_path)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sanitize_ecr_repository_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9._/-]+", "-", value.lower())
    normalized = re.sub(r"[-._/]+", "-", normalized).strip("-._/")
    return normalized or "automl-model"


def resolve_ecr_repository_name(config: dict[str, Any]) -> str:
    base_repository = nested_get(config, "ecr.repository_name")
    if not bool_config(config, "ecr.create_new_repository_per_model", default=False):
        return base_repository

    pipeline_name = nested_get(config, "project.pipeline_name")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    repository_name = sanitize_ecr_repository_name(f"{base_repository}-{pipeline_name}-{timestamp}")
    return repository_name[:256].rstrip("-._/")


def set_ecr_repository_name(config: dict[str, Any], repository_name: str) -> None:
    config.setdefault("ecr", {})["repository_name"] = repository_name


def ecr_image_exists(config: dict[str, Any], dry_run: bool) -> bool:
    repository = nested_get(config, "ecr.repository_name")
    tag = nested_get(config, "ecr.tag")
    region = nested_get(config, "aws.region")

    if dry_run:
        print(f"[dry-run] aws ecr describe-images --repository-name {repository} --image-ids imageTag={tag}")
        return False

    result = subprocess.run(
        [
            "aws",
            "ecr",
            "describe-images",
            "--repository-name",
            repository,
            "--image-ids",
            f"imageTag={tag}",
            "--region",
            region,
        ],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def ensure_ecr_repository(config: dict[str, Any], dry_run: bool) -> None:
    repository = nested_get(config, "ecr.repository_name")
    region = nested_get(config, "aws.region")
    if dry_run:
        print(f"[dry-run] aws ecr describe-repositories --repository-names {repository} --region {region}")
        return

    result = subprocess.run(
        ["aws", "ecr", "describe-repositories", "--repository-names", repository, "--region", region],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        run_command(["aws", "ecr", "create-repository", "--repository-name", repository, "--region", region])


def build_and_push_ecr(config: dict[str, Any], config_path: Path, dry_run: bool, force_build: bool) -> str:
    if not dry_run:
        require_cli("aws")
        require_cli("docker")
    image_uri = ecr_image_uri(config)

    ensure_ecr_repository(config, dry_run)
    if ecr_image_exists(config, dry_run) and not force_build:
        print(f"ECR image already exists: {image_uri}")
        return image_uri

    account_id = nested_get(config, "ecr.account_id")
    region = nested_get(config, "aws.region")
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    dockerfile = str(resolve_path(config_path, nested_get(config, "ecr.dockerfile")))
    build_context = str(resolve_path(config_path, nested_get(config, "ecr.build_context")))

    if dry_run:
        print(f"[dry-run] aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {registry}")
    else:
        token = subprocess.run(
            ["aws", "ecr", "get-login-password", "--region", region],
            text=True,
            check=True,
            capture_output=True,
        ).stdout
        run_command(["docker", "login", "--username", "AWS", "--password-stdin", registry], input_text=token)

    run_command(["docker", "build", "-f", dockerfile, "-t", image_uri, build_context], dry_run=dry_run)
    run_command(["docker", "push", image_uri], dry_run=dry_run)
    return image_uri


def dag_conf(config: dict[str, Any], image_uri: str) -> dict[str, Any]:
    return {
        "repo_root": nested_get(config, "project.repo_root"),
        "query_file": nested_get(config, "data.query_file"),
        "source_type": nested_get(config, "data.source_type"),
        "s3_input_uri": nested_get(config, "data.s3_input_uri"),
        "csv_header": str(nested_get(config, "data.csv_header") if nested_get(config, "data.csv_header") is not None else True).lower(),
        "csv_infer_schema": str(
            nested_get(config, "data.csv_infer_schema")
            if nested_get(config, "data.csv_infer_schema") is not None
            else True
        ).lower(),
        "raw_uri": nested_get(config, "data.raw_uri"),
        "etl_uri": nested_get(config, "data.etl_uri"),
        "feature_uri": nested_get(config, "data.feature_uri"),
        "processed_uri": nested_get(config, "data.processed_uri"),
        "model_dir": nested_get(config, "model.model_dir"),
        "metrics_dir": nested_get(config, "model.metrics_dir"),
        "label_column": nested_get(config, "data.label_column"),
        "feature_log_path": nested_get(config, "model.feature_log_path"),
        "model_validation_path": nested_get(config, "model.validation_report_path"),
        "model_package_group": nested_get(config, "model.model_package_group"),
        "model_artifact_s3_uri": nested_get(config, "model.model_artifact_s3_uri"),
        "inference_image_uri": image_uri,
        "previous_run_uri": nested_get(config, "data.previous_run_uri"),
        "champion_auc": str(nested_get(config, "model.champion_auc")),
        "expected_columns": nested_get(config, "data.expected_columns"),
        "freshness_column": nested_get(config, "data.freshness_column"),
        "mostly_non_null_checks": nested_get(config, "data.mostly_non_null_checks") or [],
        "range_checks": nested_get(config, "data.range_checks") or [],
        "set_checks": nested_get(config, "data.set_checks") or [],
        "drift_columns": nested_get(config, "data.drift_columns") or [],
    }


def write_dag_conf(conf: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(conf, indent=2), encoding="utf-8")
    print(f"Wrote Airflow DAG config: {output_path}")


def invoke_mwaa_cli(config: dict[str, Any], command: str, dry_run: bool) -> None:
    environment_name = nested_get(config, "airflow.mwaa_environment_name")
    region = nested_get(config, "aws.region")

    if dry_run:
        print(f"[dry-run] aws mwaa create-cli-token --name {environment_name} --region {region}")
        print(f"[dry-run] POST https://<MWAA WebServerHostname>/aws_mwaa/cli")
        print(f"[dry-run] Airflow CLI command: {command}")
        return

    require_cli("aws")
    token_result = run_command(["aws", "mwaa", "create-cli-token", "--name", environment_name, "--region", region])
    token_payload = json.loads(token_result.stdout)
    request = Request(
        f"https://{token_payload['WebServerHostname']}/aws_mwaa/cli",
        data=command.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token_payload['CliToken']}",
            "Content-Type": "text/plain",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MWAA CLI request failed with HTTP {exc.code}: {body}") from exc

    stdout = base64.b64decode(response_payload.get("stdout") or b"").decode("utf-8", errors="replace")
    stderr = base64.b64decode(response_payload.get("stderr") or b"").decode("utf-8", errors="replace")
    if stdout:
        print(stdout.strip())
    if stderr:
        print(stderr.strip(), file=sys.stderr)
        raise RuntimeError("MWAA Airflow CLI command failed.")


def register_airflow(config: dict[str, Any], conf: dict[str, Any], dry_run: bool) -> None:
    variable_name = nested_get(config, "airflow.variable_name") or "automl_pipeline_config"
    payload = json.dumps(conf)
    if airflow_provider(config) == "mwaa":
        invoke_mwaa_cli(config, f"variables set {variable_name} {quote_airflow_arg(payload)}", dry_run)
        return

    if not dry_run:
        require_cli("airflow")
    run_command(["airflow", "variables", "set", variable_name, payload], dry_run=dry_run)


def trigger_airflow(config: dict[str, Any], conf: dict[str, Any], dry_run: bool) -> None:
    dag_id = nested_get(config, "airflow.dag_id")
    payload = json.dumps(conf)
    if airflow_provider(config) == "mwaa":
        invoke_mwaa_cli(config, f"dags trigger {dag_id} --conf {quote_airflow_arg(payload)}", dry_run)
        return

    if not dry_run:
        require_cli("airflow")
    run_command(["airflow", "dags", "trigger", dag_id, "--conf", payload], dry_run=dry_run)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    print("1. Validate config")
    validate_config(config)
    validate_data_source(config)

    print("2. Validate SQL file")
    validate_sql_file(config, config_path)

    print("3. Validate S3 bucket")
    validate_s3_bucket(config, args.dry_run)

    print("4. Validate ECR image")
    repository_name = resolve_ecr_repository_name(config)
    set_ecr_repository_name(config, repository_name)
    if bool_config(config, "ecr.create_new_repository_per_model", default=False):
        print(f"Using new ECR repository for this model run: {repository_name}")

    image_uri = nested_get(config, "ecr.image_uri") or ecr_image_uri(config)
    if not args.skip_ecr:
        image_uri = build_and_push_ecr(config, config_path, args.dry_run, args.force_build)

    conf = dag_conf(config, image_uri)
    conf_path = resolve_path(config_path, nested_get(config, "airflow.generated_conf_path") or "outputs/airflow_dag_conf.json")
    write_dag_conf(conf, conf_path)

    print("5. Register in Airflow")
    if not args.skip_airflow_register:
        register_airflow(config, conf, args.dry_run)

    print("6. Trigger pipeline")
    if not args.skip_trigger:
        trigger_airflow(config, conf, args.dry_run)

    print("Pipeline submitted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
