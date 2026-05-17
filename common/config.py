"""Configuration helpers shared by pipeline scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"{name} is required.")
    return value


def csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@dataclass
class PipelinePaths:
    raw_data_uri: str = field(default_factory=lambda: env("RAW_DATA_URI", "data/raw"))
    etl_data_uri: str = field(default_factory=lambda: env("ETL_DATA_URI", "data/etl"))
    feature_data_uri: str = field(default_factory=lambda: env("FEATURE_DATA_URI", "data/features"))
    processed_data_uri: str = field(default_factory=lambda: env("PROCESSED_DATA_URI", "data/processed"))
    model_dir: str = field(default_factory=lambda: env("MODEL_DIR", "outputs/model"))
    metrics_dir: str = field(default_factory=lambda: env("METRICS_DIR", "outputs/metrics"))
    feature_log_path: str = field(default_factory=lambda: env("FEATURE_LOG_PATH", "outputs/feature_log.json"))


@dataclass
class AwsSettings:
    region: str = field(default_factory=lambda: env("AWS_REGION", "us-west-2"))
    pipeline_name: str = field(default_factory=lambda: env("PIPELINE_NAME", "aws-auto-ml-pipeline"))
    cloudwatch_namespace: str = field(default_factory=lambda: env("CLOUDWATCH_NAMESPACE", "AutoMLPipeline"))
