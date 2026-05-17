"""Logging and metric helpers for pipeline stages."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str, sort_keys=True))


def cloudwatch_settings() -> tuple[str | None, str, str]:
    return (
        os.getenv("CLOUDWATCH_NAMESPACE"),
        os.getenv("PIPELINE_NAME", "aws-auto-ml-pipeline"),
        os.getenv("AWS_REGION", "us-west-2"),
    )


def emit_pipeline_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    stage_name: str,
    extra_dimensions: dict[str, str] | None = None,
) -> None:
    namespace, pipeline_name, region = cloudwatch_settings()
    if not namespace:
        return

    dimensions = {"PipelineName": pipeline_name, "StageName": stage_name}
    dimensions.update(extra_dimensions or {})
    try:
        put_metric(namespace=namespace, metric_name=metric_name, value=value, dimensions=dimensions, region=region)
    except Exception as exc:
        logger.warning("Failed to publish CloudWatch metric %s: %s", metric_name, exc)


@contextmanager
def timed_stage(logger: logging.Logger, stage_name: str) -> Iterator[None]:
    start = time.time()
    log_event(logger, "stage_started", stage=stage_name)
    status = "failed"
    try:
        yield
        status = "passed"
    except Exception:
        logger.exception(json.dumps({"event": "stage_failed", "stage": stage_name}))
        raise
    finally:
        elapsed = round(time.time() - start, 3)
        log_event(logger, "stage_finished", stage=stage_name, status=status, elapsed_seconds=elapsed)
        emit_pipeline_metric(logger, "StageDurationSeconds", elapsed, stage_name)
        emit_pipeline_metric(logger, "StageSucceeded", 1.0 if status == "passed" else 0.0, stage_name)


def put_metric(namespace: str, metric_name: str, value: float, dimensions: dict[str, str], region: str) -> None:
    import boto3

    cloudwatch = boto3.client("cloudwatch", region_name=region)
    cloudwatch.put_metric_data(
        Namespace=namespace,
        MetricData=[
            {
                "MetricName": metric_name,
                "Value": value,
                "Dimensions": [{"Name": key, "Value": val} for key, val in dimensions.items()],
            }
        ],
    )
