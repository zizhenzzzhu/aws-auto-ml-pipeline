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


@contextmanager
def timed_stage(logger: logging.Logger, stage_name: str) -> Iterator[None]:
    start = time.time()
    log_event(logger, "stage_started", stage=stage_name)
    try:
        yield
    except Exception:
        logger.exception(json.dumps({"event": "stage_failed", "stage": stage_name}))
        raise
    finally:
        elapsed = round(time.time() - start, 3)
        log_event(logger, "stage_finished", stage=stage_name, elapsed_seconds=elapsed)
        namespace = os.getenv("CLOUDWATCH_NAMESPACE")
        if namespace:
            try:
                put_metric(
                    namespace=namespace,
                    metric_name="StageDurationSeconds",
                    value=elapsed,
                    dimensions={
                        "PipelineName": os.getenv("PIPELINE_NAME", "aws-auto-ml-pipeline"),
                        "StageName": stage_name,
                    },
                    region=os.getenv("AWS_REGION", "us-west-2"),
                )
            except Exception as exc:
                logger.warning("Failed to publish CloudWatch duration metric: %s", exc)


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
