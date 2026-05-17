"""Register model artifacts in SageMaker Model Registry."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.logging_utils import configure_logging, log_event, timed_stage


LOGGER = logging.getLogger("stages.deployment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a trained model package in SageMaker.")
    parser.add_argument("--model-package-group", required=True)
    parser.add_argument("--model-artifact-s3-uri", required=True)
    parser.add_argument("--inference-image-uri", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--approval-status", default="PendingManualApproval")
    parser.add_argument("--metrics-file", default=None)
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


_SAFE_PATTERN = re.compile(r"[^\w .:/=+\-@]")


def _safe_metadata(raw: dict) -> dict[str, str]:
    """Sanitize metrics for SageMaker CustomerMetadataProperties.

    SageMaker only allows [A-Za-z0-9 _.:/=+-@] in metadata values (max 256 chars).
    Dicts (e.g. group_auc={}) and values with curly braces are stripped out.
    """
    result = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            # Flatten non-empty dicts as key=value pairs; skip empty ones.
            if not value:
                continue
            serialized = " ".join(f"{k}={v}" for k, v in value.items())
        else:
            serialized = str(value)
        cleaned = _SAFE_PATTERN.sub("", serialized).strip()[:256]
        if cleaned:
            result[key] = cleaned
    return result


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "register_model"):
        metrics = {}
        if args.metrics_file:
            metrics = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
        if args.validation_report:
            validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
            if validation.get("status") != "passed":
                raise ValueError("Model validation report did not pass.")
            metrics.update({f"validation_{key}": value for key, value in validation.items() if key != "status"})

        request = {
            "ModelPackageGroupName": args.model_package_group,
            "ModelApprovalStatus": args.approval_status,
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": args.inference_image_uri,
                        "ModelDataUrl": args.model_artifact_s3_uri,
                    }
                ],
                "SupportedContentTypes": ["text/csv", "application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
            "CustomerMetadataProperties": _safe_metadata(metrics),
        }

        if args.dry_run:
            print(json.dumps(request, indent=2))
            return

        import boto3

        client = boto3.client("sagemaker", region_name=args.region)
        response = client.create_model_package(**request)
        log_event(LOGGER, "model_registered", model_package_arn=response["ModelPackageArn"])


if __name__ == "__main__":
    main()
