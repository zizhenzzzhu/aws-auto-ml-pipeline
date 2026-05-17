"""Submit an AWS Glue ETL job for the pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.logging_utils import configure_logging, log_event, timed_stage


LOGGER = logging.getLogger("infrastructure.glue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit an AWS Glue ETL job.")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--arguments-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "glue_submit_etl"):
        job_arguments = {
            "--input_uri": args.input_uri,
            "--output_uri": args.output_uri,
        }
        if args.arguments_json:
            job_arguments.update(json.loads(args.arguments_json))

        request = {"JobName": args.job_name, "Arguments": job_arguments}
        if args.dry_run:
            print(json.dumps(request, indent=2))
            return

        import boto3

        client = boto3.client("glue", region_name=args.region)
        response = client.start_job_run(**request)
        log_event(LOGGER, "glue_job_submitted", job_name=args.job_name, job_run_id=response["JobRunId"])


if __name__ == "__main__":
    main()
