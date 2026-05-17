"""Run the sanitized data pull stage."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.logging_utils import configure_logging, timed_stage
from data_pull import DataPullConfig, DataPullStage


LOGGER = logging.getLogger("stages.data_pull")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull source data into Parquet for model development.")
    parser.add_argument("--mode", default=os.getenv("DATA_PULL_MODE", "cloud"), choices=("cloud", "basic", "local"))
    parser.add_argument("--engine", default="spark", choices=("spark", "pandas"))
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--catalog", default=os.getenv("DATA_PLATFORM_CATALOG", ""))
    parser.add_argument("--schema", default=os.getenv("DATA_PLATFORM_SCHEMA", ""))
    parser.add_argument("--secret-name", default=os.getenv("DATA_PLATFORM_SECRET_NAME"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-west-2"))
    parser.add_argument("--partition-column", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = DataPullConfig(
        mode=args.mode,
        catalog=args.catalog,
        schema=args.schema,
        secret_name=args.secret_name,
        region=args.region,
    )
    stage = DataPullStage(config)

    with timed_stage(LOGGER, "data_pull"):
        if args.engine == "spark":
            stage.write_training_parquet(
                sql_file=args.query_file,
                output_uri=args.output_uri,
                partition_columns=args.partition_column,
            )
        else:
            frame = stage.run_pandas(sql_file=args.query_file)
            frame.to_parquet(args.output_uri, index=False)


if __name__ == "__main__":
    main()
