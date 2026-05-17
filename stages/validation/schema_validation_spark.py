"""Basic schema, row-count, and freshness checks for pulled data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir
from common.logging_utils import configure_logging, log_event, timed_stage
from data_pull import create_spark_session


LOGGER = logging.getLogger("stages.validation.schema")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pulled data before ETL.")
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--expected-column", action="append", required=True)
    parser.add_argument("--min-row-threshold", type=int, default=1000)
    parser.add_argument("--freshness-column", default=None)
    parser.add_argument("--max-data-age-days", type=int, default=7)
    parser.add_argument("--report-path", default="outputs/schema_validation.json")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "schema_validation"):
        spark = create_spark_session("aws-auto-ml-schema-validation")
        frame = spark.read.parquet(args.input_uri)

        missing_columns = sorted(set(args.expected_column) - set(frame.columns))
        if missing_columns:
            raise AssertionError(f"Missing columns: {missing_columns}")

        row_count = frame.count()
        if row_count <= args.min_row_threshold:
            raise AssertionError(f"Too few rows: {row_count}")

        freshness = None
        if args.freshness_column:
            from pyspark.sql import functions as F

            max_date = frame.agg(F.max(F.to_date(F.col(args.freshness_column))).alias("max_date")).first()["max_date"]
            if max_date is None:
                raise AssertionError(f"No valid dates found in {args.freshness_column}.")
            age_days = (date.today() - max_date).days
            freshness = {"max_date": str(max_date), "age_days": age_days}
            if age_days > args.max_data_age_days:
                raise AssertionError(f"Data is stale: max {args.freshness_column} is {max_date}.")

        report = {
            "status": "passed",
            "row_count": row_count,
            "expected_columns": args.expected_column,
            "freshness": freshness,
        }
        report_path = Path(args.report_path)
        ensure_dir(report_path.parent)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log_event(LOGGER, "schema_validation_passed", **report)


if __name__ == "__main__":
    main()
