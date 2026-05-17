"""Preprocess Parquet data with PySpark for PyTorch training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.logging_utils import configure_logging, timed_stage
from data_pull import create_spark_session


LOGGER = logging.getLogger("stages.preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and project raw Parquet data for training.")
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument("--feature-log-path", default=None)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--drop-null", action="store_true")
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--partition-column", action="append", default=[])
    return parser.parse_args()


def resolve_feature_columns(args: argparse.Namespace) -> list[str]:
    if args.feature_column:
        return args.feature_column
    if not args.feature_log_path:
        raise ValueError("Provide --feature-column or --feature-log-path.")
    payload = json.loads(Path(args.feature_log_path).read_text(encoding="utf-8"))
    selected_features = payload.get("selected_features", [])
    if not selected_features:
        raise ValueError("Feature log does not contain selected_features.")
    return selected_features


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "preprocess"):
        spark = create_spark_session("aws-auto-ml-preprocess")
        feature_columns = resolve_feature_columns(args)
        selected_columns = feature_columns + [args.label_column]
        frame = spark.read.parquet(args.input_uri).select(*selected_columns)

        if args.drop_null:
            frame = frame.dropna(subset=selected_columns)

        if args.sample_fraction is not None:
            frame = frame.sample(withReplacement=False, fraction=args.sample_fraction, seed=42)

        writer = frame.write.mode("overwrite")
        if args.partition_column:
            writer = writer.partitionBy(*args.partition_column)
        writer.parquet(args.output_uri)


if __name__ == "__main__":
    main()
