"""ETL stage after data pull and before feature engineering."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.logging_utils import configure_logging, log_event, timed_stage
from data_pull import create_spark_session


LOGGER = logging.getLogger("stages.etl")


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unnamed_column"


def unique_names(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output = []
    for column in columns:
        base = normalize_name(column)
        counts[base] = counts.get(base, 0) + 1
        output.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic ETL cleanup on pulled Parquet data.")
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--drop-duplicate", action="store_true")
    parser.add_argument("--dedupe-key", action="append", default=[])
    parser.add_argument("--required-column", action="append", default=[])
    parser.add_argument("--max-null-ratio", type=float, default=0.95)
    parser.add_argument("--partition-column", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "etl"):
        from pyspark.sql import functions as F
        from pyspark.sql import types as T

        spark = create_spark_session("aws-auto-ml-etl")
        frame = spark.read.parquet(args.input_uri)
        original_columns = frame.columns
        frame = frame.toDF(*unique_names(frame.columns))
        rename_map = dict(zip(original_columns, frame.columns))

        for field in frame.schema.fields:
            if isinstance(field.dataType, T.StringType):
                frame = frame.withColumn(field.name, F.trim(F.col(field.name)))

        if args.required_column:
            frame = frame.dropna(subset=args.required_column)

        if args.drop_duplicate:
            frame = frame.dropDuplicates(args.dedupe_key or None)

        total_rows = max(frame.count(), 1)
        keep_columns = []
        dropped_columns = []
        for column in frame.columns:
            null_count = frame.filter(F.col(column).isNull()).count()
            null_ratio = null_count / total_rows
            if null_ratio <= args.max_null_ratio:
                keep_columns.append(column)
            else:
                dropped_columns.append({"column": column, "reason": "high_null_ratio", "null_ratio": null_ratio})

        frame = frame.select(*keep_columns)
        log_event(
            LOGGER,
            "etl_summary",
            rows=total_rows,
            input_columns=len(original_columns),
            output_columns=len(keep_columns),
            renamed_columns=rename_map,
            dropped_columns=dropped_columns,
        )

        writer = frame.write.mode("overwrite")
        if args.partition_column:
            writer = writer.partitionBy(*args.partition_column)
        writer.parquet(args.output_uri)


if __name__ == "__main__":
    main()
