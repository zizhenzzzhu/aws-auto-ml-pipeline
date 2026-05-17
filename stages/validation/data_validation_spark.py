"""Run data-quality checks, optional Great Expectations checks, imbalance, and drift."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir
from common.logging_utils import configure_logging, emit_pipeline_metric, log_event, timed_stage
from data_pull import create_spark_session


LOGGER = logging.getLogger("stages.validation.data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ETL data before feature engineering.")
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--non-null-column", action="append", default=[])
    parser.add_argument("--mostly-non-null-check", action="append", default=[], help="Format: column:0.95")
    parser.add_argument("--range-check", action="append", default=[], help="Format: column:min:max")
    parser.add_argument("--set-check", action="append", default=[], help="Format: column:value1,value2,value3")
    parser.add_argument("--min-minority-ratio", type=float, default=0.05)
    parser.add_argument("--previous-run-uri", default=None)
    parser.add_argument("--drift-column", action="append", default=[])
    parser.add_argument("--min-drift-p-value", type=float, default=0.05)
    parser.add_argument("--report-path", default="outputs/data_validation.json")
    return parser.parse_args()


def run_great_expectations_if_available(frame, args: argparse.Namespace) -> dict[str, object]:
    try:
        import great_expectations as ge
    except ImportError:
        return {"enabled": False, "reason": "great_expectations_not_installed"}

    try:
        gdf = ge.dataset.SparkDFDataset(frame)
    except AttributeError:
        return {"enabled": False, "reason": "spark_dataset_api_not_available"}

    gdf.expect_column_values_to_not_be_null(args.label_column)
    for column in args.non_null_column:
        gdf.expect_column_values_to_not_be_null(column, mostly=0.95)
    for raw_check in args.mostly_non_null_check:
        column, mostly = raw_check.split(":", 1)
        gdf.expect_column_values_to_not_be_null(column, mostly=float(mostly))
    for raw_check in args.range_check:
        column, minimum, maximum = raw_check.split(":", 2)
        gdf.expect_column_values_to_be_between(column, float(minimum), float(maximum))
    for raw_check in args.set_check:
        column, values = raw_check.split(":", 1)
        gdf.expect_column_values_to_be_in_set(column, values.split(","))

    result = gdf.validate()
    return {"enabled": True, "success": bool(result.get("success")), "result": result}


def validate_class_balance(frame, label_column: str, min_minority_ratio: float) -> dict[str, object]:
    label_dist = frame.groupBy(label_column).count().collect()
    total_rows = max(sum(row["count"] for row in label_dist), 1)
    minority_ratio = min(row["count"] for row in label_dist) / total_rows
    if minority_ratio <= min_minority_ratio:
        raise AssertionError(f"Severe class imbalance: {minority_ratio:.2%}")
    return {
        "minority_ratio": minority_ratio,
        "distribution": {str(row[label_column]): row["count"] for row in label_dist},
    }


def validate_quality_rules(frame, args: argparse.Namespace) -> dict[str, object]:
    from pyspark.sql import functions as F

    reports = {"non_null": [], "range": [], "set": []}

    for column in [args.label_column, *args.non_null_column]:
        null_count = frame.filter(F.col(column).isNull()).count()
        if null_count:
            raise AssertionError(f"{column} contains {null_count} null values.")
        reports["non_null"].append({"column": column, "null_count": null_count})

    row_count = max(frame.count(), 1)
    for raw_check in args.mostly_non_null_check:
        column, mostly = raw_check.split(":", 1)
        mostly_value = float(mostly)
        non_null_count = frame.filter(F.col(column).isNotNull()).count()
        non_null_ratio = non_null_count / row_count
        if non_null_ratio < mostly_value:
            raise AssertionError(f"{column} non-null ratio {non_null_ratio:.2%} is below {mostly_value:.2%}.")
        reports["non_null"].append({"column": column, "non_null_ratio": non_null_ratio, "minimum": mostly_value})

    for raw_check in args.range_check:
        column, minimum, maximum = raw_check.split(":", 2)
        minimum_value = float(minimum)
        maximum_value = float(maximum)
        invalid_count = frame.filter((F.col(column) < minimum_value) | (F.col(column) > maximum_value)).count()
        if invalid_count:
            raise AssertionError(f"{column} has {invalid_count} values outside [{minimum}, {maximum}].")
        reports["range"].append({"column": column, "min": minimum_value, "max": maximum_value})

    for raw_check in args.set_check:
        column, values = raw_check.split(":", 1)
        allowed = values.split(",")
        invalid_count = frame.filter(~F.col(column).isin(allowed)).count()
        if invalid_count:
            raise AssertionError(f"{column} has {invalid_count} values outside allowed set.")
        reports["set"].append({"column": column, "allowed_values": allowed})

    return reports


def validate_drift(
    spark,
    frame,
    previous_run_uri: str | None,
    columns: list[str],
    min_p_value: float,
    logger: logging.Logger | None = None,
) -> list[dict[str, object]]:
    if not previous_run_uri or not columns:
        return []

    from scipy.stats import ks_2samp

    previous = spark.read.parquet(previous_run_uri)
    reports = []
    for column in columns:
        current_values = [row[column] for row in frame.select(column).dropna().limit(10000).collect()]
        previous_values = [row[column] for row in previous.select(column).dropna().limit(10000).collect()]
        if not current_values or not previous_values:
            continue
        stat, p_value = ks_2samp(current_values, previous_values)
        if logger:
            emit_pipeline_metric(
                logger,
                "DriftPValue",
                float(p_value),
                "data_validation",
                {"FeatureName": column},
            )
            emit_pipeline_metric(
                logger,
                "DriftKSStatistic",
                float(stat),
                "data_validation",
                {"FeatureName": column},
            )
        if p_value <= min_p_value:
            raise AssertionError(f"Drift detected in {column}: p={p_value:.6f}")
        reports.append({"column": column, "ks_statistic": float(stat), "p_value": float(p_value)})
    return reports


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "data_validation"):
        spark = create_spark_session("aws-auto-ml-data-validation")
        frame = spark.read.parquet(args.input_uri)
        quality_report = validate_quality_rules(frame, args)
        ge_report = run_great_expectations_if_available(frame, args)
        balance_report = validate_class_balance(frame, args.label_column, args.min_minority_ratio)
        emit_pipeline_metric(LOGGER, "MinorityClassRatio", float(balance_report["minority_ratio"]), "data_validation")
        drift_report = validate_drift(
            spark,
            frame,
            args.previous_run_uri,
            args.drift_column,
            args.min_drift_p_value,
            LOGGER,
        )

        report = {
            "status": "passed",
            "quality_rules": quality_report,
            "great_expectations": ge_report,
            "class_balance": balance_report,
            "drift": drift_report,
        }
        report_path = Path(args.report_path)
        ensure_dir(report_path.parent)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log_event(LOGGER, "data_validation_passed", **report)


if __name__ == "__main__":
    main()
