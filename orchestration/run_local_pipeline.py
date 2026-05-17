"""Run the pipeline stages locally with subprocesses."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local pipeline stages in order.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--raw-uri", default="data/raw")
    parser.add_argument("--etl-uri", default="data/etl")
    parser.add_argument("--feature-uri", default="data/features")
    parser.add_argument("--processed-uri", default="data/processed")
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--feature-log-path", default="outputs/feature_log.json")
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument("--expected-column", action="append", default=[])
    parser.add_argument("--freshness-column", default=None)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--skip-data-pull", action="store_true")
    parser.add_argument("--skip-schema-validation", action="store_true")
    parser.add_argument("--skip-etl", action="store_true")
    parser.add_argument("--skip-data-validation", action="store_true")
    parser.add_argument("--skip-feature-engineering", action="store_true")
    parser.add_argument("--skip-model-validation", action="store_true")
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    feature_args = [item for column in args.feature_column for item in ("--feature-column", column)]

    if not args.skip_data_pull:
        run_step(
            [
                sys.executable,
                "stages/data_pull/run_data_pull.py",
                "--query-file",
                args.query_file,
                "--output-uri",
                args.raw_uri,
            ]
        )

    if not args.skip_schema_validation and args.expected_column:
        schema_command = [
            sys.executable,
            "stages/validation/schema_validation_spark.py",
            "--input-uri",
            args.raw_uri,
            "--min-row-threshold",
            "1000",
        ]
        for column in args.expected_column:
            schema_command.extend(["--expected-column", column])
        if args.freshness_column:
            schema_command.extend(["--freshness-column", args.freshness_column])
        run_step(schema_command)

    if not args.skip_etl:
        run_step(
            [
                sys.executable,
                "stages/etl/etl_spark.py",
                "--input-uri",
                args.raw_uri,
                "--output-uri",
                args.etl_uri,
                "--drop-duplicate",
            ]
        )

    if not args.skip_data_validation:
        run_step(
            [
                sys.executable,
                "stages/validation/data_validation_spark.py",
                "--input-uri",
                args.etl_uri,
                "--label-column",
                args.label_column,
                "--non-null-column",
                args.label_column,
            ]
        )

    if not args.skip_feature_engineering:
        run_step(
            [
                sys.executable,
                "stages/feature_engineering/auto_feature_engineering.py",
                "--input-uri",
                args.etl_uri,
                "--output-uri",
                args.feature_uri,
                "--label-column",
                args.label_column,
                "--feature-log-path",
                args.feature_log_path,
            ]
        )

    run_step(
        [
            sys.executable,
            "stages/preprocess/preprocess_spark.py",
            "--input-uri",
            args.feature_uri,
            "--output-uri",
            args.processed_uri,
            *feature_args,
            "--feature-log-path",
            args.feature_log_path,
            "--label-column",
            args.label_column,
            "--drop-null",
        ]
    )
    run_step(
        [
            sys.executable,
            "stages/training/train_pytorch.py",
            "--train-uri",
            args.processed_uri,
            "--model-dir",
            args.model_dir,
            *feature_args,
            "--feature-log-path",
            args.feature_log_path,
            "--label-column",
            args.label_column,
        ]
    )
    run_step(
        [
            sys.executable,
            "stages/evaluation/evaluate_pytorch.py",
            "--eval-uri",
            args.processed_uri,
            "--model-dir",
            args.model_dir,
            "--metrics-dir",
            args.metrics_dir,
        ]
    )
    if not args.skip_model_validation:
        run_step(
            [
                sys.executable,
                "stages/validation/model_validation.py",
                "--test-uri",
                args.processed_uri,
                "--model-dir",
                args.model_dir,
                "--report-path",
                "outputs/model_validation.json",
            ]
        )


if __name__ == "__main__":
    main()
