"""
Pandas-based ETL + feature selection for local testing without PySpark.

Replicates what etl_spark → auto_feature_engineering → preprocess_spark do,
using only pandas and pyarrow so the training/eval/deployment stages can be
tested on a laptop without a Spark cluster.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir
from common.logging_utils import configure_logging, log_event, timed_stage

LOGGER = logging.getLogger("scripts.pandas_preprocess")

# Columns that are identifiers, free-text, or timestamps — not useful as model features.
NON_FEATURE_TYPES = {"object"}
EXCLUDE_COLUMNS = {
    "bank_transaction_id", "card_token", "user_transaction_time",
    "user_tm", "mid", "dispute_dt", "rules_triggered",
    "type", "updated_trans_type", "mcc",
    "merchant_name", "cleaned_merchant_name", "country_code",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pandas ETL + feature selection for local testing.")
    parser.add_argument("--input-uri",  required=True, help="Path to raw Parquet file or directory.")
    parser.add_argument("--output-uri", required=True, help="Path to write processed Parquet.")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--feature-log-path", default="outputs/feature_log.json")
    parser.add_argument("--max-null-ratio", type=float, default=0.95,
                        help="Drop columns whose null ratio exceeds this threshold.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "pandas_preprocess"):
        import pandas as pd

        df = pd.read_parquet(args.input_uri)
        original_shape = df.shape
        log_event(LOGGER, "loaded", rows=df.shape[0], columns=df.shape[1])

        # ── ETL: drop high-null columns ──────────────────────────────────────
        null_ratios = df.isnull().mean()
        high_null = null_ratios[null_ratios > args.max_null_ratio].index.tolist()
        df = df.drop(columns=high_null)
        log_event(LOGGER, "dropped_high_null", columns=high_null)

        # ── ETL: drop duplicate rows ─────────────────────────────────────────
        before_dedup = len(df)
        df = df.drop_duplicates()
        log_event(LOGGER, "deduplication", removed=before_dedup - len(df))

        # ── Feature selection: keep numeric columns only ──────────────────────
        numeric_cols = [
            col for col in df.columns
            if df[col].dtype.kind in ("i", "f", "u")  # int, float, unsigned
            and col != args.label_column
            and col not in EXCLUDE_COLUMNS
        ]

        # Drop constant columns (no signal)
        constant = [col for col in numeric_cols if df[col].nunique() <= 1]
        numeric_cols = [col for col in numeric_cols if col not in constant]
        log_event(LOGGER, "dropped_constant", columns=constant)

        # Fill remaining nulls with column median
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Standardize to zero-mean unit-variance so the neural network trains stably.
        # This mirrors what auto_feature_engineering.py does with __scaled columns.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        selected_features = sorted(numeric_cols)
        log_event(LOGGER, "selected_features", count=len(selected_features))

        # ── Write feature log ─────────────────────────────────────────────────
        feature_log = {
            "label_column": args.label_column,
            "selected_features": selected_features,
            "selected_feature_count": len(selected_features),
            "generated_feature_count": len(selected_features),
            "dropped_high_null_columns": high_null,
            "dropped_constant_columns": constant,
            "column_classes": {"numerical": selected_features},
            "feature_rules": [{"rule": "pandas_numeric_passthrough", "features": selected_features}],
            "dropped_features": [],
        }
        feature_log_path = Path(args.feature_log_path)
        ensure_dir(feature_log_path.parent)
        feature_log_path.write_text(json.dumps(feature_log, indent=2), encoding="utf-8")
        log_event(LOGGER, "wrote_feature_log", path=str(feature_log_path))

        # ── Write processed Parquet ───────────────────────────────────────────
        output_cols = selected_features + [args.label_column]
        processed = df[output_cols].dropna(subset=[args.label_column])

        output_path = Path(args.output_uri)
        ensure_dir(output_path.parent)
        processed.to_parquet(output_path, index=False)

        log_event(
            LOGGER, "preprocessing_complete",
            input_shape=list(original_shape),
            output_shape=list(processed.shape),
            output_path=str(output_path),
        )
        print(f"\nProcessed {processed.shape[0]} rows x {processed.shape[1]} columns")
        print(f"Features: {len(selected_features)}  |  Label: {args.label_column}")
        print(f"Output:   {output_path}")
        print(f"Feature log: {feature_log_path}")


if __name__ == "__main__":
    main()
