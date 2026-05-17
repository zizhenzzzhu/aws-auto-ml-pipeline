"""Auto-classify columns and generate validated features for model training."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir
from common.logging_utils import configure_logging, log_event, timed_stage
from data_pull import create_spark_session


LOGGER = logging.getLogger("stages.feature_engineering")
TEXT_POSITIVE_WORDS = {"good", "great", "excellent", "fast", "happy", "positive", "success", "best"}
TEXT_NEGATIVE_WORDS = {"bad", "poor", "slow", "sad", "negative", "fail", "failed", "worst", "error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate typed features and select a final feature matrix.")
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--feature-log-path", default="outputs/feature_log.json")
    parser.add_argument("--exclude-column", action="append", default=[])
    parser.add_argument("--enable-target-encoding", action="store_true")
    parser.add_argument("--max-category-cardinality", type=int, default=100)
    parser.add_argument("--max-text-hash-features", type=int, default=16)
    parser.add_argument("--max-interactions", type=int, default=10)
    parser.add_argument("--max-null-ratio", type=float, default=0.4)
    parser.add_argument("--max-correlation", type=float, default=0.98)
    parser.add_argument("--leakage-pattern", action="append", default=["leak", "future"])
    parser.add_argument("--time-order-column", default=None)
    parser.add_argument("--entity-column", default=None)
    parser.add_argument("--lag-column", action="append", default=[])
    return parser.parse_args()


def classify_columns(frame, label_column: str, exclude_columns: set[str], max_category_cardinality: int) -> dict[str, list[str]]:
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    row_count = max(frame.count(), 1)
    classes = {"numerical": [], "categorical": [], "datetime": [], "text": []}

    for field in frame.schema.fields:
        column = field.name
        if column == label_column or column in exclude_columns:
            continue

        dtype = field.dataType
        lower_name = column.lower()
        if isinstance(dtype, (T.DateType, T.TimestampType)):
            classes["datetime"].append(column)
        elif isinstance(dtype, T.NumericType):
            classes["numerical"].append(column)
        elif isinstance(dtype, T.StringType):
            summary = frame.select(
                F.avg(F.length(F.col(column))).alias("avg_len"),
                F.approx_count_distinct(F.col(column)).alias("distinct_count"),
            ).first()
            avg_len = float(summary["avg_len"] or 0)
            distinct_ratio = float(summary["distinct_count"] or 0) / row_count

            if any(token in lower_name for token in ("date", "time", "timestamp", "created_at", "updated_at")):
                classes["datetime"].append(column)
            elif avg_len >= 40 or distinct_ratio > 0.5:
                classes["text"].append(column)
            elif int(summary["distinct_count"] or 0) <= max_category_cardinality:
                classes["categorical"].append(column)
            else:
                classes["text"].append(column)

    return classes


def add_numerical_features(frame, columns: list[str], max_interactions: int) -> tuple[object, list[str], list[dict[str, object]]]:
    from pyspark.sql import functions as F

    feature_columns = []
    log_rows = []

    for column in columns:
        quantiles = frame.approxQuantile(column, [0.01, 0.5, 0.99], 0.01)
        if len(quantiles) != 3:
            continue

        lower, median, upper = quantiles
        stats = frame.select(F.mean(F.col(column)).alias("mean"), F.stddev(F.col(column)).alias("std")).first()
        mean = float(stats["mean"] if stats["mean"] is not None else median)
        std = float(stats["std"] or 1.0)
        if not math.isfinite(std) or std == 0:
            std = 1.0

        capped = f"{column}__capped"
        scaled = f"{column}__scaled"
        missing = f"{column}__is_missing"
        binned = f"{column}__bin"

        frame = frame.withColumn(missing, F.when(F.col(column).isNull(), 1.0).otherwise(0.0))
        frame = frame.withColumn(
            capped,
            F.when(F.col(column).isNull(), F.lit(float(median)))
            .when(F.col(column) < lower, F.lit(float(lower)))
            .when(F.col(column) > upper, F.lit(float(upper)))
            .otherwise(F.col(column).cast("double")),
        )
        frame = frame.withColumn(scaled, (F.col(capped) - F.lit(mean)) / F.lit(std))
        frame = frame.withColumn(
            binned,
            F.when(F.col(capped) <= lower, 0.0)
            .when(F.col(capped) <= median, 1.0)
            .when(F.col(capped) <= upper, 2.0)
            .otherwise(3.0),
        )
        feature_columns.extend([missing, scaled, binned])
        log_rows.append({"source": column, "features": [missing, scaled, binned], "rule": "numeric_missing_cap_scale_bin"})

    scaled_columns = [column for column in feature_columns if column.endswith("__scaled")]
    for left, right in itertools.islice(itertools.combinations(scaled_columns, 2), max_interactions):
        interaction = f"{left.replace('__scaled', '')}__x__{right.replace('__scaled', '')}"
        frame = frame.withColumn(interaction, F.col(left) * F.col(right))
        feature_columns.append(interaction)
        log_rows.append({"source": [left, right], "features": [interaction], "rule": "numeric_interaction"})

    return frame, feature_columns, log_rows


def add_categorical_features(
    frame,
    columns: list[str],
    label_column: str,
    enable_target_encoding: bool,
) -> tuple[object, list[str], list[dict[str, object]]]:
    from pyspark.ml.feature import StringIndexer
    from pyspark.sql import functions as F

    feature_columns = []
    log_rows = []
    row_count = max(frame.count(), 1)

    for column in columns:
        filled = f"{column}__filled"
        indexed = f"{column}__index"
        frequency = f"{column}__frequency"
        frame = frame.withColumn(filled, F.coalesce(F.col(column).cast("string"), F.lit("__missing__")))
        frame = StringIndexer(inputCol=filled, outputCol=indexed, handleInvalid="keep").fit(frame).transform(frame)

        counts = frame.groupBy(filled).count().withColumn(frequency, F.col("count") / F.lit(row_count)).drop("count")
        frame = frame.join(counts, on=filled, how="left")
        new_features = [indexed, frequency]

        if enable_target_encoding:
            encoded = f"{column}__target_mean"
            means = frame.groupBy(filled).agg(F.avg(F.col(label_column).cast("double")).alias(encoded))
            frame = frame.join(means, on=filled, how="left")
            new_features.append(encoded)

        feature_columns.extend(new_features)
        log_rows.append({"source": column, "features": new_features, "rule": "categorical_index_frequency_target"})

    return frame, feature_columns, log_rows


def add_datetime_features(frame, columns: list[str]) -> tuple[object, list[str], list[dict[str, object]]]:
    from pyspark.sql import functions as F

    feature_columns = []
    log_rows = []

    for column in columns:
        ts_col = f"{column}__ts"
        frame = frame.withColumn(ts_col, F.to_timestamp(F.col(column)))
        generated = [
            f"{column}__year",
            f"{column}__month",
            f"{column}__dayofweek",
            f"{column}__hour",
            f"{column}__month_sin",
            f"{column}__month_cos",
            f"{column}__dayofweek_sin",
            f"{column}__dayofweek_cos",
        ]
        frame = frame.withColumn(generated[0], F.year(F.col(ts_col)).cast("double"))
        frame = frame.withColumn(generated[1], F.month(F.col(ts_col)).cast("double"))
        frame = frame.withColumn(generated[2], F.dayofweek(F.col(ts_col)).cast("double"))
        frame = frame.withColumn(generated[3], F.hour(F.col(ts_col)).cast("double"))
        frame = frame.withColumn(generated[4], F.sin(2 * math.pi * F.col(generated[1]) / 12.0))
        frame = frame.withColumn(generated[5], F.cos(2 * math.pi * F.col(generated[1]) / 12.0))
        frame = frame.withColumn(generated[6], F.sin(2 * math.pi * F.col(generated[2]) / 7.0))
        frame = frame.withColumn(generated[7], F.cos(2 * math.pi * F.col(generated[2]) / 7.0))
        feature_columns.extend(generated)
        log_rows.append({"source": column, "features": generated, "rule": "datetime_decomposition_cyclical"})

    return frame, feature_columns, log_rows


def add_text_features(frame, columns: list[str], max_hash_features: int) -> tuple[object, list[str], list[dict[str, object]]]:
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    feature_columns = []
    log_rows = []
    row_count = max(frame.count(), 1)

    def hash_counts(text: str | None) -> list[float]:
        buckets = [0.0] * max_hash_features
        if not text:
            return buckets
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        total = max(len(words), 1)
        for word in words:
            stable_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            buckets[stable_hash % max_hash_features] += 1.0 / total
        return buckets

    hash_udf = F.udf(hash_counts, T.ArrayType(T.DoubleType()))

    for column in columns:
        text = F.coalesce(F.col(column).cast("string"), F.lit(""))
        length_feature = f"{column}__char_len"
        word_count_feature = f"{column}__word_count"
        sentiment_feature = f"{column}__sentiment_score"
        hash_array = f"{column}__hash_array"
        frame = frame.withColumn(length_feature, F.length(text).cast("double"))
        frame = frame.withColumn(word_count_feature, F.size(F.split(text, r"\s+")).cast("double"))
        sentiment_expr = F.lit(0.0)
        lower_text = F.lower(text)
        for word in TEXT_POSITIVE_WORDS:
            sentiment_expr = sentiment_expr + F.when(lower_text.contains(word), 1.0).otherwise(0.0)
        for word in TEXT_NEGATIVE_WORDS:
            sentiment_expr = sentiment_expr - F.when(lower_text.contains(word), 1.0).otherwise(0.0)
        frame = frame.withColumn(sentiment_feature, sentiment_expr)
        frame = frame.withColumn(hash_array, hash_udf(text))

        generated = [length_feature, word_count_feature, sentiment_feature]
        for index in range(max_hash_features):
            bucket = f"{column}__tfidf_hash_{index}"
            frame = frame.withColumn(bucket, F.col(hash_array).getItem(index))
            document_frequency = frame.filter(F.col(bucket) > 0).count()
            inverse_document_frequency = math.log((row_count + 1.0) / (document_frequency + 1.0)) + 1.0
            frame = frame.withColumn(bucket, F.col(bucket) * F.lit(inverse_document_frequency))
            generated.append(bucket)

        feature_columns.extend(generated)
        log_rows.append({"source": column, "features": generated, "rule": "text_length_sentiment_hashed_tfidf"})

    return frame, feature_columns, log_rows


def add_lag_rolling_features(
    frame,
    columns: list[str],
    time_order_column: str | None,
    entity_column: str | None,
) -> tuple[object, list[str], list[dict[str, object]]]:
    if not time_order_column or not columns:
        return frame, [], []

    from pyspark.sql import Window
    from pyspark.sql import functions as F

    order_column = F.col(time_order_column)
    window = Window.partitionBy(entity_column).orderBy(order_column) if entity_column else Window.orderBy(order_column)
    rolling_window = window.rowsBetween(-3, -1)
    feature_columns = []
    log_rows = []

    for column in columns:
        lag_1 = f"{column}__lag_1"
        lag_3 = f"{column}__lag_3"
        rolling_mean_3 = f"{column}__rolling_mean_3"
        frame = frame.withColumn(lag_1, F.lag(F.col(column).cast("double"), 1).over(window))
        frame = frame.withColumn(lag_3, F.lag(F.col(column).cast("double"), 3).over(window))
        frame = frame.withColumn(rolling_mean_3, F.avg(F.col(column).cast("double")).over(rolling_window))
        generated = [lag_1, lag_3, rolling_mean_3]
        feature_columns.extend(generated)
        log_rows.append(
            {
                "source": column,
                "features": generated,
                "rule": "time_ordered_lag_rolling",
                "time_order_column": time_order_column,
                "entity_column": entity_column,
            }
        )

    return frame, feature_columns, log_rows


def select_features(
    frame,
    feature_columns: list[str],
    label_column: str,
    max_null_ratio: float,
    max_correlation: float,
    leakage_patterns: list[str],
) -> tuple[list[str], list[dict[str, object]]]:
    from pyspark.sql import functions as F

    selected = []
    dropped = []
    row_count = max(frame.count(), 1)
    leakage_regex = re.compile("|".join(re.escape(pattern.lower()) for pattern in leakage_patterns))

    for column in feature_columns:
        lower_name = column.lower()
        if column == label_column or leakage_regex.search(lower_name):
            dropped.append({"feature": column, "reason": "possible_leakage"})
            continue

        stats = frame.select(
            F.count(F.col(column)).alias("non_null"),
            F.approx_count_distinct(F.col(column)).alias("distinct_count"),
        ).first()
        null_ratio = 1.0 - (float(stats["non_null"] or 0) / row_count)
        if null_ratio > max_null_ratio:
            dropped.append({"feature": column, "reason": "high_null_ratio", "null_ratio": null_ratio})
            continue
        if int(stats["distinct_count"] or 0) <= 1:
            dropped.append({"feature": column, "reason": "constant_or_no_variance"})
            continue
        selected.append(column)

    final_selected = []
    for column in selected:
        correlated_with = None
        for kept in final_selected:
            corr = frame.stat.corr(column, kept)
            if corr is not None and abs(corr) >= max_correlation:
                correlated_with = kept
                break
        if correlated_with:
            dropped.append(
                {
                    "feature": column,
                    "reason": "high_correlation",
                    "correlated_with": correlated_with,
                    "threshold": max_correlation,
                }
            )
        else:
            final_selected.append(column)

    return final_selected, dropped


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "feature_engineering"):
        spark = create_spark_session("aws-auto-ml-feature-engineering")
        frame = spark.read.parquet(args.input_uri)
        exclude_columns = set(args.exclude_column)
        classes = classify_columns(frame, args.label_column, exclude_columns, args.max_category_cardinality)

        engineered = frame
        feature_columns = []
        feature_log = []

        engineered, generated, rows = add_numerical_features(engineered, classes["numerical"], args.max_interactions)
        feature_columns.extend(generated)
        feature_log.extend(rows)

        engineered, generated, rows = add_categorical_features(
            engineered,
            classes["categorical"],
            args.label_column,
            args.enable_target_encoding,
        )
        feature_columns.extend(generated)
        feature_log.extend(rows)

        engineered, generated, rows = add_datetime_features(engineered, classes["datetime"])
        feature_columns.extend(generated)
        feature_log.extend(rows)

        engineered, generated, rows = add_text_features(engineered, classes["text"], args.max_text_hash_features)
        feature_columns.extend(generated)
        feature_log.extend(rows)

        lag_columns = args.lag_column or []
        engineered, generated, rows = add_lag_rolling_features(
            engineered,
            lag_columns,
            args.time_order_column,
            args.entity_column,
        )
        feature_columns.extend(generated)
        feature_log.extend(rows)

        selected_features, dropped_features = select_features(
            engineered,
            feature_columns,
            args.label_column,
            args.max_null_ratio,
            args.max_correlation,
            args.leakage_pattern,
        )

        output = engineered.select(*selected_features, args.label_column)
        output.write.mode("overwrite").parquet(args.output_uri)

        log_payload = {
            "label_column": args.label_column,
            "column_classes": classes,
            "generated_feature_count": len(feature_columns),
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "dropped_features": dropped_features,
            "feature_rules": feature_log,
        }
        feature_log_path = Path(args.feature_log_path)
        ensure_dir(feature_log_path.parent)
        feature_log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
        log_event(LOGGER, "feature_engineering_summary", **log_payload)


if __name__ == "__main__":
    main()
