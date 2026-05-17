"""Validate a candidate model before SageMaker Model Registry registration."""

from __future__ import annotations

import argparse
import json
import logging
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir
from common.logging_utils import configure_logging, emit_pipeline_metric, log_event, timed_stage
from stages.training.train_pytorch import build_model


LOGGER = logging.getLogger("stages.validation.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate candidate model quality, fairness, and latency.")
    parser.add_argument("--test-uri", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--champion-model-dir", default=None)
    parser.add_argument("--champion-metrics-file", default=None)
    parser.add_argument("--champion-auc", type=float, default=None)
    parser.add_argument("--champion-model-package-arn", default=None)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--min-auc", type=float, default=0.75)
    parser.add_argument("--champion-tolerance", type=float, default=0.01)
    parser.add_argument("--fairness-column", default=None)
    parser.add_argument("--fairness-group", action="append", default=[])
    parser.add_argument("--min-group-auc", type=float, default=0.70)
    parser.add_argument("--max-latency-ms", type=float, default=200.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--report-path", default="outputs/model_validation.json")
    return parser.parse_args()


def load_model(model_dir: Path):
    import torch

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = build_model(len(metadata["feature_columns"]), metadata["hidden_dim"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, metadata


def evaluate_auc(model, data_uri: str, metadata: dict[str, object], batch_size: int, filter_expr=None) -> float:
    import pyarrow.dataset as ds
    import torch
    from sklearn.metrics import roc_auc_score

    dataset = ds.dataset(data_uri, format="parquet")
    columns = list(metadata["feature_columns"]) + [metadata["label_column"]]
    if filter_expr is not None:
        scanner = dataset.scanner(columns=columns, filter=filter_expr, batch_size=batch_size)
    else:
        scanner = dataset.scanner(columns=columns, batch_size=batch_size)

    labels = []
    scores = []
    with torch.no_grad():
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            if frame.empty:
                continue
            features = torch.tensor(frame[metadata["feature_columns"]].values, dtype=torch.float32)
            logits = model(features).view(-1)
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(frame[metadata["label_column"]].astype(float).tolist())

    if len(set(labels)) < 2:
        raise AssertionError("AUC requires at least two label classes in the evaluated slice.")
    return float(roc_auc_score(labels, scores))


def validate_latency(model, metadata: dict[str, object], max_latency_ms: float) -> float:
    import torch

    sample_batch = torch.zeros((1, len(metadata["feature_columns"])), dtype=torch.float32)
    start = time.time()
    with torch.no_grad():
        model(sample_batch)
    latency_ms = (time.time() - start) * 1000
    if latency_ms >= max_latency_ms:
        raise AssertionError(f"Too slow: {latency_ms:.2f}ms")
    return latency_ms


def load_champion_auc(args: argparse.Namespace, test_uri: str, batch_size: int) -> float | None:
    if args.champion_auc is not None:
        return args.champion_auc

    if args.champion_metrics_file:
        metrics = json.loads(Path(args.champion_metrics_file).read_text(encoding="utf-8"))
        for key in ("auc", "candidate_auc", "validation_candidate_auc"):
            if key in metrics:
                return float(metrics[key])
        raise ValueError("Champion metrics file must include auc, candidate_auc, or validation_candidate_auc.")

    if args.champion_model_package_arn:
        import boto3

        client = boto3.client("sagemaker", region_name=args.region)
        response = client.describe_model_package(ModelPackageName=args.champion_model_package_arn)
        metadata = response.get("CustomerMetadataProperties", {})
        for key in ("auc", "candidate_auc", "validation_candidate_auc"):
            if key in metadata:
                return float(metadata[key])
        raise ValueError("Champion model package metadata does not include an AUC field.")

    if args.champion_model_dir:
        champion, champion_metadata = load_model(Path(args.champion_model_dir))
        return evaluate_auc(champion, test_uri, champion_metadata, batch_size)

    return None


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "model_validation"):
        import pyarrow.dataset as ds

        model, metadata = load_model(Path(args.model_dir))
        candidate_auc = evaluate_auc(model, args.test_uri, metadata, args.batch_size)
        emit_pipeline_metric(LOGGER, "CandidateAUC", candidate_auc, "model_validation")
        if candidate_auc < args.min_auc:
            raise AssertionError(f"AUC too low: {candidate_auc:.4f}")

        champion_auc = load_champion_auc(args, args.test_uri, args.batch_size)
        if champion_auc is not None:
            emit_pipeline_metric(LOGGER, "ChampionAUC", champion_auc, "model_validation")
            if candidate_auc < champion_auc - args.champion_tolerance:
                raise AssertionError(
                    f"New model ({candidate_auc:.3f}) worse than champion ({champion_auc:.3f})"
                )

        group_scores = {}
        if args.fairness_column and args.fairness_group:
            field = ds.field(args.fairness_column)
            for group in args.fairness_group:
                group_auc = evaluate_auc(model, args.test_uri, metadata, args.batch_size, field == group)
                emit_pipeline_metric(
                    LOGGER,
                    "FairnessGroupAUC",
                    group_auc,
                    "model_validation",
                    {"GroupName": group},
                )
                if group_auc < args.min_group_auc:
                    raise AssertionError(f"Fairness fail for {group}: AUC={group_auc:.4f}")
                group_scores[group] = group_auc

        latency_ms = validate_latency(model, metadata, args.max_latency_ms)
        emit_pipeline_metric(LOGGER, "InferenceLatencyMs", latency_ms, "model_validation")
        report = {
            "status": "passed",
            "candidate_auc": candidate_auc,
            "champion_auc": champion_auc,
            "group_auc": group_scores,
            "latency_ms": latency_ms,
        }
        report_path = Path(args.report_path)
        ensure_dir(report_path.parent)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log_event(LOGGER, "model_validation_passed", **report)


if __name__ == "__main__":
    main()
