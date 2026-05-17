"""Evaluate a trained PyTorch model against Parquet data."""

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
from common.logging_utils import configure_logging, log_event, put_metric, timed_stage
from data_pull import create_torch_dataloader
from stages.training.train_pytorch import build_model


LOGGER = logging.getLogger("stages.evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model.")
    parser.add_argument("--eval-uri", required=True)
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cloudwatch-namespace", default=None)
    parser.add_argument("--pipeline-name", default="aws-auto-ml-pipeline")
    parser.add_argument("--region", default="us-west-2")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "evaluation"):
        import torch

        model_dir = Path(args.model_dir)
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        dataloader = create_torch_dataloader(
            parquet_uri=args.eval_uri,
            feature_columns=metadata["feature_columns"],
            label_column=metadata["label_column"],
            batch_size=args.batch_size,
        )

        model = build_model(len(metadata["feature_columns"]), metadata["hidden_dim"])
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        model.eval()

        loss_fn = torch.nn.BCEWithLogitsLoss() if metadata["task"] == "binary_classification" else torch.nn.MSELoss()
        losses = []
        correct = 0
        count = 0

        with torch.no_grad():
            for features, labels in dataloader:
                labels = labels.float().view(-1, 1)
                logits = model(features)
                losses.append(float(loss_fn(logits, labels).detach().cpu()))

                if metadata["task"] == "binary_classification":
                    predicted = (torch.sigmoid(logits) >= 0.5).float()
                    correct += int((predicted == labels).sum().item())
                    count += int(labels.numel())

        metrics = {"loss": sum(losses) / max(len(losses), 1)}
        if metadata["task"] == "binary_classification" and count:
            metrics["accuracy"] = correct / count

        metrics_dir = ensure_dir(args.metrics_dir)
        (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log_event(LOGGER, "evaluation_metrics", **metrics)

        if args.cloudwatch_namespace:
            dimensions = {"PipelineName": args.pipeline_name}
            put_metric(args.cloudwatch_namespace, "EvaluationLoss", float(metrics["loss"]), dimensions, args.region)
            if "accuracy" in metrics:
                put_metric(
                    args.cloudwatch_namespace,
                    "EvaluationAccuracy",
                    float(metrics["accuracy"]),
                    dimensions,
                    args.region,
                )


if __name__ == "__main__":
    main()
