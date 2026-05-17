"""Train a simple PyTorch model from Parquet data."""

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
from common.logging_utils import configure_logging, log_event, timed_stage
from data_pull import create_torch_dataloader


LOGGER = logging.getLogger("stages.training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch model from Parquet features.")
    parser.add_argument("--train-uri", required=True)
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument("--feature-log-path", default=None)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--task", choices=("binary_classification", "regression"), default="binary_classification")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
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


def build_model(input_dim: int, hidden_dim: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "training"):
        import torch

        feature_columns = resolve_feature_columns(args)
        model_dir = ensure_dir(args.model_dir)
        dataloader = create_torch_dataloader(
            parquet_uri=args.train_uri,
            feature_columns=feature_columns,
            label_column=args.label_column,
            batch_size=args.batch_size,
        )
        model = build_model(len(feature_columns), args.hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss() if args.task == "binary_classification" else torch.nn.MSELoss()

        last_loss = None
        for epoch in range(args.epochs):
            losses = []
            for features, labels in dataloader:
                labels = labels.float().view(-1, 1)
                predictions = model(features)
                loss = loss_fn(predictions, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            last_loss = sum(losses) / max(len(losses), 1)
            log_event(LOGGER, "epoch_finished", epoch=epoch + 1, loss=round(last_loss, 6))

        torch.save(model.state_dict(), model_dir / "model.pt")
        metadata = {
            "feature_columns": feature_columns,
            "label_column": args.label_column,
            "task": args.task,
            "hidden_dim": args.hidden_dim,
            "final_training_loss": last_loss,
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
