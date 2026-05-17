"""Feature-selection gate that validates the auto feature log."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.logging_utils import configure_logging, log_event, timed_stage


LOGGER = logging.getLogger("stages.feature_engineering.selection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selected features from feature_log.json.")
    parser.add_argument("--feature-log-path", required=True)
    parser.add_argument("--min-selected-features", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "feature_selection_gate"):
        payload = json.loads(Path(args.feature_log_path).read_text(encoding="utf-8"))
        selected_features = payload.get("selected_features", [])
        if len(selected_features) < args.min_selected_features:
            raise AssertionError(f"Too few selected features: {len(selected_features)}")
        log_event(
            LOGGER,
            "feature_selection_passed",
            selected_feature_count=len(selected_features),
            generated_feature_count=payload.get("generated_feature_count"),
        )


if __name__ == "__main__":
    main()
