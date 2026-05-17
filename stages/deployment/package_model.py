"""Package trained model artifacts and upload them for SageMaker registration."""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir
from common.logging_utils import configure_logging, log_event, timed_stage


LOGGER = logging.getLogger("stages.deployment.package")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package model.pt and metadata.json into a SageMaker model.tar.gz.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-artifact-s3-uri", required=True)
    parser.add_argument("--region", default="us-west-2")
    return parser.parse_args()


def create_model_archive(model_dir: Path, archive_path: Path) -> Path:
    required_files = ("model.pt", "metadata.json")
    for name in required_files:
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"Missing required model artifact: {model_dir / name}")

    ensure_dir(archive_path.parent)
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in required_files:
            archive.add(model_dir / name, arcname=name)
    return archive_path


def upload_archive(archive_path: Path, target_uri: str, region: str) -> None:
    if not target_uri.startswith("s3://"):
        target_path = Path(target_uri)
        ensure_dir(target_path.parent)
        target_path.write_bytes(archive_path.read_bytes())
        return

    parsed = urlparse(target_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError("Model artifact URI must be a full s3://bucket/key path.")

    import boto3

    boto3.client("s3", region_name=region).upload_file(str(archive_path), bucket, key)


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "package_model"):
        model_dir = Path(args.model_dir)
        with TemporaryDirectory() as temp_dir:
            archive_path = create_model_archive(model_dir, Path(temp_dir) / "model.tar.gz")
            upload_archive(archive_path, args.model_artifact_s3_uri, args.region)

        log_event(LOGGER, "model_artifact_packaged", model_artifact_uri=args.model_artifact_s3_uri)


if __name__ == "__main__":
    main()
