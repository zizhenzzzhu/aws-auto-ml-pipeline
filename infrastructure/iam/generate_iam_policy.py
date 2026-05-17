"""Generate least-privilege IAM policy templates for the ML pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_policy(
    bucket_arn: str,
    secret_arn: str,
    model_package_group_arn: str,
    log_group_arn: str,
    ecr_repository_arn: str,
) -> dict[str, object]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListPipelineBucket",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": bucket_arn,
            },
            {
                "Sid": "ReadWritePipelineObjects",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": f"{bucket_arn}/*",
            },
            {
                "Sid": "ReadDataPlatformSecret",
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue"],
                "Resource": secret_arn,
            },
            {
                "Sid": "WriteCloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": [log_group_arn, f"{log_group_arn}:*"],
            },
            {
                "Sid": "WriteCloudWatchMetrics",
                "Effect": "Allow",
                "Action": ["cloudwatch:PutMetricData"],
                "Resource": "*",
            },
            {
                "Sid": "RegisterSageMakerModel",
                "Effect": "Allow",
                "Action": ["sagemaker:CreateModelPackage", "sagemaker:DescribeModelPackageGroup"],
                "Resource": model_package_group_arn,
            },
            {
                "Sid": "PullEcrImages",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": ecr_repository_arn,
            },
            {
                "Sid": "AuthenticateEcr",
                "Effect": "Allow",
                "Action": ["ecr:GetAuthorizationToken"],
                "Resource": "*",
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an IAM policy template for pipeline execution.")
    parser.add_argument("--bucket-arn", default="arn:aws:s3:::your-bucket")
    parser.add_argument("--secret-arn", default="arn:aws:secretsmanager:region:account-id:secret:your-secret")
    parser.add_argument(
        "--model-package-group-arn",
        default="arn:aws:sagemaker:region:account-id:model-package-group/your-model-group",
    )
    parser.add_argument("--log-group-arn", default="arn:aws:logs:region:account-id:log-group:/aws/automl/*")
    parser.add_argument("--ecr-repository-arn", default="arn:aws:ecr:region:account-id:repository/your-repository")
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = build_policy(
        bucket_arn=args.bucket_arn,
        secret_arn=args.secret_arn,
        model_package_group_arn=args.model_package_group_arn,
        log_group_arn=args.log_group_arn,
        ecr_repository_arn=args.ecr_repository_arn,
    )
    payload = json.dumps(policy, indent=2)
    if args.output_file:
        Path(args.output_file).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
