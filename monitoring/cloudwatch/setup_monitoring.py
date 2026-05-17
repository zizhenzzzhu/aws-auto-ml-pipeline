"""Create CloudWatch log groups, alarms, and a dashboard for the ML pipeline."""

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


LOGGER = logging.getLogger("monitoring.cloudwatch")


def dashboard_body(namespace: str, pipeline_name: str, region: str) -> str:
    return json.dumps(
        {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "region": region,
                        "title": "Stage Duration",
                        "metrics": [
                            [
                                {
                                    "expression": (
                                        f"SEARCH('{{{namespace},PipelineName,StageName}} "
                                        "MetricName=\"StageDurationSeconds\"', "
                                        "'Average', 300)"
                                    ),
                                    "id": "stage_duration",
                                }
                            ]
                        ],
                        "stat": "Average",
                    },
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "region": region,
                        "title": "Evaluation Metrics",
                        "metrics": [
                            [namespace, "EvaluationLoss", "PipelineName", pipeline_name],
                            [".", "EvaluationAccuracy", ".", "."],
                        ],
                        "stat": "Average",
                    },
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "region": region,
                        "title": "Data Drift",
                        "metrics": [[namespace, "DriftPValue", "PipelineName", pipeline_name]],
                        "stat": "Minimum",
                    },
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "region": region,
                        "title": "Model Validation",
                        "metrics": [
                            [namespace, "CandidateAUC", "PipelineName", pipeline_name],
                            [".", "ChampionAUC", ".", "."],
                            [".", "InferenceLatencyMs", ".", "."],
                        ],
                        "stat": "Average",
                    },
                },
            ]
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up CloudWatch monitoring resources.")
    parser.add_argument("--pipeline-name", default="aws-auto-ml-pipeline")
    parser.add_argument("--namespace", default="AutoMLPipeline")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--log-group", default="/aws/automl/pipeline")
    parser.add_argument("--alarm-topic-arn", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    with timed_stage(LOGGER, "cloudwatch_setup"):
        dashboard_name = f"{args.pipeline_name}-dashboard"
        alarm_name = f"{args.pipeline_name}-high-evaluation-loss"
        dashboard = dashboard_body(args.namespace, args.pipeline_name, args.region)

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "log_group": args.log_group,
                        "dashboard_name": dashboard_name,
                        "dashboard_body": json.loads(dashboard),
                        "alarm_name": alarm_name,
                    },
                    indent=2,
                )
            )
            return

        import boto3

        logs = boto3.client("logs", region_name=args.region)
        cloudwatch = boto3.client("cloudwatch", region_name=args.region)

        try:
            logs.create_log_group(logGroupName=args.log_group)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass

        cloudwatch.put_dashboard(DashboardName=dashboard_name, DashboardBody=dashboard)

        alarm_actions = [args.alarm_topic_arn] if args.alarm_topic_arn else []
        cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            Namespace=args.namespace,
            MetricName="EvaluationLoss",
            Dimensions=[{"Name": "PipelineName", "Value": args.pipeline_name}],
            Statistic="Average",
            Period=300,
            EvaluationPeriods=1,
            Threshold=1.0,
            ComparisonOperator="GreaterThanThreshold",
            AlarmActions=alarm_actions,
            TreatMissingData="notBreaching",
        )
        log_event(LOGGER, "cloudwatch_resources_ready", dashboard_name=dashboard_name, alarm_name=alarm_name)


if __name__ == "__main__":
    main()
